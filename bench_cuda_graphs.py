"""
Does CUDA graph capture fix the launch-overhead bottleneck?

Three variants, same model, same prompt:
  1. DynamicCache, eager          — the baseline
  2. StaticCache,  eager          — fixed shapes, still one launch per op
  3. StaticCache + torch.compile(mode="reduce-overhead")
                                  — CUDA graphs: the whole decode step
                                    replayed as a single submission

Reports, per variant: median ITL, kernel launches for ONE decode step
(prefill excluded), GPU busy time, and the generated token ids so the
three can be checked against each other for correctness.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from torch.profiler import profile, ProfilerActivity


# ----------------------------------------------------------------- timing

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Clock:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        _sync(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync(self.device)
        self.dt = time.perf_counter() - self.t0
        return False


def make_prompt(tok, n_tokens):
    ids = tok(" the" * (n_tokens * 2))["input_ids"][:n_tokens]
    prompt = tok.decode(ids)
    actual = len(tok(prompt)["input_ids"])
    assert abs(actual - n_tokens) <= 2, f"got {actual}, wanted {n_tokens}"
    return prompt


# ----------------------------------------------------------------- variants
# Each returns (itl_list, token_ids). `fwd` is the callable used for the
# DECODE steps only; prefill always runs through the uncompiled model, so
# the two different input shapes never share a compiled graph.

@torch.inference_mode()
def run_dynamic(model, input_ids, gen_tokens, device, **_):
    attn = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    cache = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
    ids, itl = [next_id.item()], []

    for _ in range(gen_tokens - 1):
        attn = torch.cat([attn, torch.ones_like(next_id)], dim=1)
        with Clock(device) as c:
            out = model(input_ids=next_id, attention_mask=attn,
                        past_key_values=cache, use_cache=True)
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cache = out.past_key_values
        itl.append(c.dt)
        ids.append(next_id.item())
    return itl, ids


@torch.inference_mode()
def run_static(model, input_ids, gen_tokens, device, cache=None, fwd=None):
    """Shared by variants 2 and 3 — only `fwd` differs.

    CUDA graphs reuse a fixed output buffer, so `next_id` MUST be cloned
    before the next replay overwrites it.
    """
    fwd = fwd or model.forward
    cache.reset()
    n_prompt = input_ids.shape[1]

    pos = torch.arange(n_prompt, device=device)
    out = model(input_ids=input_ids, past_key_values=cache,
                cache_position=pos, use_cache=True)
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True).clone()
    ids, itl = [next_id.item()], []

    for k in range(gen_tokens - 1):
        pos = torch.tensor([n_prompt + k], device=device)
        with Clock(device) as c:
            out = fwd(input_ids=next_id, past_key_values=cache,
                      cache_position=pos, use_cache=True)
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True).clone()
        itl.append(c.dt)
        ids.append(next_id.item())
    return itl, ids


# ----------------------------------------------------------------- profiling

@torch.inference_mode()
def profile_one_step(model, input_ids, device, cache=None, fwd=None):
    """Launches + GPU time for a SINGLE decode step, prefill excluded."""
    if cache is None:                                  # dynamic path
        attn = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
        c = out.past_key_values
        nid = out.logits[:, -1, :].argmax(-1, keepdim=True)
        attn = torch.cat([attn, torch.ones_like(nid)], dim=1)
        step = lambda: model(input_ids=nid, attention_mask=attn,
                             past_key_values=c, use_cache=True)
    else:                                              # static / compiled
        fwd = fwd or model.forward
        cache.reset()
        n = input_ids.shape[1]
        out = model(input_ids=input_ids, past_key_values=cache,
                    cache_position=torch.arange(n, device=device), use_cache=True)
        nid = out.logits[:, -1, :].argmax(-1, keepdim=True).clone()
        pos = torch.tensor([n], device=device)
        step = lambda: fwd(input_ids=nid, past_key_values=cache,
                           cache_position=pos, use_cache=True)

    step()                                             # settle before profiling
    _sync(device)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
    evts = prof.key_averages()
    launches = sum(e.count for e in evts if e.key == "cudaLaunchKernel")
    graph_launches = sum(e.count for e in evts if "GraphLaunch" in e.key)
    gpu_us = sum(e.self_device_time_total for e in evts
                 if e.device_type == torch.autograd.DeviceType.CUDA)
    return launches, graph_launches, gpu_us


# ----------------------------------------------------------------- main

def main(model_id="EleutherAI/pythia-410m", prompt_tokens=512, gen_tokens=64,
         dtype=torch.bfloat16, device_str="cuda", warmup=3, reps=5):
    device = torch.device(device_str)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()

    prompt = make_prompt(tok, prompt_tokens)
    input_ids = tok(prompt, return_tensors="pt").to(device)["input_ids"]

    max_len = prompt_tokens + gen_tokens + 8
    cache_eager = StaticCache(config=model.config, max_cache_len=max_len)
    cache_graph = StaticCache(config=model.config, max_cache_len=max_len)

    # reduce-overhead == CUDA graphs. Needs StaticCache (fixed shapes) and a
    # longer warmup: first call compiles, next few capture and replay.
    compiled = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    variants = [
        ("DynamicCache eager", run_dynamic, dict()),
        ("StaticCache eager",  run_static,  dict(cache=cache_eager)),
        ("StaticCache +graph", run_static,  dict(cache=cache_graph, fwd=compiled)),
    ]

    results, token_ids = {}, {}
    for name, fn, kw in variants:
        w = warmup if "graph" not in name else max(warmup, 8)
        for _ in range(w):
            fn(model, input_ids, gen_tokens, device, **kw)

        meds = []
        for _ in range(reps):
            itl, ids = fn(model, input_ids, gen_tokens, device, **kw)
            s = sorted(itl)
            meds.append(s[len(s) // 2] * 1e3)
        token_ids[name] = ids
        results[name] = sorted(meds)[len(meds) // 2]

    print(f"{'variant':22s} {'ITL':>9s} {'tok/s':>8s} {'launches':>9s} "
          f"{'graphs':>7s} {'GPU us':>8s}")
    for name, fn, kw in variants:
        pkw = {k: v for k, v in kw.items() if k in ("cache", "fwd")}
        launches, graphs, gpu = profile_one_step(model, input_ids, device, **pkw)
        itl = results[name]
        print(f"{name:22s} {itl:7.3f}ms {1e3/itl:8.1f} {launches:9d} "
              f"{graphs:7d} {gpu:8.0f}")

    base = token_ids["DynamicCache eager"]
    for name, ids in token_ids.items():
        status = "match" if ids == base else "DIFFERS"
        print(f"correctness  {name:22s} {status}")


if __name__ == "__main__":
    main()