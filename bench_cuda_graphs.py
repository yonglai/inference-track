"""
Where does batch-1 decode time go, and what does removing it cost in accuracy?

Four variants, same model, same prompt, each adding one optimization:

  1. DynamicCache eager     — baseline; cache grows by torch.cat every step
  2. StaticCache eager      — preallocated cache, fixed shapes; still one
                              kernel launch per op
  3. StaticCache compiled   — torch.compile(mode="default"): Inductor fuses
                              ops into fewer kernels, no CUDA graphs
  4. StaticCache +graph     — torch.compile(mode="reduce-overhead"): the same
                              fused kernels, captured and replayed as a
                              single CUDA graph launch

Performance, per variant: median ITL over `reps` runs, plus kernel launches,
graph launches, and GPU busy time for ONE decode step (prefill excluded).

Correctness: every variant is fed the same token sequence (teacher forcing,
so one flipped argmax cannot cascade) and its logits compared against

  - bf16 DynamicCache eager (the baseline), and
  - an fp32 copy of the model — not ground truth, but ~16 more mantissa bits,
    enough to judge which bf16 path strays further.

Metrics: per-step KL divergence (weights each token by its probability),
max logit difference over the reference's top-10 tokens, and argmax flips.
The graph path is also run twice to confirm replay is deterministic.

Inductor precision: by default Inductor may round fused intermediates
differently from eager, which rounds to bf16 after every op. Runs set
torch._inductor.config.emulate_precision_casts = True unless --no-emulate
is passed. The flag is read at compile time, so the two settings need
separate runs.

Usage:
    uv run python bench_cuda_graphs.py                # emulate casts (default)
    uv run python bench_cuda_graphs.py --no-emulate   # Inductor default rounding

Writes cuda_graphs_emulate.json or cuda_graphs_noemulate.json.
"""
import json
import sys
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
    """Static-cache decode loop, shared by variants 2, 3 and 4.

    Only `fwd` differs: model.forward (eager), or a torch.compile'd wrapper
    with or without CUDA graphs. Prefill always runs uncompiled.

    `cache_position` is required: a StaticCache is always max_cache_len long,
    so the model can't infer the current position from the cache's shape.

    No .clone() on next_id: argmax allocates a fresh tensor, so it never
    aliases the graph-owned output buffer that the next replay overwrites.
    """
    fwd = fwd or model.forward
    cache.reset()
    n_prompt = input_ids.shape[1]

    pos = torch.arange(n_prompt, device=device)
    out = model(input_ids=input_ids, past_key_values=cache,
                cache_position=pos, use_cache=True)
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
    ids, itl = [next_id.item()], []

    for k in range(gen_tokens - 1):
        pos = torch.tensor([n_prompt + k], device=device)
        with Clock(device) as c:
            out = fwd(input_ids=next_id, past_key_values=cache,
                      cache_position=pos, use_cache=True)
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
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
        nid = out.logits[:, -1, :].argmax(-1, keepdim=True)
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

@torch.inference_mode()
def logits_trace(model, input_ids, forced, device, cache=None, fwd=None):
    """Feed a fixed token sequence; return the logits row at each step."""
    rows = []
    if cache is None:
        attn = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
        c = out.past_key_values
        rows.append(out.logits[0, -1].float().cpu())
        for t in forced[:-1]:
            nid = torch.tensor([[t]], device=device)
            attn = torch.cat([attn, torch.ones_like(nid)], dim=1)
            out = model(input_ids=nid, attention_mask=attn,
                        past_key_values=c, use_cache=True)
            c = out.past_key_values
            rows.append(out.logits[0, -1].float().cpu())
    else:
        fwd = fwd or model.forward
        cache.reset()
        n = input_ids.shape[1]
        out = model(input_ids=input_ids, past_key_values=cache,
                    cache_position=torch.arange(n, device=device), use_cache=True)
        rows.append(out.logits[0, -1].float().cpu())
        for k, t in enumerate(forced[:-1]):
            nid = torch.tensor([[t]], device=device)
            out = fwd(input_ids=nid, past_key_values=cache,
                      cache_position=torch.tensor([n + k], device=device),
                      use_cache=True)
            rows.append(out.logits[0, -1].float().cpu())
    return torch.stack(rows)

def kl_report(name, ref, got):
    p = ref.log_softmax(-1)
    q = got.log_softmax(-1)
    kl = (p.exp() * (p - q)).sum(-1)
    top10 = ref.topk(10, dim=-1).indices
    top_diff = (got.gather(-1, top10) - ref.gather(-1, top10)).abs().max(-1).values
    flips = (got.argmax(-1) != ref.argmax(-1)).nonzero().flatten().tolist()
    print(f"{name:22s} KL max {kl.max():.2e} @step {kl.argmax().item():2d}   "
          f"mean {kl.mean():.2e}   top-10 |Δ| {top_diff.max():.4f}   flips {flips}")
    return kl

# ----------------------------------------------------------------- main

def main(model_id="EleutherAI/pythia-410m", prompt_tokens=512, gen_tokens=64,
         dtype=torch.bfloat16, device_str="cuda", warmup=3, reps=5,
         emulate_casts=True):
    device = torch.device(device_str)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()

    prompt = make_prompt(tok, prompt_tokens)
    input_ids = tok(prompt, return_tensors="pt").to(device)["input_ids"]

    max_len = prompt_tokens + gen_tokens + 8
    cache_eager = StaticCache(config=model.config, max_cache_len=max_len)
    cache_graph = StaticCache(config=model.config, max_cache_len=max_len)

    torch._inductor.config.emulate_precision_casts = emulate_casts
    print(f"torch {torch.__version__}   emulate_precision_casts={emulate_casts}")

    # reduce-overhead == CUDA graphs. Needs StaticCache (fixed shapes).
    compiled = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    # Inductor kernels, but NO CUDA graph capture — isolates the two
    compiled_nograph = torch.compile(model.forward, mode="default", fullgraph=True)
    cache_nograph = StaticCache(config=model.config, max_cache_len=max_len)

    variants = [
        ("DynamicCache eager",   run_dynamic, dict()),
        ("StaticCache eager",    run_static,  dict(cache=cache_eager)),
        ("StaticCache compiled", run_static,  dict(cache=cache_nograph, fwd=compiled_nograph)),
        ("StaticCache +graph",   run_static,  dict(cache=cache_graph, fwd=compiled)),
    ]

    results, token_ids = {}, {}
    for name, fn, kw in variants:
        for _ in range(warmup):
            fn(model, input_ids, gen_tokens, device, **kw)

        meds = []
        for _ in range(reps):
            itl, ids = fn(model, input_ids, gen_tokens, device, **kw)
            s = sorted(itl)
            meds.append(s[len(s) // 2] * 1e3)
        token_ids[name] = ids
        results[name] = sorted(meds)[len(meds) // 2]

    profile_rows = {}
    print(f"{'variant':22s} {'ITL':>9s} {'tok/s':>8s} {'launches':>9s} "
          f"{'graphs':>7s} {'GPU us':>8s}")
    for name, fn, kw in variants:
        launches, graphs, gpu = profile_one_step(model, input_ids, device, **kw)
        profile_rows[name] = {"launches": launches, "graph_launches": graphs,
                              "gpu_us": gpu}
        itl = results[name]
        print(f"{name:22s} {itl:7.3f}ms {1e3/itl:8.1f} {launches:9d} "
              f"{graphs:7d} {gpu:8.0f}")

    forced = token_ids["DynamicCache eager"]

    # fp32 reference: ~16 more bits of precision than bf16. Not "truth",
    # but close enough to judge which bf16 path strays further from it.
    model32 = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32).to(device).eval()
    truth = logits_trace(model32, input_ids, forced, device)
    del model32
    torch.cuda.empty_cache()

    traces = {name: logits_trace(model, input_ids, forced, device, **kw)
              for name, _, kw in variants}

    g2 = logits_trace(model, input_ids, forced, device,
                      cache=cache_graph, fwd=compiled)
    rerun = (traces["StaticCache +graph"] - g2).abs().max().item()
    print(f"\ngraph path, run vs rerun: max |Δlogit| {rerun:.4f}")

    print("\n--- each variant vs bf16 DynamicCache eager ---")
    for n, t in traces.items():
        if n != "DynamicCache eager":
            kl_report(n, traces["DynamicCache eager"], t)

    print("\n--- each bf16 variant vs fp32 reference ---")
    kls = {n: kl_report(n, truth, t) for n, t in traces.items()}

    print("\n--- per-step KL vs fp32 ---")
    for n, kl in kls.items():
        print(f"{n:22s}", " ".join(f"{x:.0e}" for x in kl.tolist()))

    with open(f"cuda_graphs_{'emulate' if emulate_casts else 'noemulate'}.json", "w") as f:
        json.dump({"torch": torch.__version__, "model": model_id,
                   "prompt_tokens": prompt_tokens, "gen_tokens": gen_tokens,
                   "emulate_casts": emulate_casts,
                   "itl_ms": results,
                   "profile": profile_rows,
                   "graph_rerun_max_abs_diff": rerun,
                   "kl_vs_fp32": {n: {"mean": k.mean().item(), "max": k.max().item(),
                                      "argmax_step": k.argmax().item()}
                                  for n, k in kls.items()}},
                  f, indent=2)

if __name__ == "__main__":
    main(emulate_casts="--no-emulate" not in sys.argv)