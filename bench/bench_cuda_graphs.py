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
Timing does not depend on prompt content, only on its length.

Correctness: every variant is fed the same token sequence (teacher forcing,
so one flipped argmax cannot cascade) and its logits compared against

  - bf16 DynamicCache eager (the baseline), and
  - an fp32 copy of the model — not ground truth, but ~16 more mantissa bits,
    enough to judge which bf16 path strays further.

Metrics: per-step KL divergence (weights each token by its probability),
max logit difference over the reference's top-10 tokens, argmax flips, and
for each flip the reference's top-2 gap measured in bf16 ULPs. The graph
path is also run twice to confirm replay is deterministic.

Prompt: 512 repeated " the" tokens by default. That input makes attention
unusually sensitive to accumulation order (investigate_fusion.py showed
correct attention kernels disagreeing by 25x in KL on it), so accuracy
numbers from it are not characteristic of the model. --real-prompt uses
ordinary prose of the same length instead.

Inductor precision: runs set torch._inductor.config.emulate_precision_casts
= True unless --no-emulate is passed. The flag is read at compile time, so
the two settings need separate runs.

Usage:
    uv run python bench_cuda_graphs.py                              # " the", emulate
    uv run python bench_cuda_graphs.py --no-emulate
    uv run python bench_cuda_graphs.py --real-prompt                # prose, emulate
    uv run python bench_cuda_graphs.py --real-prompt --no-emulate

Writes cuda_graphs_<the|real>_<emulate|noemulate>.json.
"""
import json
import math
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache


# ----------------------------------------------------------------- timing

from pathlib import Path
RESULTS = Path(__file__).resolve().parent.parent / "results"

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


# ----------------------------------------------------------------- prompts

def make_prompt(tok, n_tokens):
    """n_tokens repetitions of " the" -- controls length, content irrelevant to timing."""
    ids = tok(" the" * (n_tokens * 2))["input_ids"][:n_tokens]
    prompt = tok.decode(ids)
    actual = len(tok(prompt)["input_ids"])
    assert abs(actual - n_tokens) <= 2, f"got {actual}, wanted {n_tokens}"
    return prompt


# Original prose, written for these scripts. Varied vocabulary so attention
# sees a realistic spread of keys, unlike 512 copies of " the".
REAL_TEXT = """\
The town of Harrowmere grew up where two slow rivers met, and for most of its \
history the rivers decided everything. They decided where the mills stood, which \
families prospered, and which fields flooded every spring. The oldest maps show a \
ferry crossing near the chapel, a timber footbridge further east, and a row of \
warehouses built on stilts along the northern bank. Merchants arrived by barge in \
the autumn, carrying salt, iron nails, bolts of wool and barrels of pickled fish, \
and they left with grain, leather and the pale clay that local potters dug from \
the riverbed.

By the middle of the last century the ferry had been replaced by a stone bridge \
with three arches, and the warehouses had become apartments. The mills fell silent \
one after another. The largest was converted into a library, and the smaller ones \
into workshops for carpenters, printers and a family that repaired clocks. Visitors \
who came for the market on Saturdays often stayed to walk the towpath, which \
follows the southern river for nearly eleven kilometres before it reaches the \
reservoir.

The reservoir itself was controversial. Engineers argued that it would end the \
spring floods, while farmers worried that it would starve the lower fields of the \
silt that made them fertile. Both groups turned out to be partly right. The floods \
became rare, but yields in the lowest meadows declined slowly over twenty years, \
and several farms switched from barley to grazing sheep. A committee was formed to \
study the problem; it met every second Tuesday, published a long report, and \
recommended releasing controlled pulses of water each March.

Today the town depends on a mixture of tourism, light manufacturing and commuting. \
A regional train stops four times a day, and the station cafe is known for its \
lemon cake and its unreliable heating. The primary school has two hundred pupils, \
a small orchard, and a weather station that the older children maintain. Every \
June the town holds a regatta on the confluence, with races for rowing boats, \
canoes and a final contest in which teams build rafts from barrels and planks and \
try to cross without sinking.

Historians who study Harrowmere tend to agree on one point: the town survived \
because it adapted slowly rather than quickly. Each change, from the bridge to the \
reservoir, was argued over for years before it happened, and by the time it \
arrived most people had already found a way to live with it.
"""


def make_real_prompt(tok, n_tokens):
    """REAL_TEXT truncated to n_tokens; repeated only if it runs short."""
    ids = tok(REAL_TEXT)["input_ids"]
    while len(ids) < n_tokens:
        ids = ids + ids
    prompt = tok.decode(ids[:n_tokens])
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


# ----------------------------------------------------------------- accuracy

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


def bf16_ulp(x):
    """Spacing between adjacent bf16 values at magnitude |x| (7 stored mantissa bits)."""
    x = abs(x)
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x > 0 else 2.0 ** -133


def kl_report(name, ref, got):
    """KL, top-10 logit difference, and argmax flips of `got` against `ref`.

    For each flip, reports the reference's top-2 gap in bf16 ULPs: a flip at
    0-2 ULPs is a near-tie broken differently; a flip at a large gap would be
    a genuine disagreement.
    """
    p = ref.log_softmax(-1)
    q = got.log_softmax(-1)
    kl = (p.exp() * (p - q)).sum(-1)
    top10 = ref.topk(10, dim=-1).indices
    top_diff = (got.gather(-1, top10) - ref.gather(-1, top10)).abs().max(-1).values
    flips = (got.argmax(-1) != ref.argmax(-1)).nonzero().flatten().tolist()

    flip_ulps = []
    for s in flips:
        v = ref[s].topk(2).values
        flip_ulps.append(round((v[0] - v[1]).item() / bf16_ulp(v[1].item()), 2))

    print(f"{name:22s} KL max {kl.max():.2e} @step {kl.argmax().item():2d}   "
          f"mean {kl.mean():.2e}   top-10 |Δ| {top_diff.max():.4f}   "
          f"flips {flips}  gap(ULP) {flip_ulps}")
    return kl, flips, flip_ulps


# ----------------------------------------------------------------- main

def main(model_id="EleutherAI/pythia-410m", prompt_tokens=512, gen_tokens=64,
         dtype=torch.bfloat16, device_str="cuda", warmup=3, reps=5,
         emulate_casts=True, real_prompt=False):
    device = torch.device(device_str)
    prompt_kind = "real" if real_prompt else "the"

    torch._inductor.config.emulate_precision_casts = emulate_casts
    print(f"torch {torch.__version__}   emulate_precision_casts={emulate_casts}   "
          f"prompt={prompt_kind}")

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()

    prompt = (make_real_prompt if real_prompt else make_prompt)(tok, prompt_tokens)
    input_ids = tok(prompt, return_tensors="pt").to(device)["input_ids"]

    max_len = prompt_tokens + gen_tokens + 8
    cache_eager = StaticCache(config=model.config, max_cache_len=max_len)
    cache_nograph = StaticCache(config=model.config, max_cache_len=max_len)
    cache_graph = StaticCache(config=model.config, max_cache_len=max_len)

    # Inductor kernels, but NO CUDA graph capture -- isolates fusion from replay.
    compiled_nograph = torch.compile(model.forward, mode="default", fullgraph=True)
    # reduce-overhead == CUDA graphs. Needs StaticCache (fixed shapes).
    compiled = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    variants = [
        ("DynamicCache eager",   run_dynamic, dict()),
        ("StaticCache eager",    run_static,  dict(cache=cache_eager)),
        ("StaticCache compiled", run_static,  dict(cache=cache_nograph, fwd=compiled_nograph)),
        ("StaticCache +graph",   run_static,  dict(cache=cache_graph, fwd=compiled)),
    ]

    # ---- timing
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

    # ---- profiling
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

    # ---- accuracy
    forced = token_ids["DynamicCache eager"]
    print(f"\ngenerated (DynamicCache eager): {tok.decode(forced)!r}")

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
    vs_dynamic = {}
    for n, t in traces.items():
        if n != "DynamicCache eager":
            k, flips, ulps = kl_report(n, traces["DynamicCache eager"], t)
            vs_dynamic[n] = {"kl_mean": k.mean().item(), "kl_max": k.max().item(),
                             "flips": flips, "flip_gap_ulp": ulps}

    print("\n--- each bf16 variant vs fp32 reference ---")
    vs_fp32 = {}
    kls = {}
    for n, t in traces.items():
        k, flips, ulps = kl_report(n, truth, t)
        kls[n] = k
        vs_fp32[n] = {"kl_mean": k.mean().item(), "kl_max": k.max().item(),
                      "kl_argmax_step": k.argmax().item(),
                      "flips": flips, "flip_gap_ulp": ulps}

    print("\n--- per-step KL vs fp32 ---")
    for n, k in kls.items():
        print(f"{n:22s}", " ".join(f"{x:.0e}" for x in k.tolist()))


    RESULTS.mkdir(exist_ok=True)
    fname = RESULTS / f"cuda_graphs_{prompt_kind}_{'emulate' if emulate_casts else 'noemulate'}.json"
    with open(fname, "w") as f:
        json.dump({"torch": torch.__version__, "model": model_id,
                   "prompt": prompt_kind,
                   "prompt_tokens": prompt_tokens, "gen_tokens": gen_tokens,
                   "emulate_casts": emulate_casts,
                   "itl_ms": results,
                   "profile": profile_rows,
                   "graph_rerun_max_abs_diff": rerun,
                   "vs_dynamic": vs_dynamic,
                   "vs_fp32": vs_fp32},
                  f, indent=2)
    print(f"\nwrote {fname}")


if __name__ == "__main__":
    main(emulate_casts="--no-emulate" not in sys.argv,
         real_prompt="--real-prompt" in sys.argv)