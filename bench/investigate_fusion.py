"""
Fusion-accuracy investigation: where, and in which operation, does the
Inductor-compiled decode path lose accuracy?

Findings so far (pythia-410m, bf16, 512-token repeated " the" prompt):
  - The extra error first appears at the OUTPUT OF LAYER 5, at every step
    where it appears at all. Labels verified with a hook.
  - It is not caused by large activations: max |h| at layer 5's output is
    modest (14.4 at step 1, 8.6 at step 54); large values appear later.
  - Every non-attention piece of layer 5 (LayerNorms, GELU, MLP, residual
    add, residual + LayerNorm) is clean when isolated. By elimination the
    problem is in the ATTENTION BRANCH.
  - There are two separate effects:
      A. step 1: compiled only; fixed by emulate_precision_casts
      B. step 54: shared by compiled AND DynamicCache eager; NOT fixed by
         the flag; StaticCache eager does not have it.

This run tests whether effect B follows the ATTENTION KERNEL, and whether
the degenerate prompt is amplifying both effects.

STAGE 1 -- per-layer comparison against fp32.
  Teacher-forces four paths to the same position and reports each layer's
  relative error ||x - ref|| / ||ref|| against fp32:
    DynamicCache eager (bf16) | StaticCache eager (bf16) |
    StaticCache compiled (bf16, mode="default") | fp32 DynamicCache (ref)

STAGE 4 -- attention backend sweep (eager paths only).
  Re-runs both eager paths with torch's SDPA forced to each backend in turn
  (FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH) and reports layer L's output
  error and logits KL. If forcing StaticCache eager onto the backend that
  DynamicCache / compiled use reproduces their layer-L error, effect B is
  attention kernel selection, not fusion. A backend that cannot handle the
  inputs (e.g. flash with an explicit mask) raises and is reported "n/a".
  The forced backend applies to prefill as well as decode.

STAGE 3 -- isolate the non-attention pieces of layer L (as before).
  Captures layer L's input and branch outputs from the fp32 run, rounds each
  to bf16 once, and runs small candidate functions eager / compiled / fp32
  on those identical inputs.

AFTER ALL STEPS -- which attention kernel does each path actually run?
  Profiles ONE decode step (prefill excluded) per path and lists kernels
  whose names mention flash / fmha / attention / efficient.

Usage:
    uv run python investigate_fusion.py                          # " the" prompt
    uv run python investigate_fusion.py --emulate
    uv run python investigate_fusion.py --real-prompt            # ordinary prose
    uv run python investigate_fusion.py --real-prompt --emulate

Writes investigate_<the|real>_<emulate|noemulate>.json.
"""
import json
import sys
from contextlib import contextmanager

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from bench_cuda_graphs import _sync, make_prompt, run_dynamic


# ----------------------------------------------------------------- prompts

# Original prose, written for this script. Varied vocabulary so attention
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

from pathlib import Path
RESULTS = Path(__file__).resolve().parent.parent / "results"


def make_real_prompt(tok, n_tokens):
    """REAL_TEXT truncated to n_tokens; repeated only if it runs short."""
    ids = tok(REAL_TEXT)["input_ids"]
    while len(ids) < n_tokens:
        ids = ids + ids
    prompt = tok.decode(ids[:n_tokens])
    actual = len(tok(prompt)["input_ids"])
    assert abs(actual - n_tokens) <= 2, f"got {actual}, wanted {n_tokens}"
    return prompt


# ----------------------------------------------------------------- tracing

HS_KW = dict(use_cache=True, output_hidden_states=True)


@torch.inference_mode()
def trace_step(model, input_ids, forced, device, step, cache=None, fwd=None):
    """Teacher-force to `step`; return (hidden states, logits) for the last position.

    Step 0 is the prefill; step k is after feeding forced[0..k-1], matching
    bench_cuda_graphs.logits_trace. output_hidden_states=True is passed on
    EVERY call so one compiled graph runs throughout.
    """
    if cache is None:                                   # DynamicCache path
        attn = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attn, **HS_KW)
        c = out.past_key_values
        for t in forced[:step]:
            nid = torch.tensor([[t]], device=device)
            attn = torch.cat([attn, torch.ones_like(nid)], dim=1)
            out = model(input_ids=nid, attention_mask=attn, past_key_values=c, **HS_KW)
            c = out.past_key_values
    else:                                               # StaticCache: eager or compiled
        fwd = fwd or model.forward
        cache.reset()
        n = input_ids.shape[1]
        out = model(input_ids=input_ids, past_key_values=cache,
                    cache_position=torch.arange(n, device=device), **HS_KW)
        for k, t in enumerate(forced[:step]):
            nid = torch.tensor([[t]], device=device)
            out = fwd(input_ids=nid, past_key_values=cache,
                      cache_position=torch.tensor([n + k], device=device), **HS_KW)

    hidden = [h[0, -1].float().cpu() for h in out.hidden_states]
    return hidden, out.logits[0, -1].float().cpu()


@contextmanager
def capture_layer(model, L):
    """Record layer L's input and its two branch outputs on every forward call.

    Each call overwrites the previous one, so after a traced run the dict
    holds the values from the LAST call -- the target step.
    """
    store = {}
    layer = model.gpt_neox.layers[L]

    def pre(module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        store["x"] = h[0, -1].detach().clone()

    def post(name):
        def hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            store[name] = o[0, -1].detach().clone()
        return hook

    handles = [
        layer.register_forward_pre_hook(pre, with_kwargs=True),
        layer.attention.register_forward_hook(post("attn")),
        layer.mlp.register_forward_hook(post("mlp")),
    ]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


# ----------------------------------------------------------------- metrics

def kl(ref_logits, got_logits):
    p = ref_logits.log_softmax(-1)
    q = got_logits.log_softmax(-1)
    return (p.exp() * (p - q)).sum().item()


def rel(x, ref):
    return ((x - ref).norm() / ref.norm()).item()


# ----------------------------------------------------------------- stage 4: backends

BACKENDS = (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH)


def backend_sweep(model, input_ids, forced, device, step, cache_eager, ref_h, ref_l, L):
    """Both eager paths under each forced SDPA backend."""
    out_idx = L + 1                                     # hs index of layer L's output
    results = {}
    for backend in BACKENDS:
        row = {}
        for path, cache in (("dynamic", None), ("static", cache_eager)):
            try:
                with sdpa_kernel(backend):
                    h, lg = trace_step(model, input_ids, forced, device, step, cache=cache)
                row[path] = {"layer_out_rel": rel(h[out_idx], ref_h[out_idx]),
                             "kl": kl(ref_l, lg)}
            except RuntimeError as err:
                row[path] = {"error": str(err).splitlines()[0][:80]}
        results[backend.name] = row
    return results


def fmt_cell(cell):
    if "error" in cell:
        return f"{'n/a':>11s} {'':>10s}"
    return f"{cell['layer_out_rel']:11.2e} {cell['kl']:10.2e}"


# ----------------------------------------------------------------- stage 3: isolation

def candidates(layer, nxt):
    """Pieces of one layer, as (fn, input names). Addition order matches
    GPTNeoXLayer's parallel residual: mlp_output + attn_output + hidden_states."""
    def ln1(x):
        return layer.input_layernorm(x)

    def ln2(x):
        return layer.post_attention_layernorm(x)

    def mlp_up_gelu(x):
        m = layer.mlp
        return m.act(m.dense_h_to_4h(layer.post_attention_layernorm(x)))

    def mlp(x):
        return layer.mlp(layer.post_attention_layernorm(x))

    def residual(x, a, m):
        return m + a + x

    def residual_ln(x, a, m):
        h = m + a + x
        return nxt.input_layernorm(h), nxt.post_attention_layernorm(h)

    return {
        "ln1":         (ln1,         ("x",)),
        "ln2":         (ln2,         ("x",)),
        "mlp_up_gelu": (mlp_up_gelu, ("x",)),
        "mlp":         (mlp,         ("x",)),
        "residual":    (residual,    ("x", "a", "m")),
        "residual_ln": (residual_ln, ("x", "a", "m")),
    }


def build_pieces(model, model32, L):
    """(name, eager fn, compiled fn, fp32 fn, arg names), compiled once for all steps."""
    cb = candidates(model.gpt_neox.layers[L], model.gpt_neox.layers[L + 1])
    c32 = candidates(model32.gpt_neox.layers[L], model32.gpt_neox.layers[L + 1])
    return [(name, fn, torch.compile(fn), c32[name][0], args)
            for name, (fn, args) in cb.items()]


def as_tuple(out):
    return out if isinstance(out, tuple) else (out,)


@torch.inference_mode()
def isolate(pieces, captured, device, L):
    """Run each piece eager / compiled / fp32 on identical bf16-rounded inputs."""
    names = {"x": "x", "a": "attn", "m": "mlp"}
    bf = {k: captured[v].to(device, torch.bfloat16).view(1, 1, -1) for k, v in names.items()}
    f32 = {k: v.float() for k, v in bf.items()}          # same values, wider container

    rows = {}
    print(f"\n  layer {L} pieces, identical bf16-rounded inputs")
    print(f"  {'piece':16s} {'eager rel':>11s} {'compiled rel':>13s} {'ratio':>7s}")
    for name, fn, fn_c, fn32, args in pieces:
        ref = as_tuple(fn32(*[f32[a] for a in args]))
        eag = as_tuple(fn(*[bf[a] for a in args]))
        comp = as_tuple(fn_c(*[bf[a] for a in args]))
        for j, (r, e, c) in enumerate(zip(ref, eag, comp)):
            r, e, c = r.float().cpu(), e.float().cpu(), c.float().cpu()
            re_, rc_ = rel(e, r), rel(c, r)
            ratio = rc_ / max(re_, 1e-12)
            label = name if len(ref) == 1 else f"{name}[{j}]"
            rows[label] = {"eager_rel": re_, "compiled_rel": rc_}
            flag = "  <--" if ratio > 2 else ""
            print(f"  {label:16s} {re_:11.2e} {rc_:13.2e} {ratio:7.2f}{flag}")
    return rows


# ----------------------------------------------------------------- attention kernels

ATTN_KEYS = ("flash", "fmha", "attention", "efficient")


@torch.inference_mode()
def decode_step_attn_kernels(model, input_ids, forced, device, cache=None, fwd=None):
    """Attention-related kernel names for ONE decode step, prefill excluded."""
    nid = torch.tensor([[forced[0]]], device=device)
    if cache is None:
        attn = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attn, **HS_KW)
        c = out.past_key_values
        attn = torch.cat([attn, torch.ones_like(nid)], dim=1)
        step = lambda: model(input_ids=nid, attention_mask=attn, past_key_values=c, **HS_KW)
    else:
        fwd = fwd or model.forward
        cache.reset()
        n = input_ids.shape[1]
        model(input_ids=input_ids, past_key_values=cache,
              cache_position=torch.arange(n, device=device), **HS_KW)
        pos = torch.tensor([n], device=device)
        step = lambda: fwd(input_ids=nid, past_key_values=cache, cache_position=pos, **HS_KW)

    _sync(device)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
    return sorted({e.key[:90] for e in prof.key_averages()
                   if e.device_type == torch.autograd.DeviceType.CUDA
                   and any(s in e.key.lower() for s in ATTN_KEYS)})


# ----------------------------------------------------------------- main

def main(model_id="EleutherAI/pythia-410m", prompt_tokens=512, gen_tokens=64,
         steps=(1, 54), isolate_layer=5, emulate=False, real_prompt=False,
         device_str="cuda"):
    device = torch.device(device_str)
    torch._inductor.config.emulate_precision_casts = emulate
    prompt_kind = "real" if real_prompt else "the"
    print(f"torch {torch.__version__}   emulate_precision_casts={emulate}   "
          f"prompt={prompt_kind}   isolating layer {isolate_layer}")

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16).to(device).eval()
    model32 = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32).to(device).eval()

    prompt = (make_real_prompt if real_prompt else make_prompt)(tok, prompt_tokens)
    input_ids = tok(prompt, return_tensors="pt").to(device)["input_ids"]

    # Forced sequence: bf16 DynamicCache eager, greedy -- same rule as the benchmark.
    _, forced = run_dynamic(model, input_ids, gen_tokens, device)

    max_len = prompt_tokens + gen_tokens + 8
    cache_eager = StaticCache(config=model.config, max_cache_len=max_len)
    cache_comp = StaticCache(config=model.config, max_cache_len=max_len)
    compiled = torch.compile(model.forward, mode="default", fullgraph=True)

    L = isolate_layer
    n_layers = model.config.num_hidden_layers
    pieces = build_pieces(model, model32, L)
    report = {"torch": torch.__version__, "model": model_id, "prompt": prompt_kind,
              "prompt_tokens": prompt_tokens, "gen_tokens": gen_tokens,
              "emulate_casts": emulate, "isolate_layer": L, "steps": {}}

    for step in steps:
        # ---- stage 1
        with capture_layer(model32, L) as captured:
            ref_h, ref_l = trace_step(model32, input_ids, forced, device, step)
        d_h, d_l = trace_step(model, input_ids, forced, device, step)
        e_h, e_l = trace_step(model, input_ids, forced, device, step, cache=cache_eager)
        c_h, c_l = trace_step(model, input_ids, forced, device, step,
                              cache=cache_comp, fwd=compiled)

        kls = {"dynamic": kl(ref_l, d_l), "static": kl(ref_l, e_l),
               "compiled": kl(ref_l, c_l)}
        print(f"\n=== step {step} ===")
        print(f"logits KL vs fp32:  dynamic {kls['dynamic']:.2e}   "
              f"static {kls['static']:.2e}   compiled {kls['compiled']:.2e}")

        drift = (captured["x"].float().cpu() - ref_h[L]).abs().max().item()
        print(f"label check: max |layer {L} input - hs[{L}]| = {drift:.2e}"
              f"   ({'labels OK' if drift == 0 else 'LABELS MAY BE OFF BY ONE'})")

        print(f"{'hs':>4s} {'what':>18s} {'max|h|':>8s} {'dynamic':>10s} "
              f"{'static':>10s} {'compiled':>10s} {'comp/stat':>10s}")
        layers = []
        for k, (r, d, e, c) in enumerate(zip(ref_h, d_h, e_h, c_h)):
            what = ("embeddings" if k == 0 else
                    f"after layer {k - 1}" if k <= n_layers - 1 else "after final norm")
            rd, re_, rc_ = rel(d, r), rel(e, r), rel(c, r)
            ratio = rc_ / max(re_, 1e-12)
            flag = "  <--" if ratio > 2 else ""
            mx = r.abs().max().item()
            layers.append({"what": what, "max_abs": mx, "dynamic": rd,
                           "static": re_, "compiled": rc_})
            print(f"{k:4d} {what:>18s} {mx:8.1f} {rd:10.2e} "
                  f"{re_:10.2e} {rc_:10.2e} {ratio:10.2f}{flag}")

        # ---- stage 4: backends
        out_idx = L + 1
        backends = backend_sweep(model, input_ids, forced, device, step,
                                 cache_eager, ref_h, ref_l, L)
        print(f"\n  attention backend sweep (eager), layer {L} output rel error and KL")
        print(f"  {'backend':22s} {'dyn L-out':>11s} {'dyn KL':>10s} "
              f"{'stat L-out':>11s} {'stat KL':>10s}")
        print(f"  {'(default)':22s} {layers[out_idx]['dynamic']:11.2e} {kls['dynamic']:10.2e} "
              f"{layers[out_idx]['static']:11.2e} {kls['static']:10.2e}")
        for name, row in backends.items():
            print(f"  {name:22s} {fmt_cell(row['dynamic'])} {fmt_cell(row['static'])}")
        print(f"  {'compiled (reference)':22s} {layers[out_idx]['compiled']:11.2e} "
              f"{kls['compiled']:10.2e}")
        for name, row in backends.items():
            for path, cell in row.items():
                if "error" in cell:
                    print(f"    {name} / {path}: {cell['error']}")

        # ---- stage 3: isolation
        pieces_rows = isolate(pieces, captured, device, L)

        report["steps"][str(step)] = {"kl": kls, "layers": layers,
                                      "backends": backends, "pieces": pieces_rows}

    # ---- attention kernels, once (every compiled graph is warm by now)
    print("\nattention kernels in one decode step:")
    kernels = {
        "dynamic": decode_step_attn_kernels(model, input_ids, forced, device),
        "static": decode_step_attn_kernels(model, input_ids, forced, device,
                                           cache=cache_eager),
        "compiled": decode_step_attn_kernels(model, input_ids, forced, device,
                                             cache=cache_comp, fwd=compiled),
    }
    for path, names in kernels.items():
        print(f"  {path}:")
        for n in names or ["(none matched)"]:
            print(f"    {n}")
    report["attention_kernels"] = kernels

    RESULTS.mkdir(exist_ok=True)
    fname = RESULTS / f"investigate_{prompt_kind}_{'emulate' if emulate else 'noemulate'}.json"
    with open(fname, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {fname}")


if __name__ == "__main__":
    main(emulate="--emulate" in sys.argv, real_prompt="--real-prompt" in sys.argv)