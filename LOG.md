# Inference Performance — Measurement Log

Single source of truth for every measurement in the 42-week plan.

**Rule: no speedup claim without a baseline, hardware, and config recorded here.**

---

## Machines

| ID | Device | Accelerator | Memory | Backend | Cost |
|----|--------|-------------|--------|---------|------|
| `G1` | Alienware (desktop) | RTX 3080 10GB LHR, GA102, 8704 CUDA / 272 Tensor cores | 10 GB GDDR6X, 320-bit | CUDA | owned — **primary rig** |
| `M1` | MacBook Pro | Apple M5 Pro | __ GB unified | MPS | owned — contrast case only |
| `C1` | rented | (e.g. 1x A100 80GB SXM) | 80 GB | CUDA 12.x | $__/hr, provider __ |

### Peak theoretical — the ceilings

| | `G1` RTX 3080 | `M1` M5 Pro |
|---|---|---|
| memory bandwidth | **760 GB/s** | ~307 GB/s |
| FP32 compute | 29.8 TFLOP/s | ~8.3 TFLOP/s |
| FP16 tensor (dense) | **59.5 TFLOP/s** | not published |
| FP16 tensor (2:4 sparse) | 119 TFLOP/s | n/a |

**Roofline ridge point for `G1`:** 59.5 TFLOP/s / 760 GB/s = **~78 FLOP per byte**.

- Kernel arithmetic intensity **below 78** -> memory-bound. Judge it against 760 GB/s.
- **Above 78** -> compute-bound. Judge it against 59.5 TFLOP/s.
- LLM decode sits around 1-2 FLOP/byte. It is *always* memory-bound. This is the
  single fact the whole plan is built on.

### Notes on `G1`

- **LHR is irrelevant here.** Lite Hash Rate throttles the Ethereum mining kernel
  only. Zero effect on ML, CUDA, or anything in this plan.
- **10 GB is the real constraint.** Rough capacity at ~85% usable (8.5 GB), leaving
  room for KV cache and activations:
  - FP16 (2 bytes/param): up to ~3.5B params comfortable, 7B is tight-to-impossible
  - INT8 (1 byte/param): ~7B fits
  - INT4 (0.5 byte/param): ~13B fits
- Weeks 21-24 (quantization) are what unlock the larger models on this card.
  Until then, work at 1B-3B and measure honestly.
- Ampere = compute capability 8.6. Supports BF16, TF32, and 2:4 structured sparsity.
  **No FP8** (that is Ada/Hopper and later) — note this when the plan reaches FP8
  quantization; that section needs the rented `C1` box.

Any measured result is only meaningful as a **percentage of peak**.

---

## Entry Template

Copy this block for every measurement session. Do not skip fields — the blank
ones are usually the ones that invalidate the result three months later.

### YYYY-MM-DD — Week NN — short title

**Machine:** G1
**Question:** what am I trying to find out (one sentence)

**Setup**

- Model / kernel:
- Framework + version: (torch 2.13.x, vllm 0.x.x + commit hash)
- Precision: (fp32 / fp16 / bf16 / fp8 / int8 / int4)
- Batch size / concurrency:
- Input len -> output len:
- Warmup iters / measured iters:
- Explicit synchronize? (Y/N)  `torch.cuda.synchronize()`

**Result**

| Metric | Value | Unit |
|--------|-------|------|
| latency p50 | | ms |
| latency p95 | | ms |
| latency p99 | | ms |
| TTFT p50 | | ms |
| inter-token latency | | ms/tok |
| throughput | | tok/s |
| achieved bandwidth | | GB/s |
| achieved compute | | TFLOP/s |
| **% of peak** | | % |

**Baseline compared against:** link to a prior entry, or "none — this IS the baseline"
**Delta:** +/- __% vs ____

**Interpretation**

- Compute-bound or memory-bound? Evidence:
- Why is it not faster? Name the bottleneck:

**Surprises / what I got wrong**

-

**Confusions to revisit** (concepts that needed rewatching or rereading)

-

**Repro:** `command` · script `path/to/bench.py` · seed ____

---

## Log

### 2026-08-__ — Week 1 — Environment up

**Machine:** G1 (primary) / M1 (secondary)
**Question:** Is the GPU visible to PyTorch, and is this project reproducible?

**Setup**

- `uv init` + `uv add torch numpy matplotlib jupyter` — done
- python 3.12, torch >=2.13.0
- `uv.lock` committed to git? (Y/N)

**Result**

- `G1`: `torch.cuda.is_available()` -> ____
- `G1`: `torch.cuda.get_device_name(0)` -> ____
- `G1`: driver / CUDA runtime version -> ____
- `M1`: `torch.backends.mps.is_available()` -> ____

**Notes**

- Setup is a two-hour task, not a week-long one. micrograd starts today.
- The lockfile is the reproducibility guarantee — same instinct as pinning a Terraform provider.

**Confusions to revisit**

-

---

<!-- New entries go ABOVE this line, newest last. Keep it chronological. -->
