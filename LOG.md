# Inference Performance — Measurement Log

Single source of truth for every measurement in the 42-week plan.

**Rule: no speedup claim without a baseline, hardware, and config recorded here.**

---

## Machines

| ID   | Device              | Accelerator                                            | Memory                           | Backend   | Cost                       |
| ---- | ------------------- | ------------------------------------------------------ | -------------------------------- | --------- | -------------------------- |
| `G1` | Alienware (desktop) | RTX 3080 10GB LHR, GA102, 8704 CUDA / 272 Tensor cores | 10 GB (9.642GiB) GDDR6X, 320-bit | CUDA      | owned — **primary rig**    |
| `M1` | MacBook Pro         | Apple M5 Pro                                           | \_\_ GB unified                  | MPS       | owned — contrast case only |
| `C1` | rented              | (e.g. 1x A100 80GB SXM)                                | 80 GB                            | CUDA 12.x | $**/hr, provider **        |

### Peak theoretical — the ceilings

|                          | `G1` RTX 3080    | `M1` M5 Pro   |
| ------------------------ | ---------------- | ------------- |
| memory bandwidth         | **760 GB/s**     | ~307 GB/s     |
| FP32 compute             | 29.8 TFLOP/s     | ~8.3 TFLOP/s  |
| FP16 tensor (dense)      | **59.5 TFLOP/s** | not published |
| FP16 tensor (2:4 sparse) | 119 TFLOP/s      | n/a           |

### Measured vs peak — `G1` (2026-09-10)

|                         | peak         | measured          | % of peak |
| ----------------------- | ------------ | ----------------- | --------- |
| memory bandwidth (fp32) | 760 GB/s     | 679 GB/s          | 89%       |
| memory bandwidth (bf16) | 760 GB/s     | 610 GB/s          | 80%       |
| FP32 compute            | 29.8 TFLOP/s | 16.1 TFLOP/s      | 54%       |
| BF16 tensor (dense)     | 59.5 TFLOP/s | 27.4–31.3 TFLOP/s | 46–53%    |

Source: cuBLAS/cutlass GEMMs and elementwise kernels in a single nanoGPT
attention layer, B=1 T=1024 C=768. See log entry 2026-09-10.

Bandwidth figures are healthy. Compute figures are ~half of peak because these
are ordinary cuBLAS GEMMs at shapes that do not saturate the tensor cores —
not a hardware problem, a shape problem. Do not use peak compute in roofline
arithmetic; use these.

**Roofline ridge point for `G1`:** 59.5 TFLOP/s / 760 GB/s = **~78 FLOP per byte**.

- Kernel arithmetic intensity **below 78** -> memory-bound. Judge it against 760 GB/s.
- **Above 78** -> compute-bound. Judge it against 59.5 TFLOP/s.
- LLM decode sits around 1-2 FLOP/byte. It is _always_ memory-bound. This is the
  single fact the whole plan is built on.

**Achieved ridge (measured 2026-09-10):**

| dtype | achieved compute | achieved bandwidth | ridge              |
| ----- | ---------------- | ------------------ | ------------------ |
| fp32  | 16.1 TFLOP/s     | 679 GB/s           | **23.7** FLOP/byte |
| bf16  | 27.4 TFLOP/s     | 610 GB/s           | **45** FLOP/byte   |

Theoretical ridge (78) overstates by ~1.7x in bf16 and ~3x in fp32.

**The ridge moves with dtype.** Going fp32 -> bf16, compute nearly doubles
while bandwidth does not improve, so the ridge shifts right and _more_
operations land on the memory-bound side. Lower precision does not merely make
everything faster — it changes which kernel is the bottleneck.

The decode claim (1-2 FLOP/byte, always memory-bound) holds under any of these
ridges. It is not sensitive to the calibration.

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
- Explicit synchronize? (Y/N) `torch.cuda.synchronize()`

**Result**

| Metric              | Value | Unit    |
| ------------------- | ----- | ------- |
| latency p50         |       | ms      |
| latency p95         |       | ms      |
| latency p99         |       | ms      |
| TTFT p50            |       | ms      |
| inter-token latency |       | ms/tok  |
| throughput          |       | tok/s   |
| achieved bandwidth  |       | GB/s    |
| achieved compute    |       | TFLOP/s |
| **% of peak**       |       | %       |

**Baseline compared against:** link to a prior entry, or "none — this IS the baseline"
**Delta:** +/- **% vs \_\_**

**Interpretation**

- Compute-bound or memory-bound? Evidence:
- Why is it not faster? Name the bottleneck:

**Surprises / what I got wrong**

- **Confusions to revisit** (concepts that needed rewatching or rereading)

- **Repro:** `command` · script `path/to/bench.py` · seed \_\_\_\_

  ***

## Log

### 2026-08-15 — Week 1 — Environment up

**Machine:** G1 (primary) / M1 (secondary)
**Question:** Is the GPU visible to PyTorch, and is this project reproducible?

**Setup**

- `uv init` + `uv add torch numpy matplotlib jupyter` — done
- python 3.12, torch >=2.13.0
- `uv.lock` committed to git? (Y/N)

**Result**

- `G1`: `torch.cuda.is_available()` -> \_\_True\_\_
- `G1`: `torch.cuda.get_device_name(0)` -> \_\_NVIDIA GeForce RTX 3080\_\_
- `G1`: driver / CUDA runtime version -> \_\_595.84\_\_
- `M1`: `torch.backends.mps.is_available()` -> \_\_True\_\_

**Notes**

- Setup is a two-hour task, not a week-long one. micrograd starts today.
- The lockfile is the reproducibility guarantee — same instinct as pinning a Terraform provider.

**Confusions to revisit**

- ***

## Environment — 2026-08-16

**Machine:** g1 (Linux, EXT4, 27 GiB RAM)
**GPU:** NVIDIA Ampere, sm_86 (compute 8.6), 9.64 GiB
**Driver:** <nvidia-smi>
**PyTorch:** <torch.**version**>, built against CUDA 13.0
**System toolkit:** CUDA 13.3.1, CUDA_HOME=/usr/local/cuda-13.3

- was 12.0 — three years behind PyTorch, caused the failure below
- PATH exports in ~/.profile, pinned to explicit version not /usr/local/cuda
  **Python:** 3.12, uv project at ~/projects/inference-track
  **vLLM:** 0.27.1

### Theoretical peaks (from spec sheet)

- Memory bandwidth: \_\_760\_ GB/s
- FP32 / BF16 TFLOPs: \_\_29.8\_

---

## Issue 001 — FlashInfer JIT build failure (2026-08-16) — RESOLVED

**Symptom:** `vllm serve Qwen/Qwen3-1.7B` fails at warmup.
Model loads, KV cache allocates, CUDA graphs capture — dies on sampler warmup.
`error: cub::_V_300302_SM_860::BlockAdjacentDifference has no member "FlagHeads"`

**Cause:** FlashInfer JIT-compiles its sampling kernel and injects its own
bundled CCCL 3.3.2 via `-I`. CUB removed `FlagHeads` in CCCL 3.0. The
version-selection shim appears to key off toolkit version (12.0) rather than
the CCCL actually on the include path, so it emitted a call to a removed API.

**Workaround:** `VLLM_USE_FLASHINFER_SAMPLER=0` (native PyTorch sampler,
negligible cost at this scale)
**Fix:** upgraded system toolkit 12.0 → 13.3.1
**Cause:** FlashInfer JIT-compiles its sampling kernel and injects its own
bundled CCCL via `-I`. `sampling.cuh` lines 112-114 gate the CUB API choice on
`__CUDACC_VER_MAJOR__` / `__CUDACC_VER_MINOR__` — the _compiler_ version —
rather than on the CUB/CCCL header version actually on the include path.
FlashInfer bundles CCCL at CUB 3.3.2 (`CUB_VERSION 300302`), where `FlagHeads`
has been removed and `SubtractLeft` is available. On toolkit 12.0 the guard
took the `FlagHeads` branch, which is dead/broken code on every supported
config.

**Lesson (generalised):** version detection must key off the bundled headers,
not the compiler. Same failure mode will appear anywhere a project vendors its
own dependency headers.

**Lesson:** "CUDA version" is three independent things — driver, the runtime
PyTorch ships with, and the toolkit at /usr/bin/nvcc. Only JIT paths touch
the third. Everything else ran fine on a 3-year-old toolkit because it never
compiled anything. Will recur with Triton in Phase 3.

---

## Measurements

### 2026-08-16 — vLLM baseline, Qwen3-1.7B

Config: --max-model-len 4096 --gpu-memory-utilization 0.85

- Weights + non-torch: 3.52 GiB
- Peak activation: 0.09 GiB
- CUDA graphs: 0.47 GiB
- KV cache: 4.58 GiB → 42,880 tokens → 10.47x concurrency at 4K
- Model load: 3.14 s | torch.compile: 0.15 s (AOT cache hit) | graphs: 3 s

### 2026-09-10 — Week 4 — nanoGPT attention: backend dispatch, fusion, and two hand optimisations

**Machine:** G1

**Question:** Which SDPA backend actually runs, what does fusion buy, and how
much of the flash-vs-manual gap is avoidable inefficiency in the manual path?

**Setup**

- Kernel: `CausalSelfAttention` (nanoGPT), single layer, forward only, `.eval()`
- Shape: B=1, T=1024, C=768, n_head=12, head_dim=64
- Precision: fp32 and bf16
- Batch / concurrency: 1
- Warmup 50 iters / measured 50 iters
- Explicit synchronize: Y (before and inside the profiled region)
- Profiler: `torch.profiler`, CPU+CUDA activities, `record_shapes=True`,
  `record_function` regions around each line of `attn_manual`
- 6 runs across 2 sessions

**Result — bf16, per forward pass**

|                | flash    | manual (before) | manual (after) |
| -------------- | -------- | --------------- | -------------- |
| attention core | 74.2 us  | 460.4 us        | 307.1 us       |
| full layer fwd | 245.1 us | 635.4 us        | 484.6 us       |

Speedup flash vs manual: **6.20x** core before optimisation, **4.14x** after.
Full-layer: 2.59x before, **1.98x** after.

**Full kernel attribution — manual path, bf16, after optimisation**

| region       | kernel                                        | us/iter    |
| ------------ | --------------------------------------------- | ---------- |
| (QKV proj)   | `cutlass_80_tensorop_bf16 256x128_32x3_tn`    | 132.24     |
| `scores`     | `vectorized_elementwise` (scale, on q)        | 6.72       |
| `scores`     | `cutlass_80_wmma_tensorop_bf16 32x32_32x1_tn` | 74.56      |
| `mask`       | `masked_fill_kernel`                          | 86.87      |
| `softmax`    | `softmax_warp_forward`                        | 82.54      |
| `att_v`      | `cutlass_80_tensorop_bf16 64x64_32x6_nn`      | 56.44      |
| (contiguous) | `direct_copy_kernel`                          | 6.93       |
| (c_proj)     | `cutlass_80_tensorop_bf16 128x128_32x4_tn`    | 38.27      |
|              | **sum**                                       | **484.58** |

Total measured CUDA time: 24.229 ms / 50 = **484.58 us**. Attribution is exact
— no unaccounted kernel time.

**Flash path kernel attribution**

| kernel                                                         | us/iter    |
| -------------------------------------------------------------- | ---------- |
| QKV proj (same cutlass kernel as above)                        | 132.27     |
| `pytorch_flash::flash_fwd_kernel<64,128,128,4,...,bfloat16_t>` | 74.22      |
| c_proj (same cutlass kernel as above)                          | 38.56      |
| **sum**                                                        | **245.05** |

**Achieved rates**

| kernel      | TFLOP/s | % of 59.5 peak |
| ----------- | ------- | -------------- |
| c_proj GEMM | 31.3    | 53%            |
| att_v BMM   | 28.5    | 48%            |
| QKV GEMM    | 27.4    | 46%            |
| scores BMM  | 21.6    | 36%            |

| kernel       | GB/s | % of 760 peak |
| ------------ | ---- | ------------- |
| softmax      | 610  | 80%           |
| masked*fill* | 579  | 76%           |

**Compute-bound or memory-bound? Evidence:**

Mixed, and the split is the point. GEMMs sit at AI 140-180, well right of the
measured bf16 ridge of 45 — compute-bound, running at 46-53% of tensor-core
peak. Softmax and mask sit at AI ~0.6 — far left, running at 76-80% of
bandwidth peak. The elementwise ops are 0.8% of attention's FLOPs and 55% of
its memory traffic.

Unfused, the manual path's three middle kernels pass a 25.2 MB `att` tensor
between them and traverse it repeatedly. FlashAttention never materialises it:
reads q,k,v and writes y only, ~6.3 MB total.

**Why is it not faster? Name the bottleneck:**

For the manual path: HBM traffic on the T x T intermediate. For the flash
kernel: it is already compute-bound and the remaining ceiling is tensor-core
utilisation, not memory.

---

**Surprises / what I got wrong**

1. **`hasattr(F, 'scaled_dot_product_attention')` tells you nothing about
   which kernel runs.** fp32 dispatched to `fmha_cutlassF_f32_aligned_64x64_rf_sm80`
   — the _memory-efficient_ backend, not FlashAttention. Only bf16 (or fp16)
   satisfies the flash kernel's dtype requirement and gets
   `pytorch_flash::flash_fwd_kernel`. The kernel name in the profile is the
   only way to know. `torch.nn.attention.sdpa_kernel` can force/query it.

2. **`masked_fill` (out-of-place) clones the tensor.** 161.96 us total =
   ~74.7 us clone + 87.27 us fill. The clone copies 25.2 MB that is never
   read again. One character fixes it: `masked_fill_`. Saved **83.4 us/iter**.
   _Upstream nanoGPT has this pattern._

3. **The `1/sqrt(d_k)` scale is NOT folded into the cuBLAS alpha.** It is a
   separate `vectorized_elementwise_kernel` costing 75.1 us — a full pass over
   the 25.2 MB `att`. Scaling `q` instead (400x smaller tensor) before the
   matmul is mathematically identical and drops it to 6.7 us. Saved
   **68.9 us/iter**. _Upstream nanoGPT has this pattern too._

4. **A third of the "FlashAttention is 6x faster" gap was avoidable
   inefficiency in the baseline**, not fusion. After the two fixes above the
   ratio is 4.14x. Any speedup claim against an unoptimised baseline is
   partly measuring the baseline's bugs.

5. **Flash and manual do not compute the same FLOPs.** With `is_causal=True`
   the flash kernel skips masked tiles; the manual path computes the full
   T x T triangle and discards half. Part of the 4.14x is fusion, part is
   skipped work. To isolate fusion alone, rerun both with `causal=False`.
   (Counting full FLOPs, flash reads as 43.4 TFLOP/s; counting only the
   causal half, ~22.5 TFLOP/s. The second number is the honest one.)

6. **The flash path's `.contiguous()` is free.** `_flash_attention_forward`
   takes and returns (B, T, nh, hs) layout — visible in the recorded input
   shapes — so `y.transpose(1,2)` lands already-contiguous and the call is a
   no-op. The manual path pays 6.93 us for the same line.

7. **First bench in a cell runs ~30% slow.** Two of six runs showed the
   _shared_ QKV kernel at 174.6 / 143.3 us instead of 132.3 us. Always the
   first `bench()` call, never the second. Clock ramp — 10 warmup iters is not
   enough from idle. Raise warmup to 100, or assert the QKV kernel is within
   5% of 132 us before trusting a run.

**Run-to-run variance**

Excluding the two clock-ramp outliers: kernel-level variance **< 0.5%** across
4 runs and 2 sessions. A measured delta above ~2% is real signal.

**Correctness**

- flash vs manual, fp32: max abs diff **1.788e-07** (`assert_close` passes)
- causal masking verified behaviourally: perturbing token 5 leaves outputs
  0-4 _exactly_ zero with `causal=True`, and moves all positions with
  `causal=False`. Confirmed for both impls, 4 configurations.
- **PENDING:** rerun `verify_impls` in bf16 after the q-prescaling change.
  Pre-scaling q moves where bf16 rounding happens (rounding a (1,12,1024,64)
  tensor instead of the (1,12,1024,1024) product). Expect it to pass — bf16
  default rtol is 1.6e-2 — but record the actual max-abs.

**Baseline compared against:** none — this IS the attention baseline.

**Repro:** notebook `nanogpt-attention-profile.ipynb`, torch <fill version>,
CUDA <fill>, driver 595.84. Seed not set (input is random; shapes are what
matter for timing).

**Confusions to revisit**

- Why `att @ v` is _faster_ than `q @ k.T` in bf16 (56.4 vs 74.6 us) but was
  _slower_ in fp32 (247 vs 139 us). Different cutlass tile selections
  (`64x64_32x6_nn` vs `wmma 32x32_32x1_tn`). Unexplained, not settled.
- Whether the remaining 46-53%-of-peak on the GEMMs is shape-driven or
  something addressable.

### 2026-09-17 — Week 6 — Prefill vs decode: TTFT / ITL instrumentation

**Machine:** G1
**Question:** Can I predict batch-1 decode throughput from model size and memory bandwidth, and does the measurement agree?

**Setup**

- Model / kernel: `EleutherAI/pythia-410m` (24 layers, hidden 1024, 16 heads × 64, vocab 50304)
- Framework + version: torch 2.13.0+cu130, transformers (HF `AutoModelForCausalLM`), CUDA runtime 13.0
- Precision: bf16 (verified via `p.dtype` over all parameters — uniform)
- Batch size / concurrency: 1
- Input len -> output len: 512 -> 64
- Warmup iters / measured iters: 3 warmup (full `measure()` call, incl. tokenizer init) / 5 reps
- Explicit synchronize? **Y** — `torch.cuda.synchronize()` at both ends of every timed region
- Decoding: greedy (`argmax`), EOS ignored, fixed token count so every run is comparable
- Prompt: repeated-token filler, round-trip asserted to land within ±2 of 512 tokens

**Result**

| Metric                        | Value  | Unit    |
| ----------------------------- | ------ | ------- |
| TTFT p50                      | 13.37  | ms      |
| — of which tokenize           | 0.78   | ms      |
| — of which prefill            | 12.52  | ms      |
| — of which sample + detokenize| 0.06   | ms      |
| inter-token latency p50       | 8.53   | ms/tok  |
| inter-token latency p95        | 8.58   | ms/tok  |
| throughput (decode only)      | 117    | tok/s   |
| achieved bandwidth (weights)  | 83     | GB/s    |
| achieved bandwidth (+KV)      | 89     | GB/s    |
| **% of peak** (vs 610 measured) | **14** | %     |
| % of peak (vs 760 theoretical)| 11     | %       |

Model bytes, measured not assumed:

| Quantity                          | Bytes    |
| --------------------------------- | -------- |
| all parameters                    | 0.811 GB |
| `embed_in` (row lookup, not read) | 0.103 GB |
| **decode-relevant weights**       | **0.708 GB** |
| KV cache at position 512          | 0.050 GB |
| KV cache at position 575          | 0.057 GB |

`embed_in` is excluded because decode indexes a single 1024-element row (~2 KB), not the
51.5M-parameter table. `embed_out` is **not** excluded — pythia does not tie embeddings,
and the lm_head is a full matmul against all 50304 rows every step.

**Baseline compared against:** none — this IS the baseline for batch-1 HF decode.
**Delta:** n/a

---

**Interpretation**

**Prediction:** 610 GB/s ÷ 0.708 GB ≈ **861 tok/s** (≈ 805 tok/s including KV cache traffic).
**Measured:** 117 tok/s. **The prediction misses by ~7×.** Done-When asked for ±30%.

Per-token cost, prefill vs decode:

```
prefill:  12.52 ms / 512 tokens =  24.4 µs per token
decode:    8.53 ms /   1 token  = 8530   µs per token   → 350× more expensive
```

**Compute-bound or memory-bound? Neither — overhead-bound.** Four independent lines of
evidence:

1. **ITL tracks layer count, not bytes.** Size sweep at 512→64:

   | model  | layers | hidden | ITL     | tok/s | ms per layer |
   | ------ | ------ | ------ | ------- | ----- | ------------ |
   | 160m   | 12     | 768    | 4.69 ms | 213.0 | 0.39         |
   | 410m   | 24     | 1024   | 8.71 ms | 114.7 | 0.36         |
   | 1.4b   | 24     | 2048   | 8.66 ms | 114.5 | 0.36         |

   410m and 1.4b share 24 layers and land within 0.6% of each other despite 1.4b moving
   ~3.6× the bytes. If decode were bandwidth-bound, 1.4b would be ~3.6× slower. Per-layer
   cost is flat at ~0.36 ms across a 15× range in model size.

2. **GPU is idle two thirds of the time.** Profiler on one decode step: Self CUDA total
   2.995 ms against a measured ITL of 8.71 ms → **34% GPU utilization**.

3. **The host can barely feed the device.** 565 `cudaLaunchKernel` calls per decode step
   (23.5 per layer), 2.808 ms of CPU time, **4.97 µs per launch** — essentially equal to
   the 5.3 µs average GPU execution time per kernel.

4. **15.6% of GPU time computes nothing.** `CatArrayBatch` — 120 calls, 466 µs — is
   `DynamicCache` reallocating and copying the KV cache every step, five times per layer.

GPU time breakdown for one decode step (Self CUDA 2.995 ms total):

| Kernel                    | Calls | Time   | Share | What it is                |
| ------------------------- | ----- | ------ | ----- | ------------------------- |
| cutlass wmma bf16         | 48    | 737 µs | 24.6% | MLP matmuls (2/layer)     |
| gemvx                     | 48    | 533 µs | 17.8% | QKV + output projections  |
| **CatArrayBatch**         | 120   | 466 µs | 15.6% | **KV cache concatenation**|
| flash_fwd_splitkv         | 24    | 261 µs |  8.7% | attention                 |
| elementwise (assorted)    | ~250  | ~460 µs| ~15%  | norms, residuals, rotary  |
| gemv2T                    | 1     | 147 µs |  4.9% | lm_head                   |

Real matmul work totals ~1.68 ms — **56% of GPU time**. The rest is bookkeeping.

**Why is it not faster? Name the bottleneck:** kernel launch overhead and per-kernel
ramp-up at batch 1. 565 tiny kernels per token, each doing a matrix-vector product that
finishes in microseconds. The memory system never reaches steady-state streaming, so the
roofline's bandwidth ceiling is never approached.

---

**Surprises / what I got wrong**

- **The plan's premise needs a qualifier.** LOG.md states decode "is _always_ memory-bound."
  True about arithmetic intensity (~2 FLOP/byte), false about what actually limits batch-1
  HF decode. Bandwidth is the *ceiling*; launch overhead is the *floor I am sitting on*.
  The roofline model assumes one resource is saturated — at batch 1, neither is.

- **Smaller models are less hardware-efficient, not more.** Achieved bandwidth rises with
  model size: ~49% of 610 GB/s at 1.4b, ~14% at 410m, lower still at 160m (byte counts for
  160m/1.4b are estimates — recompute with `element_size()` before quoting). 160m wins on
  tok/s while wasting most of the card, because a fixed per-layer cost is amortized over
  less useful work. "Smaller model, faster inference" is true in tok/s and badly false in
  utilization.

- **Predicted 10 kernels/layer from the architecture; actual is 23.5.** Estimate was 2.4×
  low. Do not reason about launch counts from op counts — profile them.

- **A first-call tokenizer cost of ~6.7 ms exists** and is entirely absorbed by warmup
  (measured 0.06 ms in steady state). Lazy init in the Rust tokenizer backend.

- **Two ITL outliers across 315 measurements:** 11.18 ms (rep 1, token 7) and 8.88 ms
  (rep 2, token 35), against a median of 8.53. Single occurrences, not reproducible.
  Unattributed — CUDA-event timing would separate host stall from device stall.
  Kept the full ITL vector precisely so these stayed visible; a mean would have hidden them.

- **Reproducibility was better than expected.** Prefill across 5 independent reps:
  12.5111 / 12.5150 / 12.5160 / 12.5178 / 12.5186 ms — 0.06% spread.

**Confusions to revisit**

- The lm_head at 147 µs for ~103 MB implies ~700 GB/s, **above** the 610 GB/s measured
  ceiling. Either the byte estimate is wrong or there is cache reuse. Reconcile — this is
  the one kernel in the profile that looks bandwidth-saturating, so it would make a useful
  reference for what "good" looks like on this card.
- 0.36 ms per layer at ~23.5 kernels is ~15 µs per kernel, above the typical 5–10 µs launch
  cost. Launch overhead explains much of the gap but possibly not all of it. Nsight Systems
  (Week 28) will show whether the remainder is gaps between kernels or slow kernels.

**Next actions**

1. Prompt-length sweep (128 / 512 / 1024 / 2048) — confirm TTFT scales ~linearly and ITL
   stays flat. Method bullet 2 is still one data point.
2. `StaticCache` instead of `DynamicCache` — preallocates, writes in place. Should remove
   ~120 launches and ~466 µs per token. Directly testable with this harness.
3. CUDA graph capture of the decode step — replays 565 launches as one submission. Expected
   to remove most of the 2.808 ms host cost. This is what the 0.47 GiB of CUDA graphs in the
   Week 1 vLLM baseline was buying.
4. Add `torch.cuda.Event` timing alongside `perf_counter` in the decode loop — separates
   host launch time from device execution time by measurement rather than inference.

**Harness limitations (recorded now, relevant in Phase 2)**

- Filler prompt is repeated tokens; greedy decode on it produces degenerate output. Fine
  for timing (content does not affect cost), wrong for anything where sequences must finish
  at different times.
- `tok.decode` is inside the timed decode region — CPU work a real server pays, but it means
  ITL is not pure GPU decode. Unmeasured; run once with it commented out to size the gap.
- Batch size hardcoded to 1. Batching moves decode from matrix-vector to matrix-matrix and
  shifts it right on the roofline — the whole point of Phase 2.

**Repro:** `python bench_ttft_itl.py` · script `bench_ttft_itl.py` · greedy, no seed needed
(deterministic) · kernel `Python (mlops-jupyter)`


<!-- New entries go ABOVE this line, newest last. Keep it chronological. -->
