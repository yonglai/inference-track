# Fusion, not CUDA graphs: where the batch-1 decode speedup actually comes from

*September 2026 · Pythia-410M · RTX 3080 · batch 1*

## TL;DR

- Batch-1 decode in eager PyTorch spent **more than half of every token idle**, waiting for the CPU to launch 654 small kernels.
- `torch.compile` with CUDA graphs cut decode latency from **8.51 ms to 2.97 ms per token (2.87×)**. **Kernel fusion delivered 95% of that.** CUDA graph replay added the last 8%.
- On ordinary text, the compiled path was **as accurate as eager**: zero changed predictions across 64 steps against an fp32 reference.
- Along the way, a synthetic benchmark prompt made compilation look like it was damaging model output. It wasn't. The prompt made attention numerically fragile. That detour is described in [section 4](#4-the-benchmark-prompt-that-nearly-fooled-me).

## Setup

| | |
| --- | --- |
| GPU | NVIDIA RTX 3080 10 GB (Ampere, sm_86) |
| Memory bandwidth | 760 GB/s spec · 610 GB/s measured |
| Software | PyTorch 2.13.0+cu130 · Hugging Face transformers · CUDA runtime 13.0 |
| Model | `EleutherAI/pythia-410m`, bf16 |
| Workload | batch 1 · 512-token prompt · 64 generated tokens · greedy |
| Method | 3 warmup + 5 measured runs; median inter-token latency (ITL); explicit `torch.cuda.synchronize()` around every timed region |

This is a small model on a consumer card at batch 1. The conclusions are about where time goes in that regime, not about serving at scale.

## 1. The starting point: overhead-bound, not memory-bound

Decode is usually described as memory-bandwidth-bound: each token reads every weight once to do very little arithmetic. By that logic, this model's ~0.71 GB of decode-relevant weights at 610 GB/s should allow roughly 860 tokens/s.

Measured: **117 tokens/s**, about 14% of that ceiling.

Three measurements said the bottleneck was launch overhead instead:

- **Latency tracked layer count, not model size.** Pythia-410M and Pythia-1.4B both have 24 layers and both decoded at ~8.7 ms/token, even though 1.4B moves about 3.6× more bytes per token. Pythia-160M has 12 layers and ran at 4.7 ms.
- **The GPU was idle most of the time.** Busy time per token was well under half the wall-clock time.
- **Issuing kernels cost about as much as running them.** Each `cudaLaunchKernel` call took ~5 µs of CPU time, about the same as the average kernel's GPU execution time.

At batch 1, every operation works on a single 1024-element vector, so each kernel finishes almost as soon as it starts. The CPU can barely launch work as fast as the GPU finishes it.

## 2. Four variants, one change at a time

| Variant | What changes |
| --- | --- |
| DynamicCache eager | Baseline. The KV cache grows by concatenation every step; one kernel launch per operation. |
| StaticCache eager | KV cache preallocated to a fixed size and written in place. Still one launch per operation. |
| StaticCache compiled | `torch.compile(mode="default")`: Inductor fuses operations into fewer kernels. No CUDA graphs. |
| StaticCache + graph | `torch.compile(mode="reduce-overhead")`: the same fused kernels, captured once and replayed as a single CUDA graph launch. |

Separating the compiled variant with and without graphs is what makes it possible to say how much each technique contributes. `mode="reduce-overhead"` does both at once, and most descriptions credit the graphs.

## 3. Results: fusion does the work

| Variant | ITL | tok/s | Kernel launches per token | GPU busy | GPU busy / ITL |
| --- | --- | --- | --- | --- | --- |
| DynamicCache eager | 8.509 ms | 117.5 | 654 | 3794 µs | 45% |
| StaticCache eager | 9.543 ms | 104.8 | 692 | 3761 µs | 39% |
| StaticCache compiled | 3.217 ms | 310.9 | 73 | 2747 µs | 85% |
| **StaticCache + graph** | **2.968 ms** | **336.9** | **1** | **2658 µs** | **90%** |

Kernel counts are for one decode step, prefill excluded.

| Step | Latency | Speedup | Share of the gain |
| --- | --- | --- | --- |
| Eager → fused (654 → 73 launches) | 8.509 → 3.217 ms | 2.65× | **95%** |
| Fused → graph replay (73 → 1 launch) | 3.217 → 2.968 ms | 1.08× | 5% |
| **Total** | **8.509 → 2.968 ms** | **2.87×** | |

**Fusion removes both launches and memory traffic.** Inductor combines chains of small elementwise operations (LayerNorm, residual adds, activation, rotary embedding) into single kernels. That cuts launches from 654 to 73, and it also cuts GPU time by 27%, because intermediate results no longer make a round trip through GPU memory between operations.

**CUDA graphs remove what fusion can't.** Matmuls and attention can't be fused into their neighbours, so about 73 launches remain. A CUDA graph records that sequence once and replays it with a single call, removing the remaining ~6 µs × 73 of CPU dispatch. The kernels are the same, so the numerical results are bit-identical with and without graphs.

**The two per-launch estimates agree.** The original profile measured ~5 µs per `cudaLaunchKernel`. The fused variant's wall-clock-minus-GPU gap works out to (3217 − 2747) µs / 73 ≈ 6.4 µs per launch.

**StaticCache on its own is slower.** Preallocating the cache replaces 48 concatenations with 48 in-place writes, but it also adds 72 small integer-arithmetic kernels to compute where to write, plus mask construction over the full preallocated length. It also loses access to FlashAttention, which can't accept the explicit attention mask StaticCache passes, and falls back to memory-efficient attention. In eager mode that's a net loss. Its value is that fixed shapes are a prerequisite for compilation, and compilation removes the extra bookkeeping entirely.

**Where it lands:** 90% GPU busy and ~255 GB/s, about 42% of measured bandwidth. The remaining gap isn't host overhead any more. It's matrix-vector kernels too small to reach full memory throughput at batch 1, which is a batching problem.

## 4. The benchmark prompt that nearly fooled me

To control prompt length precisely, the first benchmark used **512 repetitions of the token " the"**. Content doesn't affect timing, so this was fine for latency. It was not fine for accuracy.

On that prompt, the compiled path looked clearly less accurate than eager:

| vs fp32 reference, " the" × 512 prompt | KL mean | KL max |
| --- | --- | --- |
| DynamicCache eager | 3.98e-3 | 0.107 |
| StaticCache eager | 2.02e-3 | 0.008 |
| Compiled, Inductor default rounding | **1.28e-2** | **0.453** |

That looked like a real finding, so I narrowed it down:

1. **Located it.** Comparing every layer's hidden state against fp32 showed the compiled path's error jumping at the output of layer 5.
2. **Ruled out the MLP and norms.** Each non-attention piece of layer 5, run in isolation on identical inputs, matched eager exactly.
3. **Found a fix.** Inductor's `emulate_precision_casts` flag, which makes fused kernels round intermediates to bf16 where eager would, removed the largest spike.
4. **Found the actual cause.** Forcing eager attention onto each available kernel showed three *correct* attention implementations disagreeing by **25× in KL** on the same input. DynamicCache eager uses FlashAttention; StaticCache eager uses memory-efficient attention. Forcing the dynamic path onto memory-efficient attention reproduced the static path's numbers exactly.

When legitimate kernels disagree that much, the input is the problem. With 512 identical tokens, keys differ only by position and values are nearly identical. The attention output then depends on tiny differences summed across 512 positions, so summation order, which differs between kernels, dominates the result. That mechanism is a hypothesis; the 25× spread is measured.

**Every changed prediction on this prompt occurred at a near-tie.** In each case, the top two candidate tokens were less than 2 bf16 ULPs apart (ULP: the gap between adjacent representable values). In one case they rounded to exactly the same bf16 value.

## 5. Accuracy on real text

Rerunning with ~500 words of ordinary prose, same length:

| vs fp32 reference, prose prompt | KL mean | KL max | Changed predictions (64 steps) |
| --- | --- | --- | --- |
| DynamicCache eager | 4.87e-3 | 0.058 | **0** |
| StaticCache eager | 3.47e-3 | 0.033 | **0** |
| Compiled / graph, Inductor default rounding | 4.68e-3 | 0.088 | **0** |
| **Compiled / graph, `emulate_precision_casts`** | **3.99e-3** | **0.033** | **0** |

- **No path changes any prediction.** Every variant picks the same token as the fp32 model at all 64 steps.
- All paths fall within a 1.4× band on mean KL, compared with 6× on the repeated-token prompt.
- All paths have their worst step at the same position, so that peak comes from the input, not from any implementation.
- `emulate_precision_casts` is worth keeping on. It brings the compiled path's worst step in line with eager, and it's marginally *faster* (about 50 µs less GPU time per token), because it also changes how Inductor splits some kernels.

Every variant is teacher-forced on the same token sequence, so one changed prediction can't cascade into later steps. KL divergence is used rather than maximum logit difference because it weights each token by how likely it is. A large error on a token nobody would pick doesn't count.

## 6. What this doesn't show

- **One small model, one GPU, batch 1.** Larger models or batches shift the balance between launch overhead and real work. At higher batch sizes each kernel does more useful work per launch, and fusion and graphs matter proportionally less.
- **One prose prompt and 64 steps.** Enough to show the repeated-token results weren't representative. Not enough to characterize accuracy in general.
- **fp32 is a reference, not ground truth.** It has about 16 more mantissa bits than bf16, enough to rank bf16 implementations but not to certify any of them.
- **Hugging Face transformers is a reference implementation**, not a serving engine. A system like vLLM makes different choices, such as computing logits only for the last position and managing the KV cache differently.

## 7. Takeaways

1. **Measure the pieces separately.** "`torch.compile` with CUDA graphs" is two optimizations. Here one did 95% of the work.
2. **At batch 1 on a small model, count kernel launches before counting bytes.** The roofline model assumes something is saturated. Here nothing was.
3. **Validate accuracy findings on realistic input.** A synthetic prompt that's perfectly fine for timing manufactured an accuracy problem that didn't exist on real text.
4. **Compare against higher precision, not against another low-precision run.** Two bf16 paths disagreeing says nothing about which one is closer to correct.

## Reproduce

```bash
# performance + accuracy, both prompts, flag on and off
uv run python bench_cuda_graphs.py --real-prompt
uv run python bench_cuda_graphs.py --real-prompt --no-emulate
uv run python bench_cuda_graphs.py
uv run python bench_cuda_graphs.py --no-emulate

# layer-by-layer investigation and attention-kernel sweep
uv run python investigate_fusion.py [--real-prompt] [--emulate]
```

- Scripts: [`bench_cuda_graphs.py`](../bench_cuda_graphs.py), [`investigate_fusion.py`](../investigate_fusion.py)
- Raw results: [`results/`](../results/)
- Full lab notes, including dead ends: [`LOG.md`](../LOG.md)
