## 2026-11-08 — Week 11 — readiness probe: /health vs dummy decode

**Machine:** G1 (k3s, single GPU, sm_86 per the FlashInfer build path; driver: <fill>)
**Question:** What does an exec readiness probe that runs a real 4-token sampled completion catch that the default `/health` check does not?

**Setup**

- Model / kernel: Qwen/Qwen2.5-1.5B-Instruct; FlashInfer top-p/top-k sampler
- Framework + version: vllm/vllm-openai:v0.30.0 (vLLM 0.30.0), flashinfer-python 0.6.18.post1, flashinfer-jit-cache 0.6.18.post1+cu130, nvcc at /usr/local/cuda/bin/nvcc
- Precision: bf16
- Batch size / concurrency: client 1 req/s, 4 tokens, temperature 0.7, top_p 0.9, top_k 40, curl --max-time 5
- Input len -> output len: "Hello" -> 4 tokens
- Warmup iters / measured iters: n/a (one pod start per scenario; S0 health x3)
- Probes: startup httpGet /health (5 s, 60 failures), liveness httpGet /health (10 s, 6 failures, 3 s timeout), readiness = variable: `PROBE_MODE=health` (/health only) or `decode` (/health + completion), exec, period 10 s, timeout 10 s, 3 failures
- Explicit synchronize? N/A (HTTP-level test)

**Result** (all times UTC unless noted; n=1 per row except S0 health)

| Scenario | Readiness probe | Outcome |
| --- | --- | --- |
| S0 healthy | health | READY at pod age 50 / 46 / 45 s; first request 42.7 / 42.4 / 40.1 ms; steady ~29 ms |
| S0 healthy | decode | READY at pod age 45 s; probe never failed; first request 28.9 ms |
| S3 engine frozen (SIGSTOP EngineCore) 19:24:40 | decode | 7 client timeouts (5 s each) over 37 s; pod 0/1 and out of endpoints at 19:25:17; then instant connection-refused; thaw 19:26:27, READY and 200 within ~1 s; 0 restarts |
| S3 engine frozen 19:28:45 | health | pod stayed 1/1 and in endpoints for the whole freeze; 18 of 18 client requests timed out (5 s each) until thaw at 19:30:33; 200 at 19:30:34; 0 restarts |
| S3 direct check, frozen | health | `/health` returned 200 in 0.018 s; sampled completion timed out at 5 s |
| S1 nvcc hidden (cache present) | health | no effect: READY at 45 s, requests 200 at ~29 ms |
| S1b nvcc hidden + flashinfer-jit-cache removed | health and decode | EngineCore fails at startup after ~41 s (ninja exit 127 building `sampling` for sm_86), CrashLoopBackOff, never READY; Service had no endpoints throughout (Recreate) |

**Baseline compared against:** S0 healthy, same pod spec, health-only readiness
**Delta:** frozen engine: removed from endpoints after 37 s with decode probe vs never (108 s+ observed) with health only

**Interpretation**

- The default check reports that the API server process is alive, not that the engine makes progress. `/health` stayed 200 (18 ms) while the engine was stopped and completions hung.
- Toolchain breakage did not escape startup: the prebuilt flashinfer-jit-cache shields the stock image (S1), and without it vLLM's own startup runs the sampler and crashes on a missing compiler (S1b). The decode probe adds nothing there.
- The decode probe's value in this test is runtime stalls after a clean startup. The 37 s detection time is failureThreshold (3) x period (10 s) plus timeout, so it is tunable.
- Liveness on /health never fired during a 108 s freeze (failureThreshold 6 x 10 s = 60 s), consistent with /health staying 200.

**Surprises / what I got wrong**

- I predicted S1b would start fine and fail on the first sampled request. Instead vLLM's startup already exercises the sampler and crashes. Startup showed `JIT kernel warmup finished in 0.00s`, but a later warmup step still built the sampling module.
- My first probe used temperature 0 (greedy), which would not have touched the FlashInfer sampler. Fixed to sample with top_p/top_k.
- First user request was ~29 ms behind the decode probe vs 40-43 ms behind the health probe: the probe's own request warms the path (n=1 for decode).
- **Confusions to revisit:** why `/health` stays 200 when EngineCore is stopped (API server and engine are separate processes); which warmup step builds the sampler (read the rest of the traceback in the crash log).

- **Repro:** `vllm-k8s/readiness/README.md` · commit <fill> · client `./client_loop.sh http://$SVC` · freeze `freeze_engine.py freeze|thaw`

