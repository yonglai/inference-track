# Readiness that means ready (experiment kit)

**Question:** what does a dummy-decode readiness probe catch that vLLM's default `/health` check does not?

**Status:** the scripts were tested against a stub HTTP server only (health OK + decode ok / hang / 500 / no server).
Nothing here has run against real vLLM or k3s yet. Expect to fix small things on the first run.

## Files
| file | purpose |
|---|---|
| `decode_probe.py` | exec readiness probe. `PROBE_MODE=health` = GET /health only; `decode` = /health + a 4-token completion |
| `entrypoint.sh` | starts vLLM; `BREAK_NVCC=1` hides nvcc first |
| `freeze_engine.py` | SIGSTOP/SIGCONT the EngineCore process to simulate a hung engine |
| `client_loop.sh` | 1 request/sec from outside, prints HTTP status + latency (what users would see) |
| `deployment.yaml` | day 3 deployment with the readiness probe swapped for the exec probe |

Only the readiness probe differs from day 3. Startup and liveness stay `httpGet /health`.

## Step 0 result (G1, vLLM v0.30.0, Qwen2.5-1.5B)
- Image has nvcc at /usr/local/cuda/bin/nvcc, CUDA_HOME=/usr/local/cuda, so `BREAK_NVCC` works.
- `enable_jit_warmup=True`, but the log says `JIT kernel warmup finished in 0.00s`: it compiled nothing.
- Same log: `Using FlashInfer for top-p & top-k sampling`. So the FlashInfer sampler is selected, but its kernel was not built at startup.
- `jit_monitor_mode='warn'`: a JIT compile during inference only logs a warning.
- Consequence: startup (and `/health`) may never touch the sampler kernel. The probe must send top_p/top_k (greedy skips that path), so `decode_probe.py` now defaults to `PROBE_SAMPLING=sampled`.

## Step 0b result (second discovery)
- The image has `flashinfer-jit-cache 0.6.18.post1+cu130` (prebuilt kernels). First sampled request was 42 ms, no JIT compile logged.
- So with the stock image, hiding nvcc (S1) is expected to change nothing: the cache shields the sampler.
- New scenario S1b: `REMOVE_JIT_CACHE=1` + `BREAK_NVCC=1` models an image without the prebuilt cache. This is a constructed scenario, say so in the post.

## Step 0: discovery (done, kept for reference)
Start from your running day 3 pod and learn what the default startup already covers:
```bash
kubectl -n vllm logs deploy/vllm-qwen | grep -i -E 'warmup|flashinfer|jit|nvcc|sampler'
kubectl -n vllm exec deploy/vllm-qwen -- vllm serve --help=all 2>/dev/null | grep -i -E 'warmup|flashinfer'
kubectl -n vllm exec deploy/vllm-qwen -- sh -c 'command -v nvcc; echo CUDA_HOME=$CUDA_HOME; python3 --version'
```
Record: is there a JIT warmup step, does the log show FlashInfer being used, does the image even contain nvcc.
If the image has no nvcc, `BREAK_NVCC` does nothing (entrypoint says so) and S1 needs another breakage (see S1 note).

## Setup
```bash
cd ~/study/MLOps/inference-track/vllm-k8s/readiness
kubectl -n vllm create configmap readiness-experiment \
  --from-file=decode_probe.py --from-file=entrypoint.sh --from-file=freeze_engine.py \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n vllm scale deploy/vllm-qwen --replicas=0   # free the GPU first (strategy is Recreate anyway)
kubectl apply -f deployment.yaml
kubectl -n vllm get endpoints vllm-qwen -w            # terminal A: watch endpoints
SVC=$(kubectl -n vllm get svc vllm-qwen -o jsonpath='{.spec.clusterIP}')
./client_loop.sh http://$SVC                          # terminal B: user's view
```
Change a scenario by editing the env values in `deployment.yaml` and re-applying (Recreate restarts the pod).

## Scenarios (run each at least 3 times; note vLLM/image/driver versions in the log)
**S0 baseline** `PROBE_MODE=health`, `BREAK_NVCC=0`. Healthy. Record startup time and time-to-endpoint.
Repeat with `PROBE_MODE=decode`: same, plus how many seconds the decode probe adds. Expected: no difference.

**S1 toolchain broken at start** `BREAK_NVCC=1`, run once with `health`, once with `decode`.
Revised prediction after Step 0: warmup compiled nothing (0.00s), so the pod probably starts and `/health` is 200, but a sampled request (top_p/top_k) fails or stalls on first use.
If so, S1 is already the key result: health-only is ready and broken, decode probe never goes ready. Also send a greedy request (temperature 0) by hand and note whether it still works.
If the pod instead crashes at startup, record that (default path caught it). If sampled requests work anyway, the kernel was prebuilt or cached: check `ls ~/.cache/flashinfer` in the pod.

**S1b no prebuilt cache and no compiler** `REMOVE_JIT_CACHE=1`, `BREAK_NVCC=1`; first with `health`, then `decode`.
Also run `REMOVE_JIT_CACHE=1` alone (nvcc present) to see how long the first sampled request takes while it compiles, and whether the probe's 10 s timeout fails during that compile.
Expectation: startup and `/health` succeed, the first sampled request fails or stalls, only the decode probe notices.

**S2 breakage that survives startup** (the interesting one, needs discovery)
Goal: pod passes startup, `/health` is 200, but some request shape fails. Candidates, try in order:
1. Start healthy, then break after ready: `kubectl -n vllm exec deploy/vllm-qwen -- sh -c 'mv $(command -v nvcc) $(command -v nvcc).off'`, then send requests with `top_p`/`top_k` set or other sampling params that may trigger a lazy JIT variant that startup didn't warm.
2. Fill GPU headroom (raise `--gpu_memory_utilization` to ~0.97) to look for OOM on first long request.
Only count it as a finding if `/health` stays 200 while real requests fail. Write down the exact request that failed.

**S3 hung engine** (mechanism check, no toolchain needed)
```bash
kubectl -n vllm exec deploy/vllm-qwen -- python3 /experiment/freeze_engine.py freeze
# ... watch client_loop, endpoints, pod READY column ...
kubectl -n vllm exec deploy/vllm-qwen -- python3 /experiment/freeze_engine.py thaw
```
Run with `health` and with `decode`. Prediction: `/health` may stay 200 (API server alive) so the health-only pod stays in endpoints while client requests time out; the decode probe drops it from endpoints after ~30 s (3 failures x 10 s) then restores it on thaw.
Note liveness stays `/health`, so it may never restart the pod. That is part of the finding.

## What to capture per run
- client_loop output around the event (first failing timestamp, last good one)
- `kubectl -n vllm get pod -w` READY transitions, with timestamps
- `kubectl -n vllm describe pod` events (Readiness probe failed ...)
- time from fault to endpoint removal

## LOG.md entry
Use the Entry Template in `../../LOG.md`. Put the scenario table in Result (scenario, probe mode, /health, decode, in endpoints?, client errors), and keep the "probe caught nothing extra" outcome if that's what you measure. The blog post follows the data (pillar: `mlops`).

## Cleanup
```bash
kubectl apply -f ../day3/deployment.yaml   # restore the original probes
kubectl -n vllm delete configmap readiness-experiment
```
