#!/bin/sh
# Container entrypoint for the readiness experiment.
# BREAK_NVCC=1 hides nvcc before vLLM starts, to simulate a missing/broken toolchain for JIT compiles.
if [ "${BREAK_NVCC:-0}" = "1" ]; then
  found=0
  for n in "$(command -v nvcc 2>/dev/null)" "${CUDA_HOME:-/nonexistent}/bin/nvcc" /usr/local/cuda/bin/nvcc; do
    if [ -n "$n" ] && [ -x "$n" ]; then
      mv "$n" "$n.off" && echo "[experiment] hid $n" && found=1
    fi
  done
  [ "$found" = "1" ] || echo "[experiment] BREAK_NVCC=1 but no nvcc found: nothing was broken"
fi
echo "[experiment] starting vllm (PROBE_MODE=${PROBE_MODE:-unset} BREAK_NVCC=${BREAK_NVCC:-0})"
exec vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --dtype bfloat16 \
  --gpu_memory_utilization 0.85 --no-enable-prefix-caching
