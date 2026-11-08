#!/usr/bin/env python3
"""Readiness probe for a vLLM pod. Exit 0 = ready, 1 = not ready.

PROBE_MODE=health : GET /health must return 200 (what the default httpGet probe checks).
PROBE_MODE=decode : /health AND a real 4-token completion must succeed within PROBE_TIMEOUT seconds.

Stdlib only, so it runs inside the stock vllm/vllm-openai image.
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("PROBE_BASE", "http://127.0.0.1:8000")
MODE = os.environ.get("PROBE_MODE", "decode")
TIMEOUT = float(os.environ.get("PROBE_TIMEOUT", "8"))  # total budget for the whole probe
DEADLINE = time.monotonic() + TIMEOUT
# "sampled" exercises the top-p/top-k sampler (the FlashInfer path); "greedy" does not.
SAMPLING = (
    {"temperature": 0}
    if os.environ.get("PROBE_SAMPLING", "sampled") == "greedy"
    else {"temperature": 0.7, "top_p": 0.9, "top_k": 40}
)


def _left():
    return max(0.5, DEADLINE - time.monotonic())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=_left()) as r:
        return r.status, r.read()


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_left()) as r:
        return r.status, r.read()


def fail(msg):
    print(f"NOT READY mode={MODE}: {msg}", file=sys.stderr)
    return 1


def main():
    t0 = time.monotonic()
    try:
        status, _ = get("/health")
        if status != 200:
            return fail(f"/health returned {status}")
        if MODE == "decode":
            _, body = get("/v1/models")
            model = json.loads(body)["data"][0]["id"]
            status, body = post(
                "/v1/completions",
                {"model": model, "prompt": "Hello", "max_tokens": 4, **SAMPLING},
            )
            tokens = json.loads(body)["usage"]["completion_tokens"]
            if status != 200 or tokens < 1:
                return fail(f"completion status={status} tokens={tokens}")
    except Exception as e:  # timeouts, refused connections, HTTP 4xx/5xx, bad JSON
        return fail(f"{type(e).__name__}: {e}")
    print(f"ok mode={MODE} {time.monotonic() - t0:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
