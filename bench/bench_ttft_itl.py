"""
Session 6 (AI-101, Prefill Versus Decode) — TTFT / ITL instrumentation.

Separates a generation call into:
  - prefill      : one forward pass over the full prompt, builds the KV cache
  - first token  : sample + detokenize  -> TTFT boundary
  - decode loop  : one forward pass per token, batch-1 matrix-vector work

Every timed region is synchronized. Greedy sampling and a forced token count
keep runs comparable across a sweep.
"""

import json
import platform
import time
from dataclasses import dataclass, field, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------- utilities

def _sync(device: torch.device) -> None:
    """Block until all queued work on `device` has actually completed."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Clock:
    """perf_counter around a synchronized region."""

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


# ---------------------------------------------------------------- result

@dataclass
class RunResult:
    prompt_tokens: int
    gen_tokens: int
    dtype: str

    tokenize_s: float = 0.0
    prefill_s: float = 0.0          # forward pass over prompt only
    first_sample_s: float = 0.0     # argmax + detokenize of token 1
    ttft_s: float = 0.0             # tokenize + prefill + first_sample
    itl_s: list = field(default_factory=list)   # per-token, tokens 2..N

    def summary(self) -> dict:
        n = len(self.itl_s)
        s = sorted(self.itl_s)
        out = asdict(self)
        if n:
            out["itl_median_ms"] = s[n // 2] * 1e3
            out["itl_p95_ms"] = s[min(n - 1, int(0.95 * n))] * 1e3
            out["itl_first3_ms"] = [x * 1e3 for x in self.itl_s[:3]]
            out["decode_tok_per_s"] = n / sum(self.itl_s)
        out["ttft_ms"] = self.ttft_s * 1e3
        return out


# ---------------------------------------------------------------- core

@torch.inference_mode()
def measure(model, tok, prompt: str, gen_tokens: int, device) -> RunResult:
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")

    # --- tokenize -------------------------------------------------------
    with Clock(device) as c:
        enc = tok(prompt, return_tensors="pt").to(device)
    tokenize_s = c.dt

    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", torch.ones_like(input_ids))
    n_prompt = input_ids.shape[1]

    r = RunResult(prompt_tokens=n_prompt, gen_tokens=gen_tokens, dtype=dtype)
    r.tokenize_s = tokenize_s

    # --- prefill: matrix-matrix, compute-bound --------------------------
    with Clock(device) as c:
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    r.prefill_s = c.dt

    cache = out.past_key_values
    logits = out.logits[:, -1, :]

    # --- first token: sample + detokenize -> closes the TTFT boundary ----
    with Clock(device) as c:
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        _ = tok.decode(next_id[0], skip_special_tokens=True)
    r.first_sample_s = c.dt

    r.ttft_s = r.tokenize_s + r.prefill_s + r.first_sample_s

    # --- decode: matrix-vector, memory-bound ----------------------------
    # EOS is ignored on purpose: every run must emit exactly `gen_tokens`.
    for _ in range(gen_tokens - 1):
        attn = torch.cat([attn, torch.ones_like(next_id)], dim=1)
        with Clock(device) as c:
            out = model(input_ids=next_id, attention_mask=attn,
                        past_key_values=cache, use_cache=True)
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            _ = tok.decode(next_id[0], skip_special_tokens=True)
        cache = out.past_key_values
        r.itl_s.append(c.dt)

    return r

def make_prompt(tok, n_tokens):
    ids = tok(" the" * (n_tokens * 2))["input_ids"][:n_tokens]
    prompt = tok.decode(ids)
    actual = len(tok(prompt)["input_ids"])
    assert abs(actual - n_tokens) <= 2, f"got {actual}, wanted {n_tokens}"
    return prompt

def run(model_id: str, prompt_tokens: int, gen_tokens: int,
        dtype=torch.bfloat16, warmup: int = 3, reps: int = 5,
        device_str: str = "cuda"):
    device = torch.device(device_str)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype).to(device).eval()

    # A prompt of a controlled token length. Verify, do not assume.
    prompt = make_prompt(tok, prompt_tokens)

    for _ in range(warmup):
        measure(model, tok, prompt, min(gen_tokens, 8), device)

    results = [measure(model, tok, prompt, gen_tokens, device) for _ in range(reps)]

    meta = {
        "model": model_id,
        "dtype": str(dtype).replace("torch.", ""),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else device_str,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": torch.cuda.get_driver_version() if hasattr(torch.cuda, "get_driver_version") else None,
        "python": platform.python_version(),
        "warmup": warmup,
        "reps": reps,
    }
    return {"meta": meta, "runs": [r.summary() for r in results]}


if __name__ == "__main__":
    # out = run("EleutherAI/pythia-410m", prompt_tokens=512, gen_tokens=64)
    # with open("../results/bench_ttft_itl.json", "w") as f:
    #     json.dump(out, f, indent=2)

    # for r in out["runs"]:
    #     print(f"TTFT {r['ttft_ms']:6.2f} ms   "
    #         f"ITL {r['itl_median_ms']:5.2f} ms   "
    #         f"{r['decode_tok_per_s']:6.1f} tok/s")

    # Sweep over a few models, print the median ITL and decode throughput.
    # for m in ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b"]:
    #     out = run(m, prompt_tokens=512, gen_tokens=64, reps=3)
    #     r = out["runs"][0]
    #     print(f"{m:30s} ITL {r['itl_median_ms']:6.2f} ms  {r['decode_tok_per_s']:6.1f} tok/s")

    # Sweep over a few prompt lengths, print the median ITL and decode throughput.
    for prompt_tokens in [128, 512, 1024, 2048]:
        out = run("EleutherAI/pythia-410m", prompt_tokens=prompt_tokens, gen_tokens=64)
        with open(f"../results/bench_ttft_itl-{prompt_tokens}.json", "w") as f:
            json.dump(out, f, indent=2)

        for r in out["runs"]:
            print(f"Prompt {prompt_tokens:4d} tok   "
                f"TTFT {r['ttft_ms']:6.2f} ms   "
                f"ITL {r['itl_median_ms']:5.2f} ms   "
                f"{r['decode_tok_per_s']:6.1f} tok/s")
