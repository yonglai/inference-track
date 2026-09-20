"""
A/B: DynamicCache (concat, reallocates every step) vs StaticCache (preallocated,
in-place writes). Measures ITL and counts CUDA kernel launches for both.

Drop alongside bench_ttft_itl.py — reuses Clock/_sync from it.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from torch.profiler import profile, ProfilerActivity

from bench_ttft_itl import Clock, make_prompt   # or paste them in


@torch.inference_mode()
def decode_dynamic(model, input_ids, gen_tokens, device):
    """Baseline: cache grows by concatenation each step."""
    attn = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    cache = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)

    itl = []
    for _ in range(gen_tokens - 1):
        attn = torch.cat([attn, torch.ones_like(next_id)], dim=1)
        with Clock(device) as c:
            out = model(input_ids=next_id, attention_mask=attn,
                        past_key_values=cache, use_cache=True)
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cache = out.past_key_values
        itl.append(c.dt)
    return itl


@torch.inference_mode()
def decode_static(model, input_ids, gen_tokens, device, cache, dtype):
    """StaticCache: preallocated to max_cache_len, written in place.

    cache_position is REQUIRED — the cache tensor is always max_cache_len long,
    so the model cannot infer the current position from its shape the way it
    can with DynamicCache.
    """
    cache.reset()                      # critical between runs
    n_prompt = input_ids.shape[1]

    pos = torch.arange(n_prompt, device=device)
    out = model(input_ids=input_ids, past_key_values=cache,
                cache_position=pos, use_cache=True)
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)

    itl = []
    for k in range(gen_tokens - 1):
        pos = torch.tensor([n_prompt + k], device=device)
        with Clock(device) as c:
            out = model(input_ids=next_id, past_key_values=cache,
                        cache_position=pos, use_cache=True)
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        itl.append(c.dt)
    return itl


def count_launches(fn):
    """Kernel launches + GPU busy time for one decode step."""
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
    evts = prof.key_averages()
    launches = sum(e.count for e in evts if e.key == "cudaLaunchKernel")
    gpu_us = sum(e.self_device_time_total for e in evts
                 if e.device_type == torch.autograd.DeviceType.CUDA)
    cats = sum(e.self_device_time_total for e in evts if "Cat" in e.key)
    return launches, gpu_us, cats


def main(model_id="EleutherAI/pythia-410m", prompt_tokens=512, gen_tokens=64,
         dtype=torch.bfloat16, device_str="cuda", warmup=3, reps=5):
    device = torch.device(device_str)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()

    prompt = make_prompt(tok, prompt_tokens)
    input_ids = tok(prompt, return_tensors="pt").to(device)["input_ids"]

    static = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=prompt_tokens + gen_tokens + 8,
        device=device,
        dtype=dtype,
    )

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2] * 1e3

    for name, fn in [
        ("DynamicCache", lambda: decode_dynamic(model, input_ids, gen_tokens, device)),
        ("StaticCache",  lambda: decode_static(model, input_ids, gen_tokens, device,
                                               static, dtype)),
    ]:
        for _ in range(warmup):
            fn()
        meds = [med(fn()) for _ in range(reps)]
        best = sorted(meds)[len(meds) // 2]
        print(f"{name:14s} ITL {best:6.3f} ms   {1e3/best:6.1f} tok/s")

    # launch + GPU-time comparison on a single decode step
    print()
    for name, fn in [
        ("DynamicCache", lambda: decode_dynamic(model, input_ids, 2, device)),
        ("StaticCache",  lambda: decode_static(model, input_ids, 2, device,
                                               static, dtype)),
    ]:
        launches, gpu_us, cats = count_launches(fn)
        print(f"{name:14s} {launches:4d} launches   "
              f"GPU {gpu_us:7.0f} us   cat {cats:6.0f} us")


if __name__ == "__main__":
    main()