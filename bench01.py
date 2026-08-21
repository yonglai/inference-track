import torch, time

dev = "cuda" if torch.cuda.is_available() else "mps"
N, ITERS, WARMUP = 4096, 50, 10
a = torch.randn(N, N, device=dev, dtype=torch.float16)
b = torch.randn(N, N, device=dev, dtype=torch.float16)

def timed(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS

t_mm = timed(lambda: a @ b)
t_add = timed(lambda: a + b)

print(f"matmul {2*N**3/t_mm/1e12:8.2f} TFLOP/s      ({2*N**3/t_mm/1e12/59.5*100:.1f}% of peak)")
print(f"add {3*N*N*2/t_add/1e9:8.2f} GB/s           ({3*N*N*2/t_add/1e9/706*100:.1f}% of peak)")