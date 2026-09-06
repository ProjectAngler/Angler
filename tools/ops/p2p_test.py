"""Before/after P2P check between her two GPUs: peer access flags, then a
device-to-device copy bandwidth measurement (64 MB)."""
import time
import torch

assert torch.cuda.device_count() >= 2, "need two GPUs visible"
print("peer access 0->1:", torch.cuda.can_device_access_peer(0, 1), " 1->0:", torch.cuda.can_device_access_peer(1, 0))
n = 64 * 1024 * 1024 // 4
a = torch.ones(n, dtype=torch.float32, device="cuda:0")
b = torch.empty(n, dtype=torch.float32, device="cuda:1")
for _ in range(3):
    b.copy_(a, non_blocking=True)
torch.cuda.synchronize(0); torch.cuda.synchronize(1)
iters = 20
t0 = time.perf_counter()
for _ in range(iters):
    b.copy_(a, non_blocking=True)
torch.cuda.synchronize(0); torch.cuda.synchronize(1)
dt = time.perf_counter() - t0
gb = iters * n * 4 / 1e9
print(f"cuda:0 -> cuda:1  {gb/dt:.1f} GB/s over {iters} x 64 MB")
t0 = time.perf_counter()
for _ in range(iters):
    a.copy_(b, non_blocking=True)
torch.cuda.synchronize(0); torch.cuda.synchronize(1)
dt = time.perf_counter() - t0
print(f"cuda:1 -> cuda:0  {gb/dt:.1f} GB/s")
print("checksum ok:", bool(torch.all(b == 1).item()))
