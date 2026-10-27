"""Check that PyTorch sees the GPU, then time a small U-Net on the v1 windows (GPU vs CPU).

Run inside WSL with the ROCm venv:
  HSA_ENABLE_DXG_DETECTION=1 ~/venvs/dm/bin/python tools/bench_gpu.py [--set data/sets/v1_48] [--steps 50]

The model is a stand-in of roughly the size we expect for v1 (a few M parameters), trained on
one-hot category windows with a dummy denoising target. Only speed and memory are measured.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class SmallUNet(nn.Module):
    def __init__(self, classes, width=64):
        super().__init__()
        w = width
        self.d1, self.d2, self.d3 = block(classes + 1, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.mid = block(4 * w, 4 * w)
        self.u3, self.u2, self.u1 = block(8 * w, 2 * w), block(4 * w, w), block(2 * w, w)
        self.out = nn.Conv2d(w, classes, 1)

    def forward(self, x, t):
        # t (noise level) as an extra constant channel: enough for a speed test.
        x = torch.cat([x, t[:, None, None, None].expand(-1, 1, *x.shape[2:])], 1)
        a = self.d1(x)                       # 48
        b = self.d2(F.max_pool2d(a, 2))      # 24
        c = self.d3(F.max_pool2d(b, 2))      # 12
        m = self.mid(c)
        c = self.u3(torch.cat([m, c], 1))
        b = self.u2(torch.cat([F.interpolate(c, scale_factor=2), b], 1))
        a = self.u1(torch.cat([F.interpolate(b, scale_factor=2), a], 1))
        return self.out(a)


def run(device, cells, classes, batch, steps):
    torch.manual_seed(0)
    model = SmallUNet(classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    n = len(cells)
    # Whole dataset lives on the device: per-step host->device copies hung the WSL GPU bridge
    # (vmbus_teardown_gpadl) on Windows 10. 6k windows x 48x48 uint8 is only ~14 MB.
    data = torch.from_numpy(cells).to(device)
    def step():
        idx = torch.randint(0, n, (batch,), device=device)
        x0 = F.one_hot(data[idx].long(), classes).permute(0, 3, 1, 2).float()
        t = torch.rand(batch, device=device)
        noisy = x0 * (1 - t[:, None, None, None]) + torch.randn_like(x0) * t[:, None, None, None]
        loss = F.cross_entropy(model(noisy, t), x0.argmax(1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss
    print(f"  {device.type}: warm-up", flush=True)
    for _ in range(3):                       # warm-up (kernel selection, allocation)
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  {device.type}: timing {steps} steps", flush=True)
    t0 = time.perf_counter()
    for i in range(steps):
        loss = step()
        if (i + 1) % 10 == 0:
            print(f"    step {i + 1}/{steps}", flush=True)
    loss.item()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    params = sum(p.numel() for p in model.parameters()) / 1e6
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0
    print(f"  {device.type}: {dt * 1000:7.1f} ms/step, {batch / dt:7.0f} windows/s, "
          f"{params:.1f}M params" + (f", peak {mem:.2f} GiB" if mem else ""))
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=os.path.join(HERE, "..", "data", "sets", "v1_48"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cpu-steps", type=int, default=5)
    args = ap.parse_args()

    print("torch", torch.__version__, "| HIP", torch.version.hip, "| GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  device 0: {p.name}, {p.total_memory / 2 ** 30:.1f} GiB, arch {getattr(p, 'gcnArchName', '?')}")
        x = torch.randn(4096, 4096, device="cuda")
        print("  matmul check:", float((x @ x).sum().isfinite()))

    cells = np.load(os.path.join(args.set, "train.npz"))["cells"]
    classes = int(cells.max()) + 1
    print(f"data: {len(cells)} windows of {cells.shape[1]}x{cells.shape[2]}, {classes} categories, batch {args.batch}")
    gpu = run(torch.device("cuda"), cells, classes, args.batch, args.steps) if torch.cuda.is_available() else None
    cpu = run(torch.device("cpu"), cells, classes, args.batch, args.cpu_steps)
    if gpu:
        print(f"GPU is {cpu / gpu:.0f}x faster; one epoch ({len(cells)} windows) "
              f"~{len(cells) / args.batch * gpu:.0f} s on GPU vs ~{len(cells) / args.batch * cpu / 60:.0f} min on CPU")


if __name__ == "__main__":
    main()
