"""Step-by-step GPU smoke test; prints (and flushes) after every step so a hang shows where it stopped.

Run inside WSL:
  HSA_ENABLE_DXG_DETECTION=1 LD_LIBRARY_PATH=/opt/rocm/lib ~/venvs/dm/bin/python -u tools/gpu_smoke.py [--from N]
"""
import argparse
import time

T0 = time.perf_counter()


def say(msg):
    print(f"[{time.perf_counter() - T0:7.2f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0, help="skip steps before this number")
    args = ap.parse_args()

    say("importing torch")
    import torch
    import torch.nn.functional as F
    say(f"torch {torch.__version__}, HIP {torch.version.hip}")
    say(f"cuda.is_available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    say(f"device: {torch.cuda.get_device_name(0)}")
    dev = torch.device("cuda")

    steps = [
        ("small tensor to GPU and back", lambda: torch.arange(10, device=dev).sum().item()),
        ("allocate 1 GiB", lambda: torch.empty(256 * 2 ** 20, device=dev).fill_(1).sum().item()),
        ("free cached memory", lambda: torch.cuda.empty_cache()),
        ("matmul 4096x4096 (rocBLAS/hipBLASLt)", lambda: (torch.randn(4096, 4096, device=dev) @ torch.randn(4096, 4096, device=dev)).sum().item()),
        ("conv2d forward 64ch 48x48 (MIOpen)", lambda: F.conv2d(torch.randn(64, 8, 48, 48, device=dev), torch.randn(64, 8, 3, 3, device=dev), padding=1).sum().item()),
        ("conv2d backward", lambda: _conv_backward(torch, F, dev)),
        ("synchronize", lambda: torch.cuda.synchronize()),
    ]
    for i, (name, fn) in enumerate(steps):
        if i < args.start:
            continue
        say(f"step {i}: {name} ...")
        t = time.perf_counter()
        out = fn()
        say(f"step {i}: ok ({time.perf_counter() - t:.2f}s){'' if out is None else f' -> {out:.4g}'}")
    say("ALL STEPS OK")


def _conv_backward(torch, F, dev):
    x = torch.randn(64, 8, 48, 48, device=dev, requires_grad=True)
    w = torch.randn(64, 8, 3, 3, device=dev, requires_grad=True)
    F.conv2d(x, w, padding=1).square().mean().backward()
    return float(w.grad.abs().sum())


if __name__ == "__main__":
    main()
