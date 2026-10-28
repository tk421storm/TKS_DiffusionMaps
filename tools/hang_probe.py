"""Probe for the WSL/ROCDXG exit hang (process stuck in vmbus_teardown_gpadl after finishing).

Usage: python -u tools/hang_probe.py <mode>
  small    - a few .item() calls, normal exit
  big      - 5 x 128 MB GPU->CPU copies, normal exit
  pinned   - 5 x 128 MB copies through one reused pinned buffer, normal exit
  big_free - like big, then drop tensors, synchronize and empty_cache before exit
  big_exit - like big_free, then os._exit(0) (skips Python/HIP teardown)
Prints "copies done" then "exiting"; a hang after "exiting" is the teardown hang.
"""
import os
import sys
import time

import torch

mode = sys.argv[1]
dev = torch.device("cuda")
x = torch.randn(32 * 2 ** 20, device=dev)          # 128 MB of float32
if mode == "small":
    for _ in range(5):
        x.sum().item()
elif mode == "pinned":
    buf = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
    for _ in range(5):
        buf.copy_(x)
else:
    for _ in range(5):
        y = x.cpu()
print("copies done", flush=True)
if mode in ("big_free", "big_exit"):
    del x
    y = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
print("exiting", flush=True)
t = time.perf_counter()
if mode == "big_exit":
    os._exit(0)
