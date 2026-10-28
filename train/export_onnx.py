"""Export a checkpoint's EMA model to ONNX and check it the way the game would run it (CPU, ONNX Runtime).

Checks:
  1. logits: PyTorch vs ONNX Runtime on random partly-masked inputs (max abs difference)
  2. sampling: deterministic (argmax) sampling in PyTorch vs a numpy + ONNX Runtime sampler that
     mirrors what the C# side will do; reports the share of identical cells
  3. timing on CPU, batch 1: one forward pass and a full sampling run, with 1 thread (a background
     thread in game) and with all threads

Usage (inside WSL, needs `pip install onnx onnxruntime` in the venv):
  ~/venvs/dm/bin/python -u train/export_onnx.py runs/v1_mdm/ckpt_best.pt [--steps 16] [--opset 17]
Writes model.onnx next to the checkpoint and onnx_check.json with the results.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusion  # noqa: E402
import model as model_lib  # noqa: E402


def load(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = model_lib.build(state["cfg"], state["num_classes"])
    net.load_state_dict(state["ema"])
    net.eval()
    return net, state


def ort_sample(session, n, size, num_classes, steps, logit_bias=None):
    """Deterministic MaskGIT sampler in numpy + ONNX Runtime; the reference for the C# port."""
    H, W = size
    mask_token = num_classes
    x = np.full((n, H, W), mask_token, np.int64)
    start = (x == mask_token).reshape(n, -1).sum(1)
    for k in range(steps):
        masked = x == mask_token
        ratio = masked.reshape(n, -1).mean(1).astype(np.float32)
        logits = session.run(None, {"tokens": x, "ratio": ratio})[0].transpose(0, 2, 3, 1)
        if logit_bias is not None:
            logits = logits + logit_bias
        logits = logits - logits.max(-1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        pick, conf = logp.argmax(-1), logp.max(-1)
        conf = np.where(masked, conf, np.inf)
        keep_hidden = np.floor(math.cos((k + 1) / steps * math.pi / 2) * start).astype(np.int64)
        flat = conf.reshape(n, -1)
        kth = np.take_along_axis(np.sort(flat, 1), np.minimum(keep_hidden, flat.shape[1] - 1)[:, None], 1)
        stay = (flat < kth).reshape(n, H, W) & (keep_hidden > 0)[:, None, None]
        x = np.where(masked & ~stay, pick, x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--adjust", type=float, default=0.0,
                    help="logit adjustment strength stored in model_meta.json and used by both samplers")
    args = ap.parse_args()
    import onnxruntime as ort
    import data
    from train import ROOT

    net, state = load(args.ckpt)
    C = state["num_classes"]
    win = state["cfg"].get("window", 48)
    out_dir = os.path.dirname(os.path.abspath(args.ckpt))
    onnx_path = os.path.join(out_dir, "model.onnx")
    cfg = state["cfg"]
    bias = np.zeros(C, np.float32)
    if args.adjust:
        weights, _ = data.WindowSet(os.path.join(ROOT, cfg["set"]), "train", "cpu").class_weights(
            cfg["weight_power"], cfg["weight_cap"])
        bias = (-args.adjust * np.log(np.maximum(weights.numpy(), 1e-6))).astype(np.float32)
    # Everything the in-game sampler needs besides the network itself.
    meta = {"window": win, "num_classes": C, "mask_token": C,
            "categories": {int(k): v for k, v in state["names"].items()},
            "inputs": {"tokens": "int64 [batch, H, W] (row 0 = south)", "ratio": "float32 [batch] masked fraction"},
            "output": "logits float32 [batch, C, H, W]",
            "logit_bias": bias.tolist(), "adjust": args.adjust,
            "sampler": {"steps": args.steps, "schedule": "keep cos(pi/2*(k+1)/steps) of the free cells masked",
                        "temperature": cfg.get("temperature", 1.0), "conf_noise": cfg.get("conf_noise", 4.5)},
            "checkpoint": os.path.basename(args.ckpt), "epoch": state["epoch"], "val_loss": state["val_loss"]}
    json.dump(meta, open(os.path.join(out_dir, "model_meta.json"), "w"), indent=1)

    tokens = torch.randint(0, C, (2, win, win))
    tokens[torch.rand(tokens.shape) < 0.5] = C
    ratio = (tokens == C).float().mean(dim=(1, 2))
    torch.onnx.export(net, (tokens, ratio), onnx_path, input_names=["tokens", "ratio"], output_names=["logits"],
                      dynamic_axes={"tokens": {0: "batch"}, "ratio": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=args.opset, dynamo=False)
    size_mb = os.path.getsize(onnx_path) / 2 ** 20
    print(f"exported {onnx_path} ({size_mb:.1f} MB, opset {args.opset})")
    report = {"onnx": onnx_path, "size_mb": round(size_mb, 2), "opset": args.opset, "ort": ort.__version__}

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # 1. logits
    g = torch.Generator().manual_seed(0)
    t = torch.randint(0, C, (args.n, win, win), generator=g)
    for i, p in enumerate(np.linspace(0.1, 1.0, args.n)):
        t[i][torch.rand((win, win), generator=g) < p] = C
    r = (t == C).float().mean(dim=(1, 2))
    with torch.no_grad():
        ref = net(t, r).numpy()
    got = sess.run(None, {"tokens": t.numpy(), "ratio": r.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    print(f"1. logits: max abs diff {diff:.2e}, argmax agreement {agree:.4%}")
    report["logits_max_abs_diff"], report["logits_argmax_agreement"] = diff, agree

    # 2. deterministic sampling, PyTorch vs numpy + ORT
    with torch.no_grad():
        ts = diffusion.sample(net, args.n, (win, win), C, steps=args.steps, device="cpu", deterministic=True,
                              logit_bias=torch.from_numpy(bias)).numpy()
    os_ = ort_sample(sess, args.n, (win, win), C, args.steps, bias)
    same = float((ts == os_).mean())
    print(f"2. deterministic sampling ({args.steps} steps, {args.n} windows): {same:.4%} identical cells")
    report["sampling_identical_cells"] = same

    # 3. CPU timing, batch 1
    timing = {}
    for threads in (1, 0):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        s = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        one = {"tokens": t[:1].numpy(), "ratio": r[:1].numpy()}
        for _ in range(3):
            s.run(None, one)
        t0 = time.perf_counter()
        for _ in range(20):
            s.run(None, one)
        fwd = (time.perf_counter() - t0) / 20
        t0 = time.perf_counter()
        ort_sample(s, 1, (win, win), C, args.steps, bias)
        full = time.perf_counter() - t0
        label = "1 thread" if threads == 1 else "all threads"
        print(f"3. CPU {label}: {fwd * 1000:.1f} ms per forward, {full:.2f} s per {args.steps}-step sample")
        timing[label] = {"forward_ms": round(fwd * 1000, 2), "sample_s": round(full, 3)}
    report["cpu_timing_batch1"] = timing
    json.dump(report, open(os.path.join(out_dir, "onnx_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
