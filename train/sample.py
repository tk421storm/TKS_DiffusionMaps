"""Generate windows from a checkpoint and write ASCII, PNG and metrics.

Usage (inside WSL):
  HSA_ENABLE_DXG_DETECTION=1 ~/venvs/dm/bin/python -u train/sample.py runs/v1_mdm/ckpt_best.pt
      [--n 64] [--steps 16] [--temperature 1.0] [--conf-noise 4.5] [--seed 42] [--out DIR]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import diffusion  # noqa: E402
import metrics  # noqa: E402
import model as model_lib  # noqa: E402
import render  # noqa: E402
from train import ROOT, reference_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--conf-noise", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adjust", type=float, default=0.0,
                    help="logit adjustment strength: add -adjust*log(class weight) to the logits (1 = full)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg, C = state["cfg"], state["num_classes"]
    names = {int(k): v for k, v in state["names"].items()}
    chars = {int(k): v for k, v in state["chars"].items()}
    net = model_lib.build(cfg, C).to(dev)
    net.load_state_dict(state["ema"])
    net.eval()
    val = data.WindowSet(os.path.join(ROOT, cfg["set"]), "val", dev)
    train = data.WindowSet(os.path.join(ROOT, cfg["set"]), "train", dev)
    bias = None
    if args.adjust:
        weights, _ = train.class_weights(cfg["weight_power"], cfg["weight_cap"])
        bias = -args.adjust * torch.log(weights.clamp(min=1e-6))
    g = torch.Generator(device=dev).manual_seed(args.seed)
    samples = diffusion.sample(net, args.n, tuple(val.cells.shape[1:]), C, steps=args.steps,
                               temperature=args.temperature, conf_noise=args.conf_noise,
                               generator=g, device=dev, logit_bias=bias).cpu().numpy()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                   f"sample_e{state['epoch']}_s{args.steps}_t{args.temperature}_c{args.conf_noise}"
                                   f"_a{args.adjust}_seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "samples.txt"), "w", encoding="utf-8") as f:
        f.write(render.ascii_sheet(samples, chars))
    render.png_sheet(samples, names, os.path.join(out, "samples.png"))
    ref, _ = reference_metrics([train, val], names)
    gen = metrics.compute(samples, names)
    json.dump({"real": ref, "generated": gen}, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print(metrics.table(ref, gen))
    print("wrote", out)


if __name__ == "__main__":
    main()
