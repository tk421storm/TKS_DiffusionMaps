"""Train the v1 masked discrete diffusion model on a window set.

Run inside WSL (see tools/README.md for the ROCm setup):
  HSA_ENABLE_DXG_DETECTION=1 ~/venvs/dm/bin/python -u train/train.py configs/train_v1.json [--epochs N]

Writes runs/<name>/: config.json, log.jsonl, ckpt_last.pt, ckpt_best.pt (best validation loss, EMA
weights), and every eval_every epochs samples_eNNN.txt / .png plus metrics_eNNN.json comparing
generated windows with clean real ones (no masked cells).
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
import data  # noqa: E402
import diffusion  # noqa: E402
import metrics  # noqa: E402
import model as model_lib  # noqa: E402
import render  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULTS = {
    "base": 64, "mults": [1, 2, 4], "nres": 2, "attn_levels": [2], "heads": 4, "dropout": 0.1,
    "batch": 64, "epochs": 150, "lr": 3e-4, "wd": 0.01, "warmup_steps": 500, "ema": 0.999,
    "weight_power": 0.5, "weight_cap": 20.0, "grad_clip": 1.0,
    "eval_every": 10, "eval_ratios": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    "sample_n": 64, "sample_steps": 16, "temperature": 1.0, "conf_noise": 4.5, "seed": 1,
}


def reference_metrics(sets, names, limit=1024, seed=0):
    """Metrics on real windows without masked cells (complete layouts only)."""
    clean = []
    for s in sets:
        keep = s.loss_mask_np.all(axis=(1, 2))
        clean.append(s.cells_np[keep])
    clean = np.concatenate(clean)
    rng = np.random.default_rng(seed)
    if len(clean) > limit:
        clean = clean[rng.choice(len(clean), limit, replace=False)]
    return metrics.compute(clean, names), len(clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(json.load(open(args.config, encoding="utf-8")))
    if args.epochs:
        cfg["epochs"] = args.epochs
    out = os.path.join(ROOT, "runs", cfg["name"])
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"])
    set_dir = os.path.join(ROOT, cfg["set"])
    train_set = data.WindowSet(set_dir, "train", dev)
    val_set = data.WindowSet(set_dir, "val", dev)
    C, mask_token = train_set.num_classes, train_set.mask_token
    weights, freq = train_set.class_weights(cfg["weight_power"], cfg["weight_cap"])
    print(f"device {dev}; train {len(train_set)} / val {len(val_set)} windows; {C} categories")
    print("class weights:", {train_set.names[i]: round(float(w), 2) for i, w in enumerate(weights.tolist())})

    net = model_lib.build(cfg, C).to(dev)
    ema = model_lib.build(cfg, C).to(dev)
    ema.load_state_dict(net.state_dict())
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"model: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters")

    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps_per_epoch = len(train_set) // cfg["batch"]
    total = steps_per_epoch * cfg["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg["warmup_steps"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    ref, n_ref = reference_metrics([train_set, val_set], train_set.names)
    print(f"reference metrics from {n_ref} clean real windows")
    log = open(os.path.join(out, "log.jsonl"), "a")
    best = float("inf")
    ema_params, net_params = list(ema.parameters()), list(net.parameters())

    for epoch in range(1, cfg["epochs"] + 1):
        net.train()
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=dev)
        for x0, lm in train_set.batches(cfg["batch"], gen):
            loss = diffusion.masked_loss(net, x0, lm, weights, mask_token, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            torch._foreach_lerp_(ema_params, net_params, 1.0 - cfg["ema"])
            loss_sum += loss.detach()
        train_loss = float(loss_sum) / steps_per_epoch     # one device->host sync per epoch

        res = diffusion.eval_loss(ema, val_set.cells, val_set.loss_mask, weights, mask_token,
                                  cfg["eval_ratios"], seed=1234)
        val_loss = float(sum(v[0] for v in res.values()) / len(res))
        val_acc = {p: round(float(v[1]), 4) for p, v in res.items()}
        dt = time.perf_counter() - t0
        rec = {"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5),
               "val_acc": val_acc, "lr": sched.get_last_lr()[0], "sec": round(dt, 1)}
        print(f"epoch {epoch:4d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"acc@0.5 {val_acc.get(0.5, 0):.3f} acc@1.0 {val_acc.get(1.0, 0):.3f}  {dt:.1f}s", flush=True)

        state = {"cfg": cfg, "num_classes": C, "names": train_set.names, "chars": train_set.chars,
                 "ema": ema.state_dict(), "net": net.state_dict(), "epoch": epoch, "val_loss": val_loss}
        torch.save(state, os.path.join(out, "ckpt_last.pt"))
        if val_loss < best:
            best = val_loss
            torch.save(state, os.path.join(out, "ckpt_best.pt"))
            rec["best"] = True

        if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
            g = torch.Generator(device=dev).manual_seed(42)
            win = val_set.cells.shape[1:]
            samples = diffusion.sample(ema, cfg["sample_n"], tuple(win), C, steps=cfg["sample_steps"],
                                       temperature=cfg["temperature"], conf_noise=cfg["conf_noise"],
                                       generator=g, device=dev).cpu().numpy()
            gen_m = metrics.compute(samples, train_set.names)
            tag = f"e{epoch:03d}"
            with open(os.path.join(out, f"samples_{tag}.txt"), "w", encoding="utf-8") as f:
                f.write(render.ascii_sheet(samples[:8], train_set.chars))
            render.png_sheet(samples, train_set.names, os.path.join(out, f"samples_{tag}.png"))
            json.dump({"real": ref, "generated": gen_m}, open(os.path.join(out, f"metrics_{tag}.json"), "w"), indent=1)
            print(metrics.table(ref, gen_m), flush=True)
            rec["metrics"] = gen_m
        log.write(json.dumps(rec) + "\n")
        log.flush()
    print(f"done; best val loss {best:.4f}; outputs in {out}")
    log.close()


def hard_exit(code=0):
    """Exit without Python/HIP teardown. Under WSL + ROCDXG (Windows 10) the normal teardown of a
    training process could hang in the kernel (vmbus_teardown_gpadl, unkillable D state)."""
    sys.stdout.flush()
    sys.stderr.flush()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        hard_exit(1)
    hard_exit(0)
