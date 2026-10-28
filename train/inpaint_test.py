"""Compare ways of generating a 36x36 settlement square with the 48x48 model (the in-game compare tool's setting).

Variants:
  ring        cells outside the centred square fixed empty (what the 1b compare tool does)
  free        nothing fixed; sample the whole window, keep the 36x36 crop that cuts the fewest wall lines
  seeds<k>    nothing fixed except k L-shaped wall corners scattered in the centred square; keep that square
  <shape><k>  the same with other seed shapes: corner (= seeds), diag (two walls touching corner to corner),
              stair (a 4-connected diagonal staircase of 5-7 cells); "mix" = one corner + one diag

Metrics are computed on the 36x36 crops (metrics.py; rooms cut by the crop edge don't count as rooms), plus:
  cut_cells        wall-like cells on the crop border that continue outside it (walls the crop slices through)
  seed_in_room     share of seeds that end up next to an enclosed room (completed by the model)
  corner_share     wall-like cells whose two wall-like 4-neighbours are perpendicular (turns; staircases raise it)
  diag_joints      wall-like cells per window touching another only corner to corner (diagonal, not sealing)
Real reference: centre 36x36 crops of the validation windows.

Usage (CPU is enough): python -u train/inpaint_test.py runs/v1_mdm/ckpt_best.pt [--n 48] [--adjust 0.6]
Writes runs/<ckpt dir>/inpaint_test/{metrics.json, <variant>.png, <variant>.txt}.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import diffusion  # noqa: E402
import metrics  # noqa: E402
import model as model_lib  # noqa: E402
import render  # noqa: E402
from train import ROOT  # noqa: E402

SQUARE = 36


def wall_like_ids(names):
    return [k for k, v in names.items() if v in ("wall", "door", "fence", "wall_mounted")]


def best_crop(grid, wl_ids, size):
    """Offset of the size x size crop with the most structure and the fewest wall lines cut by its edge."""
    wl = np.isin(grid, wl_ids)
    best, best_score = (0, 0), None
    H = grid.shape[0]
    for oz in range(H - size + 1):
        for ox in range(H - size + 1):
            inside = wl[oz:oz + size, ox:ox + size].sum()
            score = inside - 3 * cut_cells(wl, oz, ox, size)
            if best_score is None or score > best_score:
                best, best_score = (oz, ox), score
    return best


def cut_cells(wl, oz, ox, size):
    """Wall-like cells on the crop border with a wall-like neighbour just outside the crop."""
    H = wl.shape[0]
    n = 0
    for i in range(size):
        for (z, x, dz, dx) in ((oz, ox + i, -1, 0), (oz + size - 1, ox + i, 1, 0),
                               (oz + i, ox, 0, -1), (oz + i, ox + size - 1, 0, 1)):
            zz, xx = z + dz, x + dx
            if wl[z, x] and 0 <= zz < H and 0 <= xx < H and wl[zz, xx]:
                n += 1
    return n


def seed_shape(rng, shape, cz, cx):
    """Cells of one seed anchored at (cz, cx); the anchor is the first cell. Orientation is random."""
    dz, dx = rng.choice([-1, 1], 2)
    if shape == "corner":
        az, ax = rng.integers(3, 6, 2)
        return [(cz + dz * i, cx) for i in range(az)] + [(cz, cx + dx * i) for i in range(1, ax)]
    if shape == "diag":
        return [(cz, cx), (cz + dz, cx + dx)]
    if shape == "stair":
        cells, z, x = [(cz, cx)], cz, cx
        for i in range(rng.integers(5, 8) - 1):
            if i % 2 == 0:
                x += dx
            else:
                z += dz
            cells.append((z, x))
        return cells
    raise ValueError(shape)


def scatter_seeds(rng, shapes, lo, hi, min_dist=8):
    """One seed per entry of shapes, anchors at least min_dist apart, all cells inside [lo, hi)."""
    seeds, anchors = [], []
    tries = 0
    while len(seeds) < len(shapes) and tries < 1000:
        tries += 1
        cz, cx = rng.integers(lo + 1, hi - 1, 2)
        if any(abs(cz - z) + abs(cx - x) < min_dist for z, x in anchors):
            continue
        cells = seed_shape(rng, shapes[len(seeds)], cz, cx)
        if all(lo <= z < hi and lo <= x < hi for z, x in cells):
            seeds.append(cells)
            anchors.append((cz, cx))
    return seeds


def variant_shapes(variant):
    """Seed shapes for a variant name, or None for ring/free."""
    if variant == "mix":
        return ["corner", "diag"]
    for prefix, shape in (("seeds", "corner"), ("corner", "corner"), ("diag", "diag"), ("stair", "stair")):
        if variant.startswith(prefix) and variant[len(prefix):].isdigit():
            return [shape] * int(variant[len(prefix):])
    return None


def shape_metrics(crops, wl_ids):
    """corner_share and diag_joints (see module docstring)."""
    turns = cells = diag = 0
    for c in crops:
        wl = np.pad(np.isin(c, wl_ids), 1)
        core = wl[1:-1, 1:-1]
        n, s, w, e = wl[2:, 1:-1], wl[:-2, 1:-1], wl[1:-1, :-2], wl[1:-1, 2:]
        count = n.astype(int) + s + w + e
        turns += int((core & (count == 2) & ~(n & s) & ~(w & e)).sum())
        cells += int(core.sum())
        lone_diag = np.zeros_like(core)
        for (dz, dx) in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            d = wl[1 + dz:wl.shape[0] - 1 + dz, 1 + dx:wl.shape[1] - 1 + dx]
            a = wl[1 + dz:wl.shape[0] - 1 + dz, 1:-1]
            b = wl[1:-1, 1 + dx:wl.shape[1] - 1 + dx]
            lone_diag |= core & d & ~a & ~b
        diag += int(lone_diag.sum())
    return turns / max(1, cells), diag / max(1, len(crops))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--adjust", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--variants", default="ring,free,seeds4,seeds7")
    ap.add_argument("--tag", default="", help="suffix for the metrics file, so runs don't overwrite each other")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count())
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, C = state["cfg"], state["num_classes"]
    names = {int(k): v for k, v in state["names"].items()}
    chars = {int(k): v for k, v in state["chars"].items()}
    ids = {v: k for k, v in names.items()}
    net = model_lib.build(cfg, C)
    net.load_state_dict(state["ema"])
    net.eval()
    train = data.WindowSet(os.path.join(ROOT, cfg["set"]), "train", "cpu")
    val = data.WindowSet(os.path.join(ROOT, cfg["set"]), "val", "cpu")
    weights, _ = train.class_weights(cfg["weight_power"], cfg["weight_cap"])
    bias = -args.adjust * torch.log(weights.clamp(min=1e-6))
    W = val.cells.shape[1]
    lo = (W - SQUARE) // 2
    wl_ids = wall_like_ids(names)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "inpaint_test")
    os.makedirs(out_dir, exist_ok=True)

    def crop_metrics(crops, full=None, offsets=None):
        m = metrics.compute(crops, names)
        if full is not None:
            wl = np.isin(full, wl_ids)
            m["cut_cells"] = float(np.mean([cut_cells(w, oz, ox, SQUARE) for w, (oz, ox) in zip(wl, offsets)]))
        m["wall_like_cells"] = float(np.isin(crops, wl_ids).sum(axis=(1, 2)).mean())
        m["corner_share"], m["diag_joints"] = shape_metrics(crops, wl_ids)
        return m

    real = val.cells.numpy()[:, lo:lo + SQUARE, lo:lo + SQUARE]
    results = {"real_centre_crop": crop_metrics(real, val.cells.numpy(), [(lo, lo)] * len(real))}

    rng = np.random.default_rng(args.seed)
    for variant in args.variants.split(","):
        tokens = torch.full((args.n, W, W), C, dtype=torch.long)
        keep = torch.zeros((args.n, W, W), dtype=torch.bool)
        seeds_per = []
        if variant == "ring":
            keep[:] = True
            keep[:, lo:lo + SQUARE, lo:lo + SQUARE] = False
            tokens[keep] = ids["empty"]
        elif variant_shapes(variant):
            for i in range(args.n):
                s = scatter_seeds(rng, variant_shapes(variant), lo, lo + SQUARE)
                seeds_per.append(s)
                for cells in s:
                    for z, x in cells:
                        tokens[i, z, x] = ids["wall"]
                        keep[i, z, x] = True
        g = torch.Generator().manual_seed(args.seed)
        with torch.no_grad():
            full = diffusion.sample(net, args.n, (W, W), C, fixed=(tokens, keep), generator=g, device="cpu",
                                    logit_bias=bias).numpy()
        if variant == "free":
            offsets = [best_crop(w, wl_ids, SQUARE) for w in full]
        else:
            offsets = [(lo, lo)] * args.n
        crops = np.stack([w[oz:oz + SQUARE, ox:ox + SQUARE] for w, (oz, ox) in zip(full, offsets)])
        m = crop_metrics(crops, full, offsets)
        if seeds_per:
            # a seed counts as completed when a cell next to its corner lies in an enclosed region of the crop
            done = total = 0
            for crop, s in zip(crops, seeds_per):
                wl = np.isin(crop, wl_ids)
                lab, _ = ndimage.label(np.pad(~wl, 1, constant_values=True), metrics.FOUR)
                outside = lab[0, 0]
                lab = lab[1:-1, 1:-1]
                for cells in s:
                    total += 1
                    z0, x0 = cells[0][0] - lo, cells[0][1] - lo
                    near = lab[max(0, z0 - 1):z0 + 2, max(0, x0 - 1):x0 + 2]
                    done += bool(((near != outside) & (near != 0)).any())
            m["seed_in_room"] = done / max(1, total)
        results[variant] = m
        render.png_sheet(crops, names, os.path.join(out_dir, f"{variant}.png"), cols=8)
        with open(os.path.join(out_dir, f"{variant}.txt"), "w", encoding="utf-8") as f:
            f.write(render.ascii_sheet(crops, chars, per_row=3))
        print(f"{variant}: done", flush=True)

    json.dump(results, open(os.path.join(out_dir, f"metrics_{args.tag}.json" if args.tag else "metrics.json"), "w"),
              indent=1)
    keys = ["wall_like_cells", "freq_floor", "rooms_per_window", "room_area_median", "rooms_with_door",
            "stray_wall", "door_valid", "cut_cells", "seed_in_room", "corner_share", "diag_joints"]
    cols = list(results)
    print(f"{'metric':18s}" + "".join(f"{c:>18s}" for c in cols))
    for k in keys:
        print(f"{k:18s}" + "".join(f"{results[c].get(k, float('nan')):18.3f}" for c in cols))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
