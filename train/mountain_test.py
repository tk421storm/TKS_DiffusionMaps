"""Can the v1 model build against mountains and dig into them, without retraining? (docs/HISTORY.md, step A)

Synthetic mountains (smoothed noise plus a slope, so rock covers one side of the window with a ragged edge) are given to
the model in three ways:
  empty   rock cells fixed as empty (what the game does now)
  wall    rock cells fixed as wall: the mountain edge is a very thick wall to build against
  dig<k>  as wall, except the outer k cells of rock are left free; where the model draws floor, wall, door etc. the rock
          is mined out and built on, where it draws empty the rock stays
  shell   only the rock's outermost cells fixed as wall, its interior as empty: the mountain looks like one long
          building wall, which the training data has plenty of

Each with no seeds and with 2 mixed seeds (corner + diagonal, placed off the rock). Measured on the centred 36x36 square,
with remaining rock counted as wall (it closes rooms in game):
  rooms, rooms_with_door, wall_like_cells   as inpaint_test (rock excluded from wall_like_cells)
  rock_edge_built    share of open cells along the rock face that hold structure (wall-like or floor): hugging
  rooms_on_rock      rooms per square with rock as part of their outline
  dug_cells          rock cells mined out per square (dig variants)

Usage (CPU): python -u train/mountain_test.py runs/v1_mdm/ckpt_best.pt [--n 32]
Writes runs/<ckpt dir>/mountain_test/{metrics.json, <variant>.png, <variant>.txt}; rock is drawn dark blue-grey.
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
from inpaint_test import SQUARE, scatter_seeds, wall_like_ids  # noqa: E402
from train import ROOT  # noqa: E402

render.COLORS.setdefault("rock", (62, 64, 82))
SEED_MARGIN = 6


def mountain(rng, w):
    """Boolean rock mask: smoothed noise plus a slope from a random side, covering 20-45% of the window."""
    noise = ndimage.gaussian_filter(rng.normal(size=(w, w)), 3.5)
    noise /= noise.std()
    ramp = np.linspace(-1.0, 1.0, w)
    side = rng.integers(4)
    ramp = np.tile(ramp, (w, 1)) if side < 2 else np.tile(ramp[:, None], (1, w))
    if side % 2:
        ramp = -ramp
    field = 0.8 * noise + 2.2 * ramp
    target = rng.uniform(0.2, 0.45)
    rock = field > np.quantile(field, 1 - target)
    # drop specks: keep the large connected parts, fill small holes
    lab, k = ndimage.label(rock)
    sizes = ndimage.sum(rock, lab, range(1, k + 1))
    rock = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 30])
    return ndimage.binary_fill_holes(rock)


def seeds_off_rock(rng, shapes, rock, lo, hi):
    """Seeds that don't touch rock (with a one-cell gap)."""
    near = ndimage.binary_dilation(rock, iterations=1)
    for _ in range(50):
        s = scatter_seeds(rng, shapes, lo, hi)
        if s and all(not near[z, x] for cells in s for z, x in cells):
            return s
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--adjust", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--modes", default="empty,wall,dig2")
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
    weights, _ = train.class_weights(cfg["weight_power"], cfg["weight_cap"])
    bias = -args.adjust * torch.log(weights.clamp(min=1e-6))
    bias[ids["bridge"]] = -30.0   # as the game: no bridges on dry ground
    W = train.cells.shape[1]
    lo = (W - SQUARE) // 2
    wl_ids = wall_like_ids(names)
    ROCK = C                      # display / bookkeeping id, not a model token
    names_r, chars_r = dict(names), dict(chars)
    names_r[ROCK], chars_r[ROCK] = "rock", "R"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "mountain_test")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rocks = np.stack([mountain(rng, W) for _ in range(args.n)])
    seeds = [seeds_off_rock(rng, ["corner", "diag"], r, lo + SEED_MARGIN, lo + SQUARE - SEED_MARGIN) for r in rocks]

    results = {}
    for use_seeds in (False, True):
        for mode in args.modes.split(","):
            name = f"{mode}_{'mix2' if use_seeds else 'noseed'}"
            dig = int(mode[3:]) if mode.startswith("dig") else 0
            tokens = torch.full((args.n, W, W), C, dtype=torch.long)
            keep = torch.zeros((args.n, W, W), dtype=torch.bool)
            fixed_rock = np.zeros_like(rocks)
            for i, rock in enumerate(rocks):
                fixed_rock[i] = ndimage.binary_erosion(rock, iterations=dig, border_value=1) if dig else rock
                fr = torch.from_numpy(fixed_rock[i])
                if mode == "shell":
                    face = torch.from_numpy(rock & ~ndimage.binary_erosion(rock, border_value=1))
                    tokens[i][fr] = ids["empty"]
                    tokens[i][face] = ids["wall"]
                else:
                    tokens[i][fr] = ids["empty"] if mode == "empty" else ids["wall"]
                keep[i] |= fr
                if use_seeds:
                    for cells in seeds[i]:
                        for z, x in cells:
                            tokens[i, z, x] = ids["wall"]
                            keep[i, z, x] = True
            g = torch.Generator().manual_seed(args.seed)
            with torch.no_grad():
                out = diffusion.sample(net, args.n, (W, W), C, fixed=(tokens, keep), generator=g, device="cpu",
                                       logit_bias=bias).numpy()

            # what the game would end up with: rock where it was fixed, and where a free rock cell was left empty
            final_rock = fixed_rock | (rocks & (out == ids["empty"]))
            dug = (rocks & ~final_rock).sum(axis=(1, 2))
            shown = np.where(final_rock, ROCK, out)
            as_wall = np.where(final_rock, ids["wall"], out)
            sl = (slice(None), slice(lo, lo + SQUARE), slice(lo, lo + SQUARE))
            crops, crops_shown, crop_rock = as_wall[sl], shown[sl], final_rock[sl]

            m = metrics.compute(crops, names)
            built = np.isin(crops, wl_ids) & ~crop_rock
            m["wall_like_cells"] = float(built.sum(axis=(1, 2)).mean())
            # open cells along the rock face, and how many of them the layout uses
            face = ndimage.binary_dilation(crop_rock, metrics.FOUR[None]) & ~crop_rock
            structure = (np.isin(crops, wl_ids) | (crops == ids["floor"])) & ~crop_rock
            m["rock_edge_built"] = float((face & structure).sum() / max(1, face.sum()))
            rooms_on_rock = 0
            for c, r in zip(crops, crop_rock):
                lab, _ = ndimage.label(np.pad(~np.isin(c, wl_ids), 1, constant_values=True), metrics.FOUR)
                outside = lab[0, 0]
                lab = lab[1:-1, 1:-1]
                touching = np.unique(lab[ndimage.binary_dilation(r, metrics.FOUR) & ~r])
                rooms_on_rock += sum(1 for t in touching if t not in (0, outside))
            m["rooms_on_rock"] = rooms_on_rock / args.n
            m["dug_cells"] = float(dug.mean())
            m["rock_share"] = float(crop_rock.mean())
            results[name] = m
            render.png_sheet(crops_shown, names_r, os.path.join(out_dir, f"{name}.png"), cols=8)
            with open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(render.ascii_sheet(crops_shown, chars_r, per_row=3))
            print(f"{name}: done", flush=True)

    json.dump(results, open(os.path.join(out_dir, f"metrics_{args.modes.replace(',', '_')}.json"), "w"), indent=1)
    keys = ["rock_share", "wall_like_cells", "freq_floor", "rooms_per_window", "rooms_with_door", "rock_edge_built",
            "rooms_on_rock", "dug_cells", "stray_wall", "door_valid"]
    cols = list(results)
    print(f"{'metric':16s}" + "".join(f"{c:>14s}" for c in cols))
    for k in keys:
        print(f"{k:16s}" + "".join(f"{results[c].get(k, float('nan')):14.3f}" for c in cols))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
