"""Stage 2: cut training windows from the stage-1 extracts, driven by a config file.

Usage: python -I windows.py <config.json> [--extract DIR] [--out DIR] [--workers N]
  defaults: --extract ../data/extract  --out ../data/sets/<config name>

Per snapshot:
  1. Map layers to categories (categories.py).
  2. Pad the snapshot by one window of empty ground (the home-area box is not the edge of the
     map). If the snapshot was truncated, everything above the cut row is invalid.
  3. Candidate windows on a stride grid; drop any with invalid or mountain cells, or with fewer
     than min_structure_cells structure cells.
  4. A structure component (8-connected walls/doors/building interiors) is "bad" if the window cuts
     it or it contains a building whose class is not in building_classes.
       edge_policy "reject": drop windows with any bad component.
       edge_policy "mask":   keep them, clear loss_mask on bad components (grown by mask_grow)
                             and require min_structure_cells unmasked structure cells and at most
                             max_masked_fraction of the window masked.
     Every window must also hold min_complete_buildings whole enclosed buildings of an allowed class.
  5. Greedy de-duplication by most unmasked structure, skipping overlap above max_overlap.
Train/val split is by colony (world seed + tile + gameId), never by window.

Output <out>/: train.npz and val.npz (cells uint8 [N,W,W], loss_mask bool, cond_mask bool reserved
for obstacles (all true for now), source int32 [N,3] = snapshot index, z, x of the window's
lower-left corner in snapshot cells, may be negative), manifest.json.
Rows are south (min z) first; rotations and flips are left to training time.
"""
import argparse
import collections
import hashlib
import glob
import json
import multiprocessing
import os
import sys
import zlib

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import categories  # noqa: E402
import extract  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {
    "window": 48, "stride": 8, "edge_policy": "mask", "mask_grow": 1, "min_structure_cells": 20,
    "min_complete_buildings": 1, "max_masked_fraction": 1.0,
    "max_overlap": 0.5, "building_classes": ["free_standing", "unroofed_enclosure"],
    "terrain_overrides": {}, "edifice_overrides": {}, "split": {"by": "colony", "val": 0.1}, "seed": 1,
}
K8 = np.ones((3, 3), bool)

_cfg = _cats = None   # set per worker by _init


def _init(cfg, vocab):
    global _cfg, _cats
    _cfg = cfg
    _cats = categories.Categories(cfg, vocab)


def window_sums(a, W, stride):
    """Sum of `a` over every WxW window at stride offsets (integral image)."""
    s = np.pad(a.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    zs = np.arange(0, a.shape[0] - W + 1, stride)
    xs = np.arange(0, a.shape[1] - W + 1, stride)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    return Z, X, s[Z + W, X + W] - s[Z, X + W] - s[Z + W, X] + s[Z, X]


def windows_for(path):
    cfg, cats = _cfg, _cats
    W, P = cfg["window"], cfg["window"]
    e = np.load(path)
    stats = collections.Counter()
    cat = cats.grid(e)
    struct = np.isin(cat, cats.structure_ids())
    building = e["building"]
    allowed = {extract.CLASSES.index(c) for c in cfg["building_classes"]}
    bclass = e["building_class"]
    disallowed_building = np.isin(building, [i for i, c in enumerate(bclass) if i and c not in allowed])

    # Structure components and which are disallowed; bboxes for the "cut" test.
    comp, n = ndimage.label(struct | (building > 0), K8)
    comp_bad_class = np.zeros(n + 1, bool)
    comp_bad_class[np.unique(comp[disallowed_building])] = True
    comp_bad_class[0] = False
    boxes = ndimage.find_objects(comp)
    # Structure component of each building (any of its cells; a building lies within one component).
    b_comp = np.zeros(len(bclass), np.int32)
    for b in range(1, len(bclass)):
        cells = comp[building == b]
        b_comp[b] = cells[0] if len(cells) else 0

    pad = lambda a, v: np.pad(a, P, constant_values=v)
    catp, structp, compp, buildp = pad(cat, 0), pad(struct, False), pad(comp, 0), pad(building, 0)
    validp = pad(e["valid"], True)
    H = int(e["valid"].any(1).nonzero()[0].max()) + 1 if e["valid"].any() else 0
    if H < e["valid"].shape[0]:            # truncated: nothing above the cut is known
        validp[P + H:, :] = False
    mountp = pad(e["mountain"], False)

    Z, X, invalid = window_sums(~validp, W, cfg["stride"])
    _, _, mountain = window_sums(mountp, W, cfg["stride"])
    _, _, structure = window_sums(structp, W, cfg["stride"])
    stats["candidates"] += Z.size
    ok = invalid == 0
    stats["rejected_invalid"] += int((~ok).sum())
    ok2 = ok & (mountain == 0)
    stats["rejected_mountain"] += int((ok & ~ok2).sum())
    ok3 = ok2 & (structure >= cfg["min_structure_cells"])
    stats["rejected_little_structure"] += int((ok2 & ~ok3).sum())

    picks = []
    for z, x in zip(Z[ok3], X[ok3]):
        cw = compp[z:z + W, x:x + W]
        ids = np.unique(cw)
        ids = ids[ids > 0]
        bad = []
        for i in ids:
            bz, bx = boxes[i - 1]
            # Component box in padded coords; cut if any part lies outside the window.
            if (bz.start + P < z or bz.stop + P > z + W or bx.start + P < x or bx.stop + P > x + W
                    or comp_bad_class[i]):
                bad.append(i)
        if bad and cfg["edge_policy"] == "reject":
            stats["rejected_cut_or_class"] += 1
            continue
        loss = np.ones((W, W), bool)
        if bad:
            m = np.isin(cw, bad)
            if cfg["mask_grow"]:
                m = ndimage.binary_dilation(m, K8, iterations=cfg["mask_grow"])
            loss = ~m
        usable = int((structp[z:z + W, x:x + W] & loss).sum())
        if usable < cfg["min_structure_cells"] or 1 - loss.mean() > cfg["max_masked_fraction"]:
            stats["rejected_masked_too_much"] += 1
            continue
        # Complete buildings: enclosed buildings (segment.classify) of an allowed class whose
        # structure component lies entirely inside the window.
        good_comps = set(ids.tolist()) - set(bad)
        complete = sum(1 for b in np.unique(buildp[z:z + W, x:x + W]) if b and b_comp[b] in good_comps)
        if complete < cfg["min_complete_buildings"]:
            stats["rejected_no_complete_building"] += 1
            continue
        picks.append((usable, int(z), int(x), loss))

    # Greedy de-duplication, best (most usable structure) first.
    picks.sort(key=lambda p: -p[0])
    chosen = []
    for usable, z, x, loss in picks:
        if any((W - abs(z - cz)) * (W - abs(x - cx)) > cfg["max_overlap"] * W * W
               for _, cz, cx, _ in chosen if abs(z - cz) < W and abs(x - cx) < W):
            stats["rejected_overlap"] += 1
            continue
        chosen.append((usable, z, x, loss))
    stats["windows"] += len(chosen)
    stats["windows_with_mask"] += sum(1 for c in chosen if not c[3].all())
    out = [(catp[z:z + W, x:x + W].copy(), loss, z - P, x - P) for _, z, x, loss in chosen]
    return os.path.splitext(os.path.basename(path))[0], out, stats


def code_version():
    h = hashlib.sha1()
    for f in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        h.update(open(f, "rb").read())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--extract", default=os.path.join(HERE, "..", "data", "extract"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(json.load(open(args.config, encoding="utf-8")))
    out = args.out or os.path.join(HERE, "..", "data", "sets", cfg["name"])
    os.makedirs(out, exist_ok=True)
    vocab = json.load(open(os.path.join(args.extract, "vocab.json"), encoding="utf-8"))
    index = json.load(open(os.path.join(args.extract, "index.json"), encoding="utf-8"))
    sids = sorted(s for s, m in index.items() if "error" not in m)
    paths = [os.path.join(args.extract, s + ".npz") for s in sids]
    sidx = {s: i for i, s in enumerate(sids)}

    val_frac = cfg["split"]["val"]
    def is_val(sid):
        m = index[sid]
        key = f"{m.get('seed')}|{m.get('tile')}|{m.get('game_id')}|{cfg['seed']}"
        return zlib.crc32(key.encode()) % 10000 < val_frac * 10000

    sets = {"train": collections.defaultdict(list), "val": collections.defaultdict(list)}
    stats = collections.Counter()
    biomes = {"train": collections.Counter(), "val": collections.Counter()}
    with multiprocessing.Pool(args.workers, _init, (cfg, vocab)) as pool:
        for sid, wins, st in pool.imap_unordered(windows_for, paths, chunksize=8):
            stats.update(st)
            stats["snapshots_with_windows"] += bool(wins)
            split = "val" if is_val(sid) else "train"
            for cells, loss, z, x in wins:
                s = sets[split]
                s["cells"].append(cells)
                s["loss_mask"].append(loss)
                s["source"].append((sidx[sid], z, x))
            biomes[split][index[sid].get("biome")] += len(wins)

    counts = {}
    W = cfg["window"]
    for split, s in sets.items():
        n = len(s["cells"])
        cells = np.array(s["cells"], np.uint8).reshape(n, W, W)
        loss = np.array(s["loss_mask"], bool).reshape(n, W, W)
        np.savez_compressed(os.path.join(out, split + ".npz"), cells=cells, loss_mask=loss,
                            cond_mask=np.ones_like(loss), source=np.array(s["source"], np.int32).reshape(n, 3))
        hist = np.bincount(cells[loss].ravel(), minlength=max(c["id"] for c in cfg["categories"]) + 1)
        counts[split] = {"windows": n, "category_cells_in_loss": hist.tolist(),
                         "masked_fraction": float(1 - loss.mean()) if n else 0.0,
                         "biomes": dict(biomes[split].most_common())}
    manifest = {"config": cfg, "code_version": code_version(), "extract_version": extract.EXTRACT_VERSION,
                "snapshots": sids, "filter_counts": dict(stats), "splits": counts}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({"filter_counts": dict(stats),
                      "splits": {k: {kk: vv for kk, vv in v.items() if kk != "biomes"} for k, v in counts.items()}},
                     indent=1))


if __name__ == "__main__":
    main()
