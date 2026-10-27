"""Stage 3: inspect a window set, or the category mapping of the extract vocabulary.

Usage:
  python -I report.py <set dir> [--show N] [--split train|val] [--seed S]
      counts, filter breakdown, category frequencies, biomes, N ASCII sample windows
      (category chars from the config; masked cells shown as "~" when they hold structure)
  python -I report.py --vocab <config.json> [--extract DIR] [--top N]
      which terrain/edifice defs map to which category, by cell count; use this to fill
      terrain_overrides / edifice_overrides
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import categories  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def show_set(args):
    manifest = json.load(open(os.path.join(args.target, "manifest.json"), encoding="utf-8"))
    cfg = manifest["config"]
    names = {c["id"]: c["name"] for c in cfg["categories"]}
    chars = {c["id"]: c.get("char", "?") for c in cfg["categories"]}
    print(f"set {cfg['name']}  window {cfg['window']}  edge_policy {cfg['edge_policy']}  "
          f"code {manifest['code_version']}  extract v{manifest['extract_version']}")
    print("filter counts:")
    for k, v in manifest["filter_counts"].items():
        print(f"  {k:28s} {v}")
    for split, s in manifest["splits"].items():
        hist = np.array(s["category_cells_in_loss"], float)
        frac = ", ".join(f"{names.get(i, i)} {100 * h / max(1, hist.sum()):.1f}%" for i, h in enumerate(hist))
        print(f"{split}: {s['windows']} windows, {100 * s['masked_fraction']:.1f}% of cells masked; {frac}")
        print("   biomes:", ", ".join(f"{b} {n}" for b, n in list(s["biomes"].items())[:8]))
    if not args.show:
        return
    d = np.load(os.path.join(args.target, args.split + ".npz"))
    rnd = random.Random(args.seed)
    for i in rnd.sample(range(len(d["cells"])), min(args.show, len(d["cells"]))):
        cells, loss = d["cells"][i], d["loss_mask"][i]
        sid = manifest["snapshots"][d["source"][i][0]]
        print(f"\n===== #{i}  {sid}  z={d['source'][i][1]} x={d['source'][i][2]}  "
              f"masked {100 * (1 - loss.mean()):.0f}%")
        rows = []
        for r in range(cells.shape[0] - 1, -1, -1):   # north at the top
            rows.append("".join(chars[c] if loss[r, j] or c == 0 else "~" for j, c in enumerate(cells[r])).rstrip())
        print("\n".join(rows))


def show_vocab(args):
    cfg = json.load(open(args.target, encoding="utf-8"))
    vocab = json.load(open(os.path.join(args.extract, "vocab.json"), encoding="utf-8"))
    cats = categories.Categories(cfg, vocab)
    names = {c["id"]: c["name"] for c in cfg["categories"]}
    for kind, lut in (("terrain", cats.terrain_lut), ("edifice", cats.edifice_lut)):
        v = vocab[kind]
        order = np.argsort(v["counts"])[::-1]
        print(f"\n{kind}: {len(v['names']) - 1} defs; top {args.top} by cell count")
        by_cat = {}
        for i in order:
            if not v["names"][i]:
                continue
            by_cat.setdefault(int(lut[i]), []).append(i)
        for cid, ids in sorted(by_cat.items()):
            total = sum(v["counts"][i] for i in ids)
            label = names.get(cid, cid) if kind == "terrain" or cid else "(no rule: terrain decides)"
            print(f"  -> {label}: {len(ids)} defs, {total} cells")
            for i in ids[:args.top]:
                extra = f"  wall/door flags {v['flags'][i]}" if kind == "edifice" else ""
                print(f"       {v['counts'][i]:9d}  {v['names'][i]}{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="set directory, or a config file with --vocab")
    ap.add_argument("--vocab", action="store_true")
    ap.add_argument("--extract", default=os.path.join(HERE, "..", "data", "extract"))
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    (show_vocab if args.vocab else show_set)(args)


if __name__ == "__main__":
    main()
