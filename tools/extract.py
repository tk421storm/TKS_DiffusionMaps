"""Stage 1: extract every RealRuins snapshot into a per-snapshot .npz of full-detail layers.

Independent of window size and categories (those are stage 2). Incremental: snapshots already
extracted with the current EXTRACT_VERSION are skipped. Def names go into a shared vocabulary whose
ids never change, so earlier extracts stay valid when new files add new defs.

Usage: python -I extract.py [--raw DIR] [--out DIR] [--workers N] [--force]
  defaults: --raw ../data/raw  --out ../data/extract  (relative to this file)

Output per snapshot <out>/<id>.npz:
  terrain_id    uint16 (H, W)  terrain vocab id, 0 = nothing recorded
  edifice_id    uint16 (H, W)  def on the cell flagged isDoor/actsAsWall, 0 = none
  edifice_flag  uint8  (H, W)  bit 1 = actsAsWall, bit 2 = isDoor
  roof          bool   (H, W)  any roof (natural rock roof included)
  building      int16  (H, W)  building index from segment.classify, 0 = none
  building_class uint8 (B+1,)  class code per building index (CLASSES), [0] unused
  mountain      bool   (H, W)  mountain blobs + smoothed walls, grown by MOUNTAIN_GROW cells
  valid         bool   (H, W)  inside the snapshot and below the truncation row
  things        structured (N,) z, x, def, stuff, rot, stack for all other top-level things
Shared: <out>/vocab.json, <out>/index.json
Pawns and corpses are skipped entirely (names, health and stories are not stored).
"""
import argparse
import json
import multiprocessing
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_snapshot  # noqa: E402
import segment  # noqa: E402

EXTRACT_VERSION = 1
CLASSES = ["none", "free_standing", "mountain_adjacent", "unroofed_enclosure", "clipped_at_edge"]
MOUNTAIN_GROW = 2
SKIP_DEFS = ("Pawn", "corpse")
VOCAB_KINDS = ("terrain", "edifice", "thing", "stuff")
THING_DTYPE = np.dtype([("z", "i2"), ("x", "i2"), ("def", "u2"), ("stuff", "u2"), ("rot", "u1"), ("stack", "i4")])

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW = os.path.join(HERE, "..", "data", "raw")
DEFAULT_OUT = os.path.join(HERE, "..", "data", "extract")


class LocalVocab:
    """Per-worker name -> local id; remapped to global ids in the main process."""

    def __init__(self):
        self.names = {k: [""] for k in VOCAB_KINDS}
        self.ids = {k: {} for k in VOCAB_KINDS}
        self.flags = {}   # edifice name -> [wall count, door count]

    def id(self, kind, name):
        if not name:
            return 0
        d = self.ids[kind]
        if name not in d:
            d[name] = len(self.names[kind])
            self.names[kind].append(name)
        return d[name]


def extract_one(path):
    """Worker: returns (snapshot id, arrays with local ids, local vocab, metadata) or an error string."""
    sid = os.path.splitext(os.path.basename(path))[0]
    try:
        root, truncated = rr_snapshot.load(path)
        g = rr_snapshot.Grids(root, truncated)
    except Exception as e:
        return sid, None, None, f"{type(e).__name__}: {e}"
    v = LocalVocab()
    shape = g.wall.shape
    terrain = np.zeros(shape, np.uint16)
    edifice = np.zeros(shape, np.uint16)
    eflag = np.zeros(shape, np.uint8)
    things = []
    for c in root.iter("cell"):
        x, z = int(c.get("x")), int(c.get("z"))
        if x >= g.width or z >= g.height:
            continue
        t = c.find("terrain")
        if t is not None:
            terrain[z, x] = v.id("terrain", t.get("def", ""))
        for it in c.findall("item"):   # top-level only; contents of containers are ignored
            d = it.get("def", "")
            if d in SKIP_DEFS:
                continue
            is_door, is_wall = it.get("isDoor") == "1", it.get("actsAsWall") == "1"
            if is_door or is_wall:
                f = v.flags.setdefault(d, [0, 0])
                f[0] += is_wall
                f[1] += is_door
                # A door wins over a wall-like thing on the same cell.
                if is_door or not eflag[z, x] & 2:
                    edifice[z, x] = v.id("edifice", d)
                    eflag[z, x] = (1 if is_wall else 0) | (2 if is_door else 0)
                continue
            things.append((z, x, v.id("thing", d), v.id("stuff", it.get("stuffDef", "")),
                           int(it.get("rot", 0) or 0), int(it.get("stackCount", 1) or 1)))

    found = list(segment.classify(g))
    summary = found.pop()
    building = np.zeros(shape, np.int16)
    bclass = [0]
    for i, b in enumerate(found, start=1):
        building[b["mask"]] = i
        bclass.append(CLASSES.index(b["cls"]))
    # Mountain = blob cells kept by segment.mountain_labels, plus smoothed rock walls.
    open_ = np.pad(~g.barrier, 1, constant_values=True)
    lab, _ = ndimage.label(open_)
    outside = (lab == lab[0, 0])[1:-1, 1:-1]
    mlab, msizes, _ = segment.mountain_labels(g, outside)
    mountain = np.isin(mlab, np.nonzero(msizes)[0]) & (mlab > 0) | g.smooth
    if mountain.any():
        mountain = ndimage.binary_dilation(mountain, segment.K8, iterations=MOUNTAIN_GROW)
    valid = np.zeros(shape, bool)
    valid[:g.usable_height] = True

    world = root.find("world")
    meta = {
        "id": sid, "file": os.path.basename(path), "version": root.get("version"),
        "biome": root.get("biomeDef"), "map_size": int(root.get("mapSize", 0) or 0),
        "width": g.width, "height": g.height, "usable_height": int(g.usable_height),
        "truncated": bool(truncated),
        "seed": world.get("seed") if world is not None else None,
        "tile": world.get("tile") if world is not None else None,
        "game_id": world.get("gameId") if world is not None else None,
        "buildings": {c: bclass.count(i) for i, c in enumerate(CLASSES) if i},
        "extract_version": EXTRACT_VERSION,
    }
    arrays = {
        "terrain_id": terrain, "edifice_id": edifice, "edifice_flag": eflag, "roof": g.roof,
        "building": building, "building_class": np.array(bclass, np.uint8),
        "mountain": mountain, "valid": valid,
        "things": np.array(things, THING_DTYPE),
    }
    return sid, arrays, v, meta


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--force", action="store_true", help="re-extract everything")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    vocab_path, index_path = os.path.join(args.out, "vocab.json"), os.path.join(args.out, "index.json")
    # vocab: kind -> {"names": [...], "counts": [...]} (+ "flags" for edifices); id = list position.
    vocab = load_json(vocab_path, {k: {"names": [""], "counts": [0]} for k in VOCAB_KINDS})
    vocab["edifice"].setdefault("flags", [[0, 0]] * len(vocab["edifice"]["names"]))
    index = load_json(index_path, {})
    lookup = {k: {n: i for i, n in enumerate(vocab[k]["names"])} for k in VOCAB_KINDS}

    files = sorted(f for f in os.listdir(args.raw) if f.endswith(".bp"))
    todo = [os.path.join(args.raw, f) for f in files
            if args.force or index.get(os.path.splitext(f)[0], {}).get("extract_version") != EXTRACT_VERSION]
    print(f"{len(files)} raw files, {len(todo)} to extract, {args.workers} workers")

    def global_id(kind, name):
        if name not in lookup[kind]:
            lookup[kind][name] = len(vocab[kind]["names"])
            vocab[kind]["names"].append(name)
            vocab[kind]["counts"].append(0)
            if kind == "edifice":
                vocab[kind]["flags"].append([0, 0])
        return lookup[kind][name]

    errors = 0
    with multiprocessing.Pool(args.workers) as pool:
        for n, (sid, arrays, lv, meta) in enumerate(pool.imap_unordered(extract_one, todo, chunksize=8), 1):
            if arrays is None:
                errors += 1
                index[sid] = {"id": sid, "error": meta, "extract_version": EXTRACT_VERSION}
                continue
            # Remap local ids to global vocab ids and update counts.
            maps = {}
            for kind in VOCAB_KINDS:
                maps[kind] = np.array([0] + [global_id(kind, nm) for nm in lv.names[kind][1:]], np.uint16)
            for kind, key in (("terrain", "terrain_id"), ("edifice", "edifice_id")):
                arrays[key] = maps[kind][arrays[key]]
                ids, cnt = np.unique(arrays[key][arrays[key] > 0], return_counts=True)
                for i, c in zip(ids, cnt):
                    vocab[kind]["counts"][i] += int(c)
            th = arrays["things"]
            if len(th):
                th["def"] = maps["thing"][th["def"]]
                th["stuff"] = maps["stuff"][th["stuff"]]
                for kind, col in (("thing", "def"), ("stuff", "stuff")):
                    ids, cnt = np.unique(th[col][th[col] > 0], return_counts=True)
                    for i, c in zip(ids, cnt):
                        vocab[kind]["counts"][i] += int(c)
            for name, (w, d) in lv.flags.items():
                f = vocab["edifice"]["flags"][lookup["edifice"][name]]
                f[0] += w
                f[1] += d
            np.savez_compressed(os.path.join(args.out, sid + ".npz"), **arrays)
            index[sid] = meta
            if n % 250 == 0:
                print(f"  {n}/{len(todo)}")
                save_json(vocab_path, vocab)
                save_json(index_path, index)
    save_json(vocab_path, vocab)
    save_json(index_path, index)
    ok = sum(1 for m in index.values() if "error" not in m)
    print(f"done: {ok} extracted in total, {errors} errors this run; vocab sizes:",
          {k: len(vocab[k]["names"]) - 1 for k in VOCAB_KINDS})


if __name__ == "__main__":
    main()
