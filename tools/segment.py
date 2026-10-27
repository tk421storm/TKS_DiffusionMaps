"""Split RealRuins snapshots into buildings and classify each one.

Classes:
  free_standing       enclosed by constructed walls, mostly roofed, no mountain next to it
  mountain_adjacent   has smoothed rock walls, or touches a mountain blob (see below)
  unroofed_enclosure  walled but mostly unroofed: pens, yards, killbox courts
  clipped_at_edge     touches the snapshot edge or the truncation row, so it may be incomplete

Mountain blobs: roofed cells that are not enclosed by walls. Natural rock is not stored but its thick
roof is, so unmined mountain shows up as roof with nothing else. Blobs smaller than MIN_MOUNTAIN
cells, or mostly lined by walls (a constructed roof leaking through a wall gap), are not mountain.

Usage:
  python -I segment.py <cache dir> [--limit N] [--show N] [--seed S]
    --show N   ASCII-render up to N small sample buildings per class
               (# wall, S smoothed rock, + door, o roof inside walls, r roof outside walls, . floor)
"""
import argparse
import collections
import os
import random
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_snapshot  # noqa: E402

MIN_MOUNTAIN = 40          # roofed cells; smaller unenclosed roofs count as overhangs
MIN_INTERIOR = 4           # ignore enclosures smaller than this
MOUNTAIN_RING = 2          # cells around a building checked for mountain contact
SMOOTH_WALL_FRACTION = 0.1
ROOFED_FRACTION = 0.5
K8 = np.ones((3, 3), bool)


def mountain_labels(g, outside):
    """Labels unenclosed roof blobs; returns (labels, sizes) with non-mountain blobs zeroed in sizes."""
    leak = g.roof & outside
    labels, _ = ndimage.label(leak)
    sizes = np.bincount(labels.ravel())
    near_wall = ndimage.binary_dilation(g.barrier, K8)
    edge = leak & ~ndimage.binary_erosion(leak, K8, border_value=1)
    for b in range(1, len(sizes)):
        if sizes[b] < MIN_MOUNTAIN:
            sizes[b] = 0
            continue
        e = edge & (labels == b)
        if (e & near_wall).sum() > 0.5 * max(1, e.sum()):
            sizes[b] = 0
    return labels, sizes, leak


def classify(g):
    """Yields dicts {cls, mask, area, size} for each building in a snapshot, plus a summary dict last."""
    barrier = g.barrier
    # Outside = open cells connected to the snapshot border (padding makes the border one region).
    open_ = np.pad(~barrier, 1, constant_values=True)
    lab, _ = ndimage.label(open_)
    outside = (lab == lab[0, 0])[1:-1, 1:-1]
    interior = ~barrier & ~outside
    mlab, msizes, leak = mountain_labels(g, outside)
    # A building is an enclosed interior plus the walls around it (8-connected, so rooms sharing walls join).
    body = interior | (barrier & ndimage.binary_dilation(interior, K8))
    blab, n = ndimage.label(body, K8)
    top = g.usable_height - 1
    for i in range(1, n + 1):
        m = blab == i
        inner = m & interior
        area = int(inner.sum())
        if area < MIN_INTERIOR:
            continue
        ys, xs = np.nonzero(m)
        walls = m & barrier
        smooth_frac = (g.smooth & walls).sum() / max(1, walls.sum())
        roof_frac = (g.roof & inner).sum() / area
        ring = ndimage.binary_dilation(m, K8, iterations=MOUNTAIN_RING) & ~m
        touches_mountain = any(msizes[t] for t in np.unique(mlab[ring]) if t)
        on_edge = xs.min() == 0 or ys.min() == 0 or xs.max() == g.width - 1 or ys.max() >= top
        if smooth_frac > SMOOTH_WALL_FRACTION or touches_mountain:
            cls = "mountain_adjacent"
        elif roof_frac < ROOFED_FRACTION:
            cls = "unroofed_enclosure"
        elif on_edge:
            cls = "clipped_at_edge"
        else:
            cls = "free_standing"
        yield {"cls": cls, "mask": m, "area": area, "size": int(max(np.ptp(xs), np.ptp(ys)) + 1)}
    yield {"summary": True, "mountain": bool(msizes[1:].any()), "leak": leak}


def render(g, leak, mask, pad=4):
    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, ys.min() - pad), ys.max() + pad + 1
    x0, x1 = max(0, xs.min() - pad), xs.max() + pad + 1
    ch = np.full(g.wall.shape, " ")
    ch[g.floor & ~g.roof] = "."
    ch[g.roof] = "o"
    ch[leak] = "r"
    ch[g.wall] = "#"
    ch[g.door] = "+"
    ch[g.smooth] = "S"
    # Row 0 is minimum z; print north at the top.
    return "\n".join("".join(row).rstrip() for row in ch[y0:y1, x0:x1][::-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.cache_dir) if os.path.isfile(os.path.join(args.cache_dir, f)))
    rnd = random.Random(args.seed)
    if args.limit:
        rnd.shuffle(files)
        files = files[:args.limit]
    counts = collections.Counter()
    fs_sizes, fs_areas = [], []
    samples = collections.defaultdict(list)
    for f in files:
        path = os.path.join(args.cache_dir, f)
        try:
            g = rr_snapshot.open_grids(path)
        except Exception as e:  # unreadable file: count and move on
            counts["unreadable"] += 1
            continue
        counts["snapshots"] += 1
        counts["truncated"] += g.truncated
        counts["with_smoothed_walls"] += bool(g.smooth.any())
        counts["with_natural_terrain"] += bool(g.natural_terrain_cells)
        found = list(classify(g))
        summary = found.pop()
        counts["with_mountain"] += summary["mountain"]
        for b in found:
            counts[b["cls"]] += 1
            if b["cls"] == "free_standing":
                fs_sizes.append(b["size"]); fs_areas.append(b["area"])
            if args.show and 6 <= b["size"] <= 40:
                samples[b["cls"]].append((f, g, summary["leak"], b["mask"]))

    print("snapshots / buildings")
    for k, v in sorted(counts.items()):
        print(f"  {k:22s} {v}")
    if fs_sizes:
        fs = np.array(fs_sizes)
        print("free-standing longest side p10/p50/p90:", *np.percentile(fs, [10, 50, 90]).astype(int),
              " max", fs.max(), " median interior", int(np.median(fs_areas)), "cells,",
              int((fs <= 38).sum()), "fit a vanilla 38x38 settlement rect")
    for cls, items in samples.items():
        for f, g, leak, mask in rnd.sample(items, min(args.show, len(items))):
            print(f"\n===== {cls}  {f}")
            print(render(g, leak, mask))


if __name__ == "__main__":
    main()
