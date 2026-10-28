"""Structural checks on category grids (numpy), for comparing generated windows with real ones.

"Wall-like" = wall, fence, wall_mounted, door (anything that blocks or sits in a wall line).
Per set of windows:
  freq_<name>          category frequencies
  stray_wall           wall/fence/mounted cells with no wall-like 4-neighbour (lone posts)
  door_valid           doors with wall-like cells on two opposite sides and open cells on the other two
  mounted_valid        same rule for wall-mounted objects (vents, coolers, windows)
  rooms_per_window     enclosed open regions (not reachable from the window edge)
  room_area_median     median cells per enclosed region
  rooms_with_door      enclosed regions touching at least one door
  floor_enclosed       share of floor cells inside enclosed regions
"""
import numpy as np
from scipy import ndimage

FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def compute(cells, names):
    ids = {v: k for k, v in names.items()}
    wall, door = ids.get("wall", -1), ids.get("door", -1)
    fence, mounted, floor = ids.get("fence", -1), ids.get("wall_mounted", -1), ids.get("floor", -1)
    n = len(cells)
    out = {}
    counts = np.bincount(cells.ravel(), minlength=max(names) + 1)
    for i, nm in names.items():
        out["freq_" + nm] = counts[i] / cells.size

    stray = blocking = 0
    door_ok = door_n = mount_ok = mount_n = 0
    rooms = 0
    areas = []
    rooms_door = 0
    floor_in = floor_all = 0
    for c in cells:
        wl = np.isin(c, [wall, door, fence, mounted])
        p = np.pad(wl, 1)
        nbr = p[:-2, 1:-1].astype(int) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        blocks = np.isin(c, [wall, fence, mounted])
        stray += int((blocks & (nbr == 0)).sum())
        blocking += int(blocks.sum())
        ns, ew = p[:-2, 1:-1] & p[2:, 1:-1], p[1:-1, :-2] & p[1:-1, 2:]
        open_ns, open_ew = ~p[:-2, 1:-1] & ~p[2:, 1:-1], ~p[1:-1, :-2] & ~p[1:-1, 2:]
        valid = (ns & open_ew) | (ew & open_ns)
        d, m = c == door, c == mounted
        door_ok += int((valid & d).sum()); door_n += int(d.sum())
        mount_ok += int((valid & m).sum()); mount_n += int(m.sum())

        open_ = np.pad(~wl, 1, constant_values=True)
        lab, k = ndimage.label(open_, FOUR)
        outside = lab[0, 0]
        lab = lab[1:-1, 1:-1]
        door_touch = ndimage.binary_dilation(d, FOUR)
        fl = c == floor
        floor_all += int(fl.sum())
        for r in range(1, k + 1):
            if r == outside:
                continue
            region = lab == r
            a = int(region.sum())
            if a == 0:
                continue
            rooms += 1
            areas.append(a)
            rooms_door += bool((region & door_touch).any())
            floor_in += int((region & fl).sum())

    out["stray_wall"] = stray / max(1, blocking)
    out["door_valid"] = door_ok / max(1, door_n)
    out["mounted_valid"] = mount_ok / max(1, mount_n)
    out["rooms_per_window"] = rooms / max(1, n)
    out["room_area_median"] = float(np.median(areas)) if areas else 0.0
    out["rooms_with_door"] = rooms_door / max(1, rooms)
    out["floor_enclosed"] = floor_in / max(1, floor_all)
    return out


def table(real, gen):
    lines = [f"  {'metric':22s} {'real':>9s} {'generated':>10s}"]
    for k in real:
        lines.append(f"  {k:22s} {real[k]:9.4f} {gen.get(k, float('nan')):10.4f}")
    return "\n".join(lines)
