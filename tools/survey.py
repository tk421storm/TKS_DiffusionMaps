"""Quick inventory of a RealRuins snapshot cache: versions, biomes, sizes, duplicates, truncation.

Usage: python -I survey.py <cache dir>
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_snapshot  # noqa: E402

HEADER = re.compile(r'<snapshot version="([^"]+)"[^>]*width="(\d+)" height="(\d+)" biomeDef="([^"]*)"')
WORLD = re.compile(r'tile="(\d+)" gameId="(\d+)"')


def main():
    root = sys.argv[1]
    versions, biomes, colonies = collections.Counter(), collections.Counter(), collections.Counter()
    areas = []
    counts = collections.Counter()
    for f in os.listdir(root):
        p = os.path.join(root, f)
        if not os.path.isfile(p):
            continue
        try:
            s = rr_snapshot.read_text(p)
        except Exception:
            counts["unreadable"] += 1
            continue
        m = HEADER.search(s)
        if not m:
            counts["no_header"] += 1
            continue
        counts["snapshots"] += 1
        versions[m.group(1)] += 1
        biomes[m.group(4)] += 1
        areas.append(int(m.group(2)) * int(m.group(3)))
        w = WORLD.search(s)
        if w:
            colonies[w.groups()] += 1
        counts["truncated"] += not s.rstrip().endswith("</snapshot>")
        counts["with_doors"] += 'isDoor="1"' in s
        counts["with_walls"] += 'actsAsWall="1"' in s
    areas.sort()
    for k, v in sorted(counts.items()):
        print(f"{k:12s} {v}")
    print("unique colonies (seed tile + gameId):", len(colonies), " most snapshots of one colony:",
          max(colonies.values(), default=0))
    if areas:
        print("bounding-box area median", areas[len(areas) // 2], " p90", areas[int(len(areas) * 0.9)])
    print("versions", versions.most_common(8))
    print("biomes  ", biomes.most_common(12))


if __name__ == "__main__":
    main()
