"""Read RealRuins blueprint snapshots (.bp) into numpy grids.

A .bp file is gzipped (or plain) XML written by RealRuins' SnapshotGenerator:
  <snapshot version x z width height biomeDef mapSize inGameYear>
    <world seed tile gameId percentage/>
    <cell x z> <terrain def/>? <roof/>? <item def stuffDef rot stackCount actsAsWall isDoor>...</item>* </cell>
Cells are written row by row, z ascending, and only when they hold something.

What the format does and does not record (checked against the RealRuins 1.6 decompile):
- terrain only if it is player-buildable (current versions; old 2.x snapshots also kept natural terrain)
- <roof/> for ANY roof, natural thick rock roof included; the roof type is not stored
- things except natural rock (building.isNaturalRock); Smoothed* rock walls are NOT natural rock,
  so they are kept and flagged actsAsWall
- many uploads are truncated (also on the server); load() keeps the complete cells
"""
import gzip
import re
import xml.etree.ElementTree as ET

import numpy as np

# Natural terrain that only old snapshot versions recorded.
NATURAL_TERRAIN = ("Soil", "SoilRich", "Sand", "SoftSand", "Gravel", "MarshyTerrain", "Marsh", "Mud",
                   "MossyTerrain", "PackedDirt", "Ice", "WaterShallow", "WaterDeep", "WaterMovingShallow",
                   "WaterMovingChestDeep", "WaterOceanShallow", "WaterOceanDeep")
NATURAL_TERRAIN_SUFFIXES = ("_Rough", "_RoughHewn", "_Smooth")
RAW_ROCK = ("Granite", "Limestone", "Marble", "Sandstone", "Slate")

_CELL = re.compile(r"<cell [^>]*/>|<cell [^>]*>.*?</cell>", re.S)


def read_text(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


def load(path):
    """Returns (snapshot XML root, truncated flag)."""
    s = read_text(path)
    try:
        return ET.fromstring(s), False
    except ET.ParseError:
        # Truncated upload: keep whole cells only. The last complete-looking cell may still hold an
        # unfinished pawn record, so it is dropped too.
        head = s[:s.index(">") + 1]
        cells = _CELL.findall(s, len(head))
        return ET.fromstring(head + "".join(cells[:-1]) + "</snapshot>"), True


class Grids:
    """Boolean (z, x) grids for one snapshot. Row 0 is the snapshot's minimum z."""

    def __init__(self, root, truncated=False):
        self.root = root
        self.truncated = truncated
        self.width = int(root.get("width")) + 1
        self.height = int(root.get("height")) + 1
        shape = (self.height, self.width)
        self.wall = np.zeros(shape, bool)      # actsAsWall (constructed walls, smoothed rock, other full-fill)
        self.door = np.zeros(shape, bool)
        self.roof = np.zeros(shape, bool)      # any roof, natural or constructed
        self.floor = np.zeros(shape, bool)     # player-buildable terrain (floors, bridges)
        self.smooth = np.zeros(shape, bool)    # Smoothed* rock walls
        self.raw_rock = np.zeros(shape, bool)  # natural rock; never present in current versions
        self.natural_terrain_cells = 0
        self.last_z = 0
        for c in root.iter("cell"):
            x, z = int(c.get("x")), int(c.get("z"))
            if x >= self.width or z >= self.height:
                continue
            self.last_z = max(self.last_z, z)
            if c.find("roof") is not None:
                self.roof[z, x] = True
            t = c.find("terrain")
            if t is not None:
                d = t.get("def", "")
                if d in NATURAL_TERRAIN or d.endswith(NATURAL_TERRAIN_SUFFIXES):
                    self.natural_terrain_cells += 1
                else:
                    self.floor[z, x] = True
            for it in c.findall("item"):
                d = it.get("def", "")
                if d.startswith("Smoothed"):
                    self.smooth[z, x] = True
                if d in RAW_ROCK:
                    self.raw_rock[z, x] = True
                if it.get("isDoor") == "1":
                    self.door[z, x] = True
                elif it.get("actsAsWall") == "1":
                    self.wall[z, x] = True

    @property
    def barrier(self):
        return self.wall | self.door | self.smooth | self.raw_rock

    @property
    def usable_height(self):
        """Rows past the truncation point are missing, so the cut row counts as the snapshot edge."""
        return self.last_z if self.truncated else self.height


def open_grids(path):
    root, truncated = load(path)
    return Grids(root, truncated)
