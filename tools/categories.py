"""Category mapping shared by windows.py, report.py and later training code.

A config's "categories" list gives id, name, display char and an optional rule. Rules are checked in
list order and the first match wins, so put specific categories (door, bridge) before general ones
(wall, floor). Category 0 must be the default (no rule).

Rule grammar: conditions joined by "&", each optionally negated with "!":
  edifice.door        cell's edifice has isDoor
  edifice.wall        cell's edifice has actsAsWall
  edifice=<DefName>   cell's edifice def
  terrain.bridge      terrain is a bridge (vanilla bridge defs, or name contains "Bridge")
  terrain.natural     terrain is natural ground/water (game Data natural files + name heuristics)
  terrain.any         any terrain recorded
  terrain=<DefName>   terrain def
Overrides win over rules, exact names first:
  "edifice_overrides" / "terrain_overrides": {def name: category name}
  "edifice_patterns" / "terrain_patterns":   [[regex, category name, optional rule], ...]
      first match wins; regexes are case-sensitive; the optional rule (same grammar as above)
      must also hold, e.g. "edifice.wall & !edifice.door" keeps FenceGate a door
"""
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GAME_DATA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "Data"))

# Natural terrain is only recorded by old RealRuins versions, often from biome mods; catch those by name.
NATURAL_WORDS = ("Soil", "Sand", "Mud", "Marsh", "Water", "Gravel", "Ice", "Dirt", "Moss", "Grass",
                 "Lava", "Riverbed", "Snow", "Tilled", "Fertile", "Clay", "Peat", "Ground")
FLOOR_WORDS = ("Tile", "Flagstone", "Floor", "Carpet", "Concrete", "Paved", "Plank", "Brick", "Stuffed")
NATURAL_SUFFIXES = ("_Rough", "_RoughHewn", "_Smooth")
NATURAL_FILES = ("Terrain_Natural.xml", "Terrain_Water.xml", "Terrain_Road.xml")
_DEFNAME = re.compile(r"<defName>([^<]+)</defName>")


def game_natural_terrain(game_data=DEFAULT_GAME_DATA):
    names = set()
    for f in NATURAL_FILES:
        for path in glob.glob(os.path.join(game_data, "*", "Defs", "TerrainDefs", f)):
            names.update(_DEFNAME.findall(open(path, encoding="utf-8").read()))
    return names


def game_bridges(game_data=DEFAULT_GAME_DATA):
    names = set()
    for path in glob.glob(os.path.join(game_data, "*", "Defs", "TerrainDefs", "Terrain_Foundation.xml")):
        # Bridge defs inherit from BridgeBase / HeavyBridge; the def names all contain "Bridge".
        names.update(n for n in _DEFNAME.findall(open(path, encoding="utf-8").read()) if "Bridge" in n)
    return names


class TerrainKinds:
    def __init__(self, game_data=DEFAULT_GAME_DATA):
        self.natural = game_natural_terrain(game_data)
        self.bridges = game_bridges(game_data)

    def is_bridge(self, name):
        return name in self.bridges or "Bridge" in name

    def is_natural(self, name):
        if name in self.natural or name.endswith(NATURAL_SUFFIXES):
            return True
        # "TileSandstone", "FlagstoneSlate" etc. contain natural words but are floors.
        if any(w in name for w in FLOOR_WORDS):
            return False
        return any(w in name for w in NATURAL_WORDS)


class Categories:
    """Builds lookup tables (vocab id -> category id) for one config + vocab."""

    def __init__(self, config, vocab, game_data=DEFAULT_GAME_DATA):
        self.cats = config["categories"]
        if self.cats[0].get("rule"):
            raise ValueError("category 0 is the default and must have no rule")
        self.ids = {c["name"]: c["id"] for c in self.cats}
        self.chars = {c["id"]: c.get("char", "?") for c in self.cats}
        kinds = TerrainKinds(game_data)
        tnames = vocab["terrain"]["names"]
        enames = vocab["edifice"]["names"]
        eflags = vocab["edifice"]["flags"]
        rules = [(c["id"], self._parse(c.get("rule", ""))) for c in self.cats[1:]]
        t_ovr = self._overrides(config, "terrain")
        e_ovr = self._overrides(config, "edifice")

        # Terrain table: category from terrain alone (empty terrain id 0 -> category 0).
        self.terrain_lut = np.zeros(len(tnames), np.uint8)
        for i, n in enumerate(tnames[1:], 1):
            facts = {"terrain.any": True, "terrain.bridge": kinds.is_bridge(n),
                     "terrain.natural": kinds.is_natural(n), "terrain=" + n: True}
            o = t_ovr(n, facts)
            self.terrain_lut[i] = o if o is not None else self._match(rules, facts, "terrain")
        # Edifice table: category from edifice alone; 0 means "no edifice rule matched, use terrain".
        # An edifice overridden to category 0 therefore also falls back to the terrain underneath.
        self.edifice_lut = np.zeros(len(enames), np.uint8)
        for i, n in enumerate(enames[1:], 1):
            wall, door = eflags[i]
            facts = {"edifice.wall": wall > 0, "edifice.door": door >= wall and door > 0, "edifice=" + n: True}
            o = e_ovr(n, facts)
            self.edifice_lut[i] = o if o is not None else self._match(rules, facts, "edifice")

    def _overrides(self, config, kind):
        exact = config.get(kind + "_overrides", {})
        patterns = [(re.compile(p[0]), p[1], self._parse(p[2] if len(p) > 2 else ""))
                    for p in config.get(kind + "_patterns", [])]

        def lookup(name, facts):
            if name in exact:
                return self.ids[exact[name]]
            for rx, c, conds in patterns:
                if rx.search(name) and all(facts.get(k, False) != neg for k, neg in conds):
                    return self.ids[c]
            return None
        return lookup

    @staticmethod
    def _parse(rule):
        conds = []
        for part in filter(None, (p.strip() for p in rule.split("&"))):
            neg = part.startswith("!")
            conds.append((part.lstrip("!"), neg))
        return conds

    @staticmethod
    def _match(rules, facts, domain):
        for cid, conds in rules:
            # A rule applies to a table only if all of its conditions are about that domain.
            if not conds or any(not c.startswith(domain) for c, _ in conds):
                continue
            if all(facts.get(c, False) != neg for c, neg in conds):
                return cid
        return 0

    def grid(self, extract):
        """Category grid (H, W) uint8 for one extract (dict-like of arrays)."""
        cat = self.terrain_lut[extract["terrain_id"]]
        ed = self.edifice_lut[extract["edifice_id"]]
        return np.where(ed > 0, ed, cat)

    def structure_ids(self):
        """Category ids that count as structure (everything except 0 and plain floor-like terrain)."""
        return [c["id"] for c in self.cats if c.get("structure")]
