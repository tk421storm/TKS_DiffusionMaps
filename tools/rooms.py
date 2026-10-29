"""Room types from room contents, driven by the config's "rooms" section. Used by windows.py.

A room is a 4-connected region of open cells (not wall, door, fence or rock) that doesn't touch the snapshot's edge or its
invalid part, has min_cells..max_cells cells and is at least min_roof roofed. Its type comes from the things inside it:

  "things": [[regex, target], ...]   first match on the thing def name wins. target is a category name, or a special:
                                     "ignore" (lamps, chairs, conduits...), "bed", "table"
  decision, in order:
    1. any thing whose target is in "priority" (e.g. hospital, nursery): the first such target present
    2. beds: 1..barracks_beds-1 -> bed_one, barracks_beds or more -> bed_many
    3. dining_min_tables tables, or one table of dining_big_table_cells cells or more ("2x4" in its name) -> dining
    4. the category with the most matching things (ties: earlier rule wins)
    5. storage_stacks or more item stacks (stack > 1) and nothing else typed -> storage
    6. nothing but ignored things -> empty; anything else (including a single small table) -> untyped
  A room typed storage that touches a cooler (edifice or thing matching freezer_coolers) becomes freezer.
The room's cells that hold `empty` or `floor` take the type's category; other cells (turrets, cover...) keep theirs.
"""
import re

import numpy as np
from scipy import ndimage

IGNORE, BED, TABLE = -1, -2, -3
SPECIAL = {"ignore": IGNORE, "bed": BED, "table": TABLE}
_SIZE = re.compile(r"(\d)x(\d)")


class RoomTyper:
    def __init__(self, config, vocab, cats):
        rr = config["rooms"]
        self.rr = rr
        ids = cats.ids
        self.cat = lambda name: ids[name]
        self.targets = []                      # category ids that things can vote for, in rule order
        tnames = vocab["thing"]["names"]
        skip = re.compile(rr.get("skip_things", r"^(Frame_|Blueprint_|Minified)"))
        rules = []
        for rx, target in rr["things"]:
            if target in SPECIAL:
                code = SPECIAL[target]
            else:
                if ids[target] not in self.targets:
                    self.targets.append(ids[target])
                code = self.targets.index(ids[target])
            rules.append((re.compile(rx), code))
        # per thing def: vote code (>= 0: index into targets), or IGNORE/BED/TABLE, or -9 = unmatched
        self.code = np.full(len(tnames), -9, np.int16)
        self.big_table = np.zeros(len(tnames), bool)
        cooler = re.compile(rr["freezer_coolers"])
        self.cooler_thing = np.zeros(len(tnames), bool)
        for i, n in enumerate(tnames):
            if not n or skip.search(n):
                self.code[i] = IGNORE if n else -9
                continue
            self.cooler_thing[i] = bool(cooler.search(n))
            for rx, code in rules:
                if rx.search(n):
                    self.code[i] = code
                    if code == TABLE:
                        m = _SIZE.search(n)
                        self.big_table[i] = bool(m) and int(m.group(1)) * int(m.group(2)) >= rr["dining_big_table_cells"]
                    break
        self.cooler_edifice = np.array([bool(n) and bool(cooler.search(n)) for n in vocab["edifice"]["names"]])
        self.priority = [self.targets.index(ids[p]) for p in rr["priority"]]
        self.floorlike = np.array([ids["empty"], ids["floor"]])

    def assign(self, e, cat, blocking, valid):
        """Writes room types into `cat` (in place). blocking: cells that bound rooms. Returns {category id: rooms}."""
        rr = self.rr
        open_ = ~blocking & valid
        lab, k = ndimage.label(open_)
        counts = {}
        if k == 0:
            return counts
        edge = np.zeros(k + 1, bool)
        border = ~ndimage.binary_erosion(valid, border_value=0)
        edge[np.unique(lab[border & open_])] = True
        edge[0] = True
        idx = np.arange(k + 1)
        size = ndimage.sum(open_, lab, idx)
        roof = ndimage.sum(open_ & e["roof"], lab, idx) / np.maximum(size, 1)
        is_room = ~edge & (size >= rr["min_cells"]) & (size <= rr["max_cells"]) & (roof >= rr["min_roof"])
        if not is_room.any():
            return counts

        T = len(self.targets)
        votes = np.zeros((k + 1, T), np.int32)
        beds = np.zeros(k + 1, np.int32)
        tables = np.zeros(k + 1, np.int32)
        big = np.zeros(k + 1, bool)
        stacks = np.zeros(k + 1, np.int32)
        other = np.zeros(k + 1, np.int32)          # things not ignored
        t = e["things"]
        if len(t):
            r = lab[t["z"], t["x"]]
            code = self.code[t["def"]]
            sel = r > 0
            r, code, d, st = r[sel], code[sel], t["def"][sel], t["stack"][sel]
            v = code >= 0
            np.add.at(votes, (r[v], code[v]), 1)
            np.add.at(beds, r[code == BED], 1)
            np.add.at(tables, r[code == TABLE], 1)
            big[r[(code == TABLE) & self.big_table[d]]] = True
            np.add.at(stacks, r[st > 1], 1)
            np.add.at(other, r[code != IGNORE], 1)
        cool = self.cooler_edifice[e["edifice_id"]]
        if len(t):
            ct = self.cooler_thing[t["def"]]
            cool[t["z"][ct], t["x"][ct]] = True
        near_cool = np.zeros(k + 1, bool)
        near_cool[np.unique(lab[ndimage.binary_dilation(cool) & open_])] = True

        room_type = np.zeros(k + 1, np.int32)
        storage, freezer = self.cat(rr["storage"]), self.cat(rr["freezer"])
        for i in np.nonzero(is_room)[0]:
            pr = next((p for p in self.priority if votes[i, p]), None)
            if pr is not None:
                c = self.targets[pr]
            elif beds[i]:
                c = self.cat(rr["bed_many"] if beds[i] >= rr["barracks_beds"] else rr["bed_one"])
            elif tables[i] >= rr["dining_min_tables"] or big[i]:
                c = self.cat(rr["dining"])
            elif votes[i].any():
                c = self.targets[int(np.argmax(votes[i]))]
            elif stacks[i] >= rr["storage_stacks"]:
                c = storage
            elif other[i] == 0:
                c = self.cat(rr["empty"])
            else:
                c = self.cat(rr["untyped"])
            if c == storage and near_cool[i]:
                c = freezer
            room_type[i] = c
            counts[c] = counts.get(c, 0) + 1
        typed = room_type[lab]
        sel = (typed > 0) & np.isin(cat, self.floorlike)
        cat[sel] = typed[sel]
        return counts
