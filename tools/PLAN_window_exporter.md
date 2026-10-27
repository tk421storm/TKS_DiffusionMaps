# Plan: training-window exporter (step a)

Status: **built 2026-10-27** (`extract.py`, `categories.py`, `windows.py`, `report.py`,
`../configs/v1.json`). The plan text below is kept as written. Where the build differs:

- `edge_ring` was dropped. A structure component counts as cut when any part of it lies outside the window.
  `mask_grow` (default 1) grows the masked area around cut components.
- New window filters:
  - `min_complete_buildings` (default 1): the window must hold at least one whole enclosed building of an
    allowed class. This removes windows that only contain paths and loose wall pieces.
  - `max_masked_fraction` (v1: 0.5).
- New `edifice_patterns` / `terrain_patterns` (regex → category), next to the exact-name overrides.
  v1 uses them to send single-cell-recorded machines (generators, turbines, thrusters...) and mod
  rock/snow to `empty`, and player packed dirt to `floor`.
- Fences, palisades, razor wire, vents, coolers and windows all carry `actsAsWall`. The v1 config splits them out
  by name (pattern plus "is a wall, not a door") into `fence` (id 5) and `wall_mounted` (id 6, things set into wall
  slots: vents, coolers, A/C, windows, intake fans). Both are rare: 0.1% and 0.04% of training cells,
  appearing in 308 and 1,499 of 6,088 train windows. Training will need class weighting.
- `building_class` codes are listed in `extract.CLASSES`.

Results on 3,533 raw snapshots (v1 config):

| Stage | Result | Time |
|---|---|---|
| Extract | 3,533 snapshots, 0 errors, 38 MB | 34 s |
| Windows | 6,489 windows (6,088 train / 401 val) from 1,439 snapshots | 11 s |

- 4,582 of those windows have some cells masked; about 10% of all cells are masked.
- With `edge_policy: reject` the same config gives 2,364 windows.
- Categories within the loss mask (train split): empty 80.1%, floor 14.2%, wall 5.0%, bridge 0.4%, door 0.3%.

## Decisions this plan follows (user, 2026-10-27)

- Flat ground only; mountains and mountain bases are set aside.
- Train on **complete buildings** inside **48x48 windows** cut from whole snapshots.
- Categories keep **bridge** for later use.
- Model starts **unconditional**; inpainting around obstacles comes later through masks, so a mask
  channel is planned but not filled yet.
- Data: the cached RealRuins snapshots only, for now.

## Design goals

Window size, categories, filters and outputs change with a config file, not code. Each change
reruns only the stages after it. Every output carries its config and code version so runs can be
reproduced and compared.

## Pipeline

```
raw .bp files ──(1) extract──> per-snapshot intermediates ──(2) windows──> dataset shards ──(3) report
                 slow, rarely rerun      data/extract/<id>.npz        fast, rerun per config     data/sets/<name>/
```

### Stage 1: extract (one file per snapshot, independent of window size and categories)

For each `.bp` (using `rr_snapshot.py`), save `data/extract/<snapshot id>.npz` with **full-detail layers**
so later category changes do not need a re-parse:

| Layer | dtype | Content |
|---|---|---|
| `terrain_id` | uint16 | index into a global terrain vocabulary (0 = none recorded) |
| `edifice_id` | uint16 | thing def on the cell flagged `actsAsWall`/`isDoor`, else 0 |
| `thing_ids` | sparse list | every other thing: (z, x, def id, stuff id, rot, stack); furniture later |
| `roof` | bool | any roof |
| `building_label` | int16 | building index from `segment.classify` (0 = none) |
| `building_class` | per-building table | free_standing / mountain_adjacent / unroofed / clipped, size, area |
| `mountain` | bool | mountain-blob cells and smoothed walls, grown by 2 cells |
| `valid` | bool | cell is inside the snapshot and below the truncation row |

Plus `data/extract/vocab.json`: def name → id and count, for terrain, edifices, things and stuff.
It is written incrementally and only appended to, so ids stay stable.
`data/extract/index.json` lists each snapshot's metadata: file, version, biome, map size, world
seed, tile, gameId, truncated, size.

Rerun stage 1 only when `rr_snapshot.py` / `segment.py` change (bump `EXTRACT_VERSION`) or new
raw files arrive. Already-extracted files with the current version are skipped.

### Stage 2: windows (driven by a config file)

`configs/v1.json` (example; every key has a default):

```json
{
  "name": "v1_48",
  "window": 48,
  "stride": 8,
  "categories": [
    {"id": 0, "name": "empty"},
    {"id": 1, "name": "wall",   "rule": "edifice:actsAsWall"},
    {"id": 2, "name": "door",   "rule": "edifice:isDoor"},
    {"id": 3, "name": "floor",  "rule": "terrain:buildable & !bridge"},
    {"id": 4, "name": "bridge", "rule": "terrain:bridge"}
  ],
  "category_overrides": {"Cooler": "wall", "Vent": "wall"},
  "edge_policy": "mask",
  "edge_ring": 2,
  "min_structure_cells": 20,
  "max_overlap": 0.5,
  "exclude": ["mountain", "invalid"],
  "building_classes": ["free_standing", "unroofed_enclosure"],
  "augment": "none",
  "split": {"by": "colony", "val": 0.1, "seed": 1},
  "seed": 1
}
```

Steps per snapshot:
1. Map layers to categories using `categories` + `category_overrides` (ordered rules; the first match wins).
   The override values above are placeholders; fill them in from the `vocab.json` report of which defs carry
   `actsAsWall` but are not walls.
   Unknown edifice defs fall back on their `actsAsWall` / `isDoor` flags.
2. Candidate windows on a `stride` grid. The snapshot is padded by `window` cells of `empty`, since the
   area outside the home area is open ground.
3. Reject a window if it has any `exclude` cell (mountain, past truncation), fewer than
   `min_structure_cells` wall/door cells, or a building whose class is not in `building_classes`.
4. **Edge policy**: what to do with buildings cut by the window edge (walls inside `edge_ring`):
   - `reject`: drop the window (complete buildings only; about 2.2k windows from the cache)
   - `mask` (default): keep the window, but mark every cell of a cut building in a
     **loss mask** so the model is never trained on fragments (training uses complete buildings
     only; about 22k windows before masking, fewer usable cells)
5. Greedy de-duplication: skip a window overlapping an already chosen one by more than `max_overlap`.
6. Train/val split **by colony** (world seed + tile + gameId), never by window, so near-identical
   windows from one base cannot land on both sides.

Output `data/sets/<name>/`:
- `train.npz`, `val.npz`: `cells` uint8 [N, W, W], `loss_mask` bool [N, W, W],
  `cond_mask` bool [N, W, W] (all true for now; reserved for inpainting/obstacles), `source` index
- `manifest.json`: full resolved config, code version (git hash of `tools/`), extract version,
  counts at each filter step, category histogram
- Augmentation (8 rotations/flips) is applied **at training time**, not stored. `augment` stays
  in the config only for setups that want it stored.

### Stage 3: report

`report.py <set dir>`: counts, category frequencies, windows per biome/version, the rejection
breakdown, `--show N` ASCII samples (same legend as `segment.py`), and an optional PNG contact
sheet for quick visual checks.

## How the likely changes map onto this

| Change | What to do |
|---|---|
| Window size (e.g. 64) | new config, rerun stage 2 |
| Add/merge categories (indoor vs outdoor floor, roof, furniture classes) | edit `categories`, rerun stage 2 (layers already extracted) |
| Furniture / multi-cell things | needs footprint sizes: dump `ThingDef.size` from the game (dev action, includes mods) into `vocab.json`, then a new category rule `thing:<group>` |
| Room-role labels / wealth | in-game labelling harness writes `roles.npz` per snapshot next to the extract; stage 2 gains a `labels` key |
| Conditioning (tech level, size) | per-window vector in `cond`; source fields come from the extract index |
| Obstacle inpainting | fill `cond_mask` from sampled vanilla-map buildable masks (stage 2 option) |
| More raw data | drop files into `data/raw/`, rerun stage 1 (incremental) and stage 2 |
| Mountains (later) | `exclude` without `mountain`, `building_classes` + `mountain_adjacent`, a `rock` category |

## Files to create

```
tools/
  rr_snapshot.py      (exists)
  segment.py          (exists)
  survey.py           (exists)
  extract.py          stage 1
  categories.py       rule parser + category mapping, shared by stages 2-3 and later training code
  windows.py          stage 2
  report.py           stage 3
configs/v1.json
data/                 (gitignored) raw/, extract/, sets/
```

## Before running

- **Copy the cache out first.** RealRuins' `diskCacheLimit` defaults to 256 MB and the cache is
  already 266 MB, so RealRuins is evicting old files. Copy
  `%USERPROFILE%\AppData\LocalLow\Ludeon Studios\RimWorld by Ludeon Studios\RealRuins\*.bp` into
  `data/raw/`. Raise the limit in RealRuins' Network/cache settings before downloading more.
- Strip pawn/corpse records (names, health, story) at extraction; they are not needed.
