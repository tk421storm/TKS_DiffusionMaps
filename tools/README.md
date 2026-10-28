# DiffusionMaps analysis tools

Python scripts for inspecting RealRuins blueprint snapshots (`.bp`) as possible training data.
Written 2026-10-27 during the feasibility review of `../proposal.txt`.

## Setup

- Python 3.10+ with `numpy` and `scipy` (tested with Python 3.11.9, numpy 1.26.4, scipy 1.16.3)
- Run with `python -I` (isolated mode) so nothing in the current folder is imported by accident.
- Snapshot cache: RealRuins stores downloaded blueprints in
  `%USERPROFILE%\AppData\LocalLow\Ludeon Studios\RimWorld by Ludeon Studios\RealRuins`
  (2,646 files on 2026-10-27). Subfolders such as `david-250-50` are per-world folders and are skipped.

```
set CACHE=%USERPROFILE%\AppData\LocalLow\Ludeon Studios\RimWorld by Ludeon Studios\RealRuins
python -I survey.py "%CACHE%"
python -I segment.py "%CACHE%"
python -I segment.py "%CACHE%" --limit 100 --show 2
```

In Git Bash use `"$APPDATA/../LocalLow/Ludeon Studios/RimWorld by Ludeon Studios/RealRuins"`.

## Scripts

| File | What it does |
|---|---|
| `rr_snapshot.py` | Library. `open_grids(path)` reads a `.bp` file (gzipped or plain, truncated or not) into boolean `(z, x)` grids: `wall`, `door`, `roof`, `floor`, `smooth`, `raw_rock`. Row 0 is the snapshot's minimum z. |
| `survey.py` | Inventory of a cache: snapshot count, truncated count, unique colonies (tile + gameId), bounding-box sizes, RealRuins versions, biomes. |
| `segment.py` | Splits each snapshot into buildings and classifies them (below). Prints totals and free-standing size stats. `--show N` draws N sample buildings per class as text, `--limit N` samples N random files, `--seed` makes the sample repeatable. |

ASCII legend for `--show`: `#` wall, `S` smoothed rock wall, `+` door, `o` roof inside walls,
`r` roof outside walls (mountain or overhang), `.` floor. North is at the top.

## Training-window pipeline

Design and results: `PLAN_window_exporter.md`. Run from this folder:

```
python -I extract.py                                   # stage 1: data/raw/*.bp -> data/extract (incremental)
python -I windows.py ../configs/v1.json                 # stage 2: -> data/sets/v1_48/{train,val}.npz + manifest.json
python -I report.py ../data/sets/v1_48 --show 5         # stage 3: counts + ASCII samples (~ = masked structure)
python -I report.py --vocab ../configs/v1.json          # which defs map to which category
```

| File | What it does |
|---|---|
| `extract.py` | Writes one `.npz` of full-detail layers per snapshot, plus a vocabulary (`vocab.json`) whose ids never change and `index.json`. Rerun after adding files to `data/raw/`. Bump `EXTRACT_VERSION` when parsing or segmentation changes. |
| `categories.py` | Config-driven def → category rules, overrides and regex patterns. Shared with training code. |
| `windows.py` | Cuts windows per config. Edge policy `mask` or `reject`, filters, de-duplication, train/val split by colony. |
| `report.py` | Inspects a set or the category mapping. |

To get more data, copy new `.bp` files (including those in the per-world subfolders) into `data/raw/`, then rerun
`extract.py` and `windows.py`. `data/` is generated and can be rebuilt from `data/raw/` at any time.

## GPU training environment (WSL2 + ROCm, AMD RX 7900 XTX)

Verified 2026-10-27 on Windows 10 19045, Adrenalin 26.3.1, WSL 3.0.1, Ubuntu-22.04.
AMD lists ROCDXG for Windows 11 only, but it works here.

1. Inside WSL, ROCm uses ROCDXG, **not** the old `hsa-runtime-rocr4wsl-amdgpu`:
   - installed: `hsa-rocr` 7.2.0, `rocdxg-roct` 1.2.2, `rocminfo`, plus the libraries the PyTorch wheel links against:
     `hip-runtime-amd miopen-hip hipblas hipblaslt hipfft hiprand hipsolver hipsparse hipsparselt rccl rocblas rocsolver roctracer`
   - `/opt/rocm/lib` registered with the linker: `echo /opt/rocm/lib | sudo tee /etc/ld.so.conf.d/rocm.conf && sudo ldconfig`
   - `export HSA_ENABLE_DXG_DETECTION=1` (required before ROCm 7.13) and `/opt/rocm/bin` on `PATH`
   - do not use the snap `rocminfo` (community ROCm 6.4 build; fails with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`)
2. `bash tools/setup_wsl_torch.sh` installs AMD's ROCm 7.2.1 PyTorch 2.9.1 wheels into `~/venvs/dm`.
   These are AMD's "lw" wheels, which do not bundle ROCm, hence the system libraries in step 1.
3. Check the GPU:
   - `python -u tools/gpu_smoke.py` (step by step, shows where a hang happens)
   - `python -u tools/bench_gpu.py` (speed): 24.8 ms/step GPU vs 1,261 ms/step CPU for a 3.4M-parameter U-Net, batch 64 → **51x faster**, about 2 s per epoch of v1

**Known hang:** GPU processes can get stuck in the kernel while exiting, after their work is done.
- Symptoms: `D` state, wchan `vmbus_teardown_gpadl`. `kill -9` does not work. While the process is stuck, new GPU processes block in
  `dxgglobal_acquire_process_adapter_lock`. Only `wsl --shutdown` clears it.
- None of these fixed it: matching ROCm 7.2.1, `HSA_ENABLE_SDMA=0`, `os._exit`. Large copies alone don't trigger it (`hang_probe.py`).
- Suspected cause: the WSL kernel (6.18) against a Windows 10 host. AMD tests kernel 5.15 on Windows 11. Not verified.
- Workaround: run GPU jobs through `run_wsl_gpu.sh`, which shuts WSL down once the job prints `done`.
- Also keep training data on the GPU, and avoid pinned memory: allocating 128 MB pinned fails under WSL.

**Upgrading ROCm:** the `rocdxg-roct` deb installs through the `/opt/rocm` symlink. After an upgrade switches the symlink,
reinstall it (`sudo dpkg -i rocdxg-roct_*.deb && sudo ldconfig`), or `librocdxg.so` is not found.
Before upgrading, `simulate_rocm_upgrade.sh <version>` dry-runs the upgrade without root.

## Building classes (`segment.py`)

A *building* is an enclosed area (open cells that walls cut off from the snapshot border) plus its walls.
Rooms that share walls form one building.

| Class | Rule |
|---|---|
| `free_standing` | constructed walls, at least 50% roofed, no mountain within 2 cells, not on an edge |
| `mountain_adjacent` | more than 10% of its walls are `Smoothed*` rock, or it touches a mountain blob |
| `unroofed_enclosure` | less than 50% roofed: animal pens, walled yards, killbox courts |
| `clipped_at_edge` | touches the snapshot edge or the truncation row, so it may be incomplete |

**Mountain blob:** at least 40 connected roofed cells that are *not* enclosed by walls, and not mostly lined by walls.
The second rule keeps a constructed roof that leaks through a gap in a wall from counting as mountain.
The constants are at the top of `segment.py`.

Results on the 2026-10-27 cache: 6,642 free-standing, 2,782 mountain-adjacent, 889 unroofed,
257 clipped. Precision has **not** been measured. Spot checks look right, but buildings with gaps in their walls
are a known failure (they merge with the outside and are lost, or partly counted).

## What the snapshot format records

Checked against a decompile of RealRuins 1.6 (`SnapshotGenerator.Generate` / `EncodeThing`):

- **Area:** the bounding box of the home area only.
- **Terrain:** only player-buildable terrain (floors, bridges). Old 2.x versions also wrote natural
  terrain (227 of 2,646 cached files).
- **Roof:** `<roof/>` for any roof, **natural thick rock roof included**; the roof type is not stored.
- **Things:** everything except natural rock (`building.isNaturalRock`). `Smoothed*` rock walls have
  `isNaturalRock=false`, so they are kept and flagged `actsAsWall="1"`. Doors carry `isDoor="1"`.
  Pawns and corpses are written with names, health and story (strip these before using the data).
- **Truncation:** 1,653 of 2,646 files (62%) are cut off mid-file, and the copies on the server
  (`https://realruinsv2.sfo2.digitaloceanspaces.com/<id>.bp`) are cut off at the same point. Cells run
  bottom to top, so the top rows are missing. `rr_snapshot.load` keeps the complete cells and
  `Grids.usable_height` marks the cut row.

Consequences:

- Unmined mountain shows up as roof with nothing else, which is why the mountain mask works.
- An unmined rock cell and a mined-out empty cell look exactly the same, so the rock face of a mountain base cannot be recovered
  (except where walls were smoothed).

## Data source etiquette

The RealRuins API (`https://api.woolstrand.art/maps/random?limit=N`) and bucket are run by the
RealRuins author (Woolstrand / mr.teterew) for the mod's own use. Ask before downloading in bulk or
training on player uploads, and keep request rates low.
