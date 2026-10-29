# DiffusionMaps history

A dated log of experiments, results and decisions. Newest entries at the bottom.
Run outputs in `runs/` are not in git; the files needed to read each entry are copied into `docs/history/<run>/`.

## 2026-10-27: feasibility and dataset

- Data source: cached RealRuins snapshots (3,533 unique `.bp` files). Details of the format and its limits are in
  `tools/README.md`; dataset design in `tools/PLAN_window_exporter.md`.
- Decisions:
  - flat-ground faction bases only (mountain bases set aside, because vanilla faction bases always sit on flat, unroofed ground)
  - 48x48 windows cut from whole snapshots
  - categories empty / wall / door / floor / bridge / fence / wall_mounted
  - unconditional model first, inpainting later
- Set `data/sets/v1_48` (`configs/v1.json`, edge policy `mask`): 6,489 windows, 6,088 train / 401 val, split by colony.
  Edge policy `reject` would leave only 2,364.
- GPU: AMD RX 7900 XTX through WSL2 + ROCDXG on Windows 10 (not officially supported, but it works). PyTorch 2.9.1 + ROCm 7.2.1.
  51x faster than CPU. GPU processes can hang on exit; `tools/run_wsl_gpu.sh` works around it.

## 2026-10-27: v1 model (`v1_mdm`)

Files: `docs/history/v1_mdm/` (config, full training log, metrics per logit-adjust strength, ONNX check, sample sheets, `model_meta.json`).

### Setup

- MaskGIT-style masked discrete diffusion. U-Net with base 64 and mults [1,2,4], 2 res blocks per level and attention at 12x12;
  conditioned on the masked fraction (FiLM). 16.47M parameters.
- Loss: cross-entropy on masked cells, class weights freq^-0.5 capped at 20, cells cut at window edges excluded.
  8-way rotation/flip augmentation on the GPU.
- AdamW lr 3e-4, cosine schedule, batch 64, 150 epochs, EMA 0.999. About 20 s per epoch (the first epoch took 263 s to compile kernels),
  about 1 hour total.
- Sampling: 16 steps, cosine unmasking schedule, temperature 1.0, confidence noise 4.5.

### Training

| Epoch | Val loss | Hidden cells correct at 50% masked | at 90% masked |
|---|---|---|---|
| 1 | 1.803 | 32.5% | 61.3% |
| 10 | 0.633 | 96.2% | 91.5% |
| 50 | 0.428 | 97.8% | 93.4% |
| 100 | 0.413 | 97.9% | 93.7% |
| 150 | **0.407** | **98.0%** | 93.8% |

Val loss was still slowly falling at 150, with no sign of overfitting (train loss 0.423).

### The over-generation problem and logit adjustment

The raw model (no correction) produced far too much rare stuff. At epoch 150: 48 rooms per window with a median of 2 cells,
26% bridge, 11% wall_mounted, only 24% empty. Cause: the class weights teach it to over-predict rare categories, and
sampling compounds this over 16 steps. The loss looked fine throughout.

Fix without retraining: at sampling, add `logit_bias = -a * log(class_weight)` to the logits. Sweep at epoch 150, 64 samples each:

| Metric | Real | a=0 | a=0.5 | **a=0.6** | a=0.7 | a=0.8 |
|---|---|---|---|---|---|---|
| Empty cells | 89.4% | 24.3% | 73.0% | 78.9% | 83.6% | 86.5% |
| Wall cells | 3.4% | 8.0% | 6.6% | 5.1% | 3.9% | 3.0% |
| Floor cells | 6.8% | 22.5% | 16.7% | 14.4% | 11.5% | 9.8% |
| Bridge cells | 0.08% | 26.4% | 3.2% | 1.3% | 0.85% | 0.57% |
| Rooms per window | 2.75 | 48.3 | 4.6 | **3.3** | 2.1 | 1.4 |
| Median room size | 25 | 2 | 16.5 | 19 | 20 | 25 |
| Doors set in a wall line | 88% | 51% | 86% | **87%** | 83% | 86% |
| Rooms with a door | 97% | 44% | 81% | 77% | 75% | 74% |
| Floor inside rooms | 48% | 46% | 24% | 21% | 20% | 19% |
| Stray wall pieces | 1.4% | 1.2% | 0.9% | 0.9% | 1.3% | 1.5% |

a=1.0 (full correction, checked at epoch 41) is far too sparse: 0.6 rooms per window. **0.6 was chosen**, as the best balance of room
count and door placement. It is stored in `model_meta.json`.

Sample sheets: `docs/history/v1_mdm/samples_e150_a0.6.png` (chosen) and `samples_e150_raw.png` (no correction).

### ONNX (the in-game path)

- Opset 17, ONNX Runtime 1.23.2, 63 MB.
- Logits vs PyTorch: max difference 1.15e-5, 100% same argmax.
- Deterministic 16-step sampling, PyTorch vs numpy + ORT reference sampler (`export_onnx.ort_sample`): 100% identical.
- CPU, batch 1: 24 ms per forward and 0.41 s per sample on all threads; 86 ms and 1.33 s on one thread.

### Analysis

What works:
- Structure: walls form rooms, rooms share walls, and doors sit in wall lines as often as in real data (87% vs 88%).
  Stray walls are rarer than in real data. The samples read as plausible RimWorld bases: rows of bedroom-sized rooms, multi-room buildings, paths.
- The network learned the data well (98% of hidden cells correct at 50% masked). The remaining problems are in the balance of what it
  generates, not in what it learned.

Weaknesses, with likely causes:
- **Too much outdoor floor** (14% vs 7% of cells, only 21% inside rooms vs 48%). The single bias per class cannot separate indoor from
  outdoor floor. Candidates: milder class weights, or no weight on floor.
- **Rooms without doors** (77% vs 97%). Doors are rare and placed late. Candidates: a separate door bias, more sampling steps, or a
  repair pass in game.
- **Some walls don't close.** No metric for this yet; add a wall-closure check.
- **Bridges appear as blobs on dry land.** The model has no information about water, so it needs terrain conditioning.
- **Wall-mounted objects almost never appear** (0.02% vs 0.03%, and only 72% placed correctly). Too rare for the shared bias.
- Rooms are slightly small (median 19 vs 25 cells).

Lessons:
- Track generation metrics during training, not only the loss. The loss improved steadily while raw samples got worse (rooms per window 3 → 29 → 48).
- Logit adjustment is a cheap, retraining-free knob, and it carries straight into ONNX through `model_meta.json`.

### Next options (not yet chosen)

1. Improve v1: milder class weights, wall-closure metric, 24 steps, per-class bias for floor and door.
2. Inpainting test: fix obstacle cells and check that the model builds around them. This is the step toward filling vanilla's settlement rect.
3. In-game: a minimal C# GenStep running the ONNX model.

## 2026-10-28: step 1a, the model runs inside RimWorld 1.6

A minimal C# mod (`DiffusionMaps.csproj`, `Source/`) loads `Models/v1_mdm/model.onnx` with ONNX Runtime 1.23.2 (NuGet
`Microsoft.ML.OnnxRuntime.Managed`, net48) and samples layouts from dev-mode actions (category "DiffusionMaps").

- The native `onnxruntime.dll` is shipped in `Native/win-x64/` and loaded by full path (`OnnxNative.cs`) before first use.
  It can't go in `Assemblies/`: RimWorld loads every DLL there as .NET and stops at the first failure. Loading it ourselves also
  avoids the old Windows ML copy in `System32`.
- Under Unity Mono (RimWorld 1.6.4871) the managed package and its dependencies (System.Memory, System.Buffers,
  System.Numerics.Vectors, System.Runtime.CompilerServices.Unsafe) loaded without errors.
- `Source/LayoutModel.cs` is a C# port of the MaskGIT sampler with no game code, so it can be shared with a 1.7 build.
  Deterministic sampling matches the Python reference (`train/ort_reference.py`) on all 2,304 cells, both outside the game
  (`tools/csharp_check`, .NET Framework 4.8) and in game. Doing the log-softmax in float32 like numpy was necessary:
  with doubles, near-ties in confidence ordered differently and 8 cells differed.

In-game timings (i7-10700K, main thread):

| | Time |
|---|---|
| First model load | 0.91 s |
| One layout, all threads | 0.34 to 0.43 s |
| One layout, 1 thread | 1.35 s |

Conclusion: the ONNX route works on 1.6, and is fast enough to generate during map generation. Next: step 1b (vanilla settlement
next to a generated layout, placed as real walls, doors and floors) and step 2 (vanilla layout export for three-way metrics,
plus the inpainting test).

## 2026-10-28: step 1b, compare tool (vanilla settlement next to ours)

Dev actions (map only, category "DiffusionMaps"):
- **Compare with vanilla settlement...**: pick a faction, then click. Vanilla's settlement generator (`BaseGen` symbol `settlement`, as
  `GenStep_Settlement` calls it, no pawns) builds a 36x36 base left of the click, and ours builds one to the right with the same faction.
- **Place diffusion layout...**: builds only ours.

`Source/LayoutPlacer.cs` builds a layout:
- clears plants, items and buildings (both sides, since real map generation adds plants after the settlement)
- fixes obstacle cells (water, marsh, natural rock, mountain roof) as empty before sampling
- places walls, doors and fences in the faction's cheap wall material and floor in its basic floor, both chosen with vanilla's
  `BaseGenUtility`
- places vents for wall_mounted cells, and lays bridge cells as floor
- roofs enclosed rooms

First in-game test (7 pairs, `Player.log`): obstacle handling works, with up to 579 cells fixed and built around. But our layouts were
very sparse (1-2 small buildings, almost no floor), and some rooms had no door or unclosed walls.

## 2026-10-28: inpainting test, density and seed shapes (`train/inpaint_test.py`)

Cause of the sparse layouts: the 1b tool fixed the 6-cell ring of the 48x48 window outside the 36x36 square as empty. The model reads
the known-empty context as "this is an empty area". The test reproduces the in-game setting in Python (48 samples per variant,
measured on the 36x36 square). Files: `docs/history/v1_mdm/inpaint/`.

| 36x36 square | Real (centre crop) | Ring | No ring, best crop | 1 corner seed | 2 corners | 4 corners | 7 corners |
|---|---|---|---|---|---|---|---|
| Wall/door cells | 114 | 36 | 122 | 217 | 251 | 335 | 454 |
| Floor | 22% | 0.1% | 20% | 34% | 37% | 41% | 36% |
| Complete rooms | 2.1 | 1.0 | 3.2 | 5.5 | 7.6 | 12.5 | 20.7 |
| Rooms with a door | 95% | 75% | 81% | 82% | 80% | 74% | 60% |
| Wall cells cut by the square's edge | 6.2 | 0 | 0.8 | 7.9 | 9.4 | 12.2 | 16.7 |
| Seeds completed into a room | | | | 65% | 78% | 77% | 76% |

- **No ring** brings back real player-base density. Choosing the best 36x36 crop of the 48x48 window cuts almost no walls.
- **Seeding** (fixing a few wall cells for the model to complete, the user's idea) is a strong density control: even one L-shaped corner
  almost doubles the structure, and 4 or more fill the square like a city block. A faction base probably needs 1-2. Vanilla's room count is
  still to be measured.

Seed shapes, one or two per square (the user asked whether diagonal walls break up the square look):

| | Real | No seeds | 1 corner | 1 diag pair | 2 diag pairs | 1 staircase | 2 staircases | corner + diag |
|---|---|---|---|---|---|---|---|---|
| Wall/door cells | 114 | 122 | 217 | 185 | 240 | 186 | 242 | 256 |
| Complete rooms | 2.1 | 3.2 | 5.5 | 4.0 | 6.2 | 4.1 | 7.2 | 7.2 |
| Median room size | 24 | 20 | 16 | 15 | 15 | 11 | 7 | 16 |
| Rooms with a door | 95% | 81% | 82% | 71% | 80% | 56% | 54% | 79% |
| Doors set in a wall line | 87% | 84% | 86% | 86% | 84% | 75% | 70% | 83% |
| Turns (share of wall cells) | 10.7% | 8.9% | 7.4% | 8.3% | 9.0% | 12.8% | 13.2% | 7.4% |
| Diagonal-only joints per square | 0.9 | 1.4 | 1.6 | 1.9 | 3.0 | 3.4 | 4.6 | 2.1 |

- A diagonal pair (two walls touching corner to corner) is mostly completed as a **cut corner**, giving octagon-like rooms, at about
  the same quality as corner seeds.
- A **staircase** (a 4-connected diagonal) gives visibly curved and round walls, more turns than real data, but worse rooms: smaller
  rooms, many without a door, and some scribbled wall fragments.
- All shapes raise the number of walls touching only corner to corner. RimWorld rooms connect through cell sides, so such a joint does
  not seal a room. The repair pass has to fill one cell of each joint.

Plan for the C# side (agreed):
- no ring: only real obstacles fixed
- best crop when there are no seeds
- seed count and shape in the menu
- repair pass: close rooms cut by the edge, turn one-cell gaps that close a room into doors, add a door to doorless rooms, fill
  diagonal joints, remove stray walls, fill 1-2 cell pockets
- log vanilla's side in the same format, to measure its density

## 2026-10-28: generation and repair in C#, seeds in the dev menu

Code (no game dependencies, so `tools/csharp_check` tests it outside the game):
- `Source/LayoutGenerator.cs` changes how a layout is generated:
  - no empty ring: only real obstacles (water, marsh, rock, mountain roof, off-map) are fixed as empty
  - seeds are placed in the middle 24x24
  - the 36x36 square is the crop of the 48x48 window that cuts the fewest walls, while still containing every seed, so it can shift
    up to 6 cells
  - bridges are ruled out while sampling (logit bias -30), since on dry ground they were floor blobs
- `Source/LayoutRepair.cs` runs these steps in order:
  1. close rooms cut by the crop edge, or clear them when only a sliver is inside
  2. fill corner-to-corner joints
  3. remove stray walls
  4. turn one-cell gaps that would close off a room into doors
  5. fill sealed 1-2 cell pockets
  6. add doors until every room can be reached from outside (preferring outside, then a reachable room)
- `Source/LayoutGrid.cs`: regions, `LayoutStats` (the train/metrics.py measures for one grid), ASCII rendering.
- Dev menu: faction, then seed count 0-4, then shape (corner, diagonal pair, staircase, mixed). The compare tool reads vanilla's
  side back from the map and logs both sides' measures and ASCII side by side.
- Roofs go on enclosed rooms up to 150 cells. Fenced pens stay open.

Repair test outside the game (`LayoutCheck . repair 16`, 16 layouts per setting):

| Setting | Rooms, model then after repair | Rooms with a door before repair | Unreachable after |
|---|---|---|---|
| 0 seeds | 3.4 → 5.4 | 74% | 0 |
| 2 corners | 11.4 → 14.6 | 73% | 3 rooms in 16 layouts, behind double walls |
| 2 diagonal pairs | 7.6 → 12.1 | 76% | 0 |
| 1 staircase | 6.7 → 8.8 | 72% | 0 |
| 2 mixed | 9.7 → 14.0 | 81% | 0 |

No stray walls or corner-to-corner joints remain in any of the 80 layouts.

In game (25 compares, built layouts, averages):

| | Vanilla (n=7) | 0 seeds (3) | 1 corner (4) | 1 staircase (5) | 3 mixed (7) | 3 diagonal (6) |
|---|---|---|---|---|---|---|
| Wall-like cells | 155 | 80 | 124 | 71 | 242 | 273 |
| Floor cells | 268 | 50 | 239 | 49 | 259 | 431 |
| Rooms | 7.0 | 3.7 | 6.5 | 1.4 | 12.6 | 12.5 |

- Vanilla settlements average about 7 rooms per 36x36 square. One corner seed matches that density, and 3 seeds are about double.
- The user's view: 2 mixed (corner + diagonal) seeds give the most interesting shapes.
- Obstacles: the model builds around water well, but it also keeps away from mountains. Players hug mountains and dig into them.
  Cause: obstacles are fixed as *empty*, so the model sees open ground, not rock, and the training set excluded mountain-adjacent
  buildings.
- Staircase seeds were the weakest in game: few rooms, little floor.

## 2026-10-28: step A, mountains without retraining (`train/mountain_test.py`)

Synthetic mountains (smoothed noise plus a slope, covering 20-45% of the window from one side), 32 windows, measured on the centred
36x36 square with the remaining rock counted as wall. Ways of giving rock to the model:
- **empty**: what the game does now
- **wall**: all rock fixed as wall
- **dig2**: as wall, but the outer 2 cells of rock are left free; built cells there count as mined out
- **shell**: only rock's outermost cells fixed as wall, the inside as empty

Files: `docs/history/v1_mdm/mountain/`.

| 2 mixed seeds | empty | wall | dig2 | shell |
|---|---|---|---|---|
| Built wall-like cells | 71 | 260 | 287 | 175 |
| Floor | 3.5% | 44% | 50% | 16% |
| Complete rooms | 2.9 | 8.2 | 8.8 | 5.9 |
| Rooms with a door (before repair) | 65% | 15% | 18% | 22% |
| Open cells along the rock face that are built on | 1% | 97% | 94% | 73% |
| Rooms using rock as part of their outline | 0.03 | 0.75 | 0.78 | 2.1 |
| Rock cells mined out | | | 166 | |

- **empty**: the model treats rock like the empty ring. It builds almost nothing near mountains, which is why it stays away from them in
  game.
- **wall** and **dig2**: a mass of fixed wall reads as "inside a huge building". The model paves half the square, thickens the rock face
  with a 1-3 cell wall band, and leaves most rooms without a door. In dig2 it "digs" every freed cell, but only to make that band. This is
  not digging in any useful sense.
- **shell** is the only variant that hugs rock in a plausible way: rooms use the rock face as a wall (2 per square), with moderate floor.
  The weak points are doorless rooms (the repair pass handles those), rooms following the rock's curves, and an occasional cluster of
  scribbled walls.
- Conclusion: without retraining, the shell gives hugging but no digging. Real digging needs a model that has seen mountain bases.

## 2026-10-28: rock shell in game (`GenerateOptions.rockShell`, dev action "Toggle rock shell")

C# version of the shell variant. Natural rock is kept separate from other obstacles:
- its outermost cells are fixed as wall and its inside as empty
- after sampling it stays rock (`Categories.Rock`), which repair treats as a permanent wall
- a door goes through rock only as a last resort (mined)

Water, marsh and cave floor stay fixed as empty.

Outside the game (`LayoutCheck . rock 10`, smooth wavy cliff, 2 mixed seeds), shell off vs on:
- built wall cells: 54 vs 103
- rooms: 2.0 vs 1.7
- rooms using rock: 0 vs 0.3

In game: 8 compares near mountains, 2 mixed seeds, shell on. After repair, against vanilla's ~155 wall-like cells and ~7 rooms:

| | Mean | Range |
|---|---|---|
| Built wall-like cells | 285 | 103-419 |
| Floor cells | 43 | 0-185 (0-9 in 6 of 8) |
| Rooms | 9.3 | 3-19 |
| Rooms with no door before repair | 76% | |
| Rooms still unreachable after repair | 1.0 | 0-3 |

- The user's view: layouts do hug the mountains, but almost all of them have large walled-in spaces against the rock. Overall the
  result feels not much better than noise.
- The ASCII logs show the same failure as the Python wall and dig variants, only milder. Next to rock, the model draws solid
  masses of wall several cells thick, with almost no floor, and the rooms between them have no doors. The same masses appear beside
  large non-rock obstacles: one layout had no rock, only 45 water/marsh cells.
- Reading: big fixed regions of any kind are far outside what v1 saw in training, because it never saw a constraint during training
  and never saw a building next to a mountain. Repair can make rooms reachable, but it can't turn a wall mass into a building.
- Conclusion: prompting tricks have reached their limit. Mountains and obstacles need a model trained with them as an input (v2).

## 2026-10-28: more data (set `v1_48_more`)

The RealRuins cache had grown to 8,501 blueprints, mostly downloaded on 2026-10-28. All new files (including the per-world subfolders) were
copied into `data/raw`, which now holds 8,508 unique blueprints (v1 used 3,533). `extract.py` processed 4,975 new files with 0 errors.
`windows.py configs/v1.json --out data/sets/v1_48_more` uses the same config as v1; the v1 set is left as trained.

| | v1 (`v1_48`) | all (`v1_48_more`) |
|---|---|---|
| Blueprints | 3,533 | 8,508 |
| Colonies | 1,249 | 3,065 |
| Truncated uploads | 2,285 (65%) | 5,444 (64%) |
| Free-standing buildings | 8,607 | 20,816 |
| Mountain-adjacent buildings | 3,825 | 9,260 |
| Unroofed enclosures | 1,131 | 2,803 |
| Windows (train / val) | 6,489 (6,088 / 401) | 15,704 (14,848 / 856) |
| Snapshots that give a window | 1,439 | 3,414 |
| Masked fraction | 0.10 | 0.10 |

The new data has the same make-up as the old: the same truncation rate, ratio of building classes, and windows per blueprint.
Mountain-adjacent buildings, about a third of all buildings, are still excluded by the v1 config (windows with any mountain
cell are dropped).

## 2026-10-28: v1 baseline on more data (`runs/v1_more_mdm`, `configs/train_v1_more.json`)

This is the v1 config unchanged (16.5M-parameter MaskGIT U-Net, 150 epochs, batch 64), trained on `v1_48_more`. It ran at about 37 s per
epoch (v1: about 20 s), about 1.6 hours in total. The WSL exit hang occurred again and the runner cleared it.

- Best validation loss: 0.3826 (v1: 0.407). The validation sets differ (856 vs 401 windows), so this is only a rough comparison.
- Without the logit adjustment, generated samples are as poor as v1's (62 rooms per window, median room size 2). The class-weight
  problem is unchanged.

Both checkpoints sampled with the same seed, 64 samples each (`train/sweep_adjust.sh`). Real-data reference:

| Metric | Real | v1 a=0.5 | **more a=0.5** | v1 a=0.6 | **more a=0.6** | v1 a=0.7 | **more a=0.7** |
|---|---|---|---|---|---|---|---|
| Empty cells | 89.5% | 73.0% | 74.5% | 78.9% | 80.5% | 83.6% | 84.7% |
| Wall cells | 3.5% | 6.6% | 7.0% | 5.1% | 5.1% | 3.9% | 4.1% |
| Floor cells | 6.6% | 16.7% | 16.0% | 14.4% | 13.1% | 11.5% | 10.4% |
| Rooms per window | 2.74 | 4.6 | 5.4 | 3.3 | 3.1 | 2.1 | 2.5 |
| Median room size | 25 | 16.5 | 15 | 19 | 20 | 20 | 20 |
| Doors set in a wall line | 90% | 86% | 87% | 87% | 87% | 83% | 88% |
| Rooms with a door | 95% | 81% | 82% | 77% | **86%** | 75% | **87%** |
| Floor inside rooms | 53% | 24% | 30% | 21% | 25% | 20% | 22% |
| Stray wall pieces | 1.4% | 0.9% | 1.1% | 0.9% | 1.1% | 1.3% | 0.8% |

(The real-data columns from the two sets differ by less than 0.02 on every metric except floor inside rooms: 48% vs 53%.)

Files: `docs/history/v1_more_mdm/` (sample sheet at a=0.6, metrics at a=0.5/0.6/0.7, training log and config).

- With 2.4 times the data, the model improves moderately but not fundamentally. Rooms are more often closed with a door (77% → 86% at
  a=0.6), there is slightly less floor outside rooms, and a=0.7 now keeps the room count near real (2.5) without losing doors.
- The sample sheet looks much like v1's. The same faults remain: floor blobs outside buildings, bridge patches, occasional
  loose wall lines.
- The faults that look like noise in game come mostly from things more data doesn't change:
  - the class weighting, which the logit bias corrects only coarsely
  - large fixed regions the model never saw in training (see the rock shell entry)
- Next: v2 changes to the training itself rather than more of the same data:
  - rock/obstacle input channel
  - mountain-adjacent buildings
  - random fixed regions during training
  - milder class weights, so the sampling bias matters less

## 2026-10-29: v2 data set (`configs/v2.json`, `data/sets/v2_48`; plan in `docs/PLAN_v2.md`)

New in the exporter:
- `tools/rooms.py`: room types from room contents. The rules live in the config's `rooms` section.
- `categories.py`: `thing_patterns`, used for cover, traps and turrets.
- `windows.py`:
  - a rock grid, saved per window as `cond` (1 = rock)
  - mountain windows allowed (`allow_mountain`, `max_rock_fraction` 0.6); rock is left out of the loss
- `report.py`: shows rock as `R`.

Rock is taken to be mountain-mask cells that are roofed, unbuilt and not enclosed by walls. Rock also bounds rooms, so rooms dug
into a mountain are typed too.

28 categories:
- base: empty, wall, door, floor, fence, wall_mounted
- defences: embrasure, cover, trap, turret
- 18 room types

Bridges are dropped; bridge terrain counts as floor.

| | v1_48_more | **v2_48** |
|---|---|---|
| Windows (train / val) | 15,704 (14,848 / 856) | **37,661** (35,689 / 1,972) |
| Windows with rock | 0 | 22,264 |
| Snapshots that give a window | 3,414 | 5,494 |
| Rock cells | | 8.5% |
| Masked cells (cut buildings) | 10% | 8.5% |

Rooms typed over all snapshots: 121,771. The shares are close to the probe's:

| Type | Share |
|---|---|
| bedroom | 34.6% |
| room_empty | 16.0% |
| room_untyped | 8.7% |
| storage | 6.3% |
| barracks | 4.7% |
| kitchen | 3.9% |
| workshop | 3.4% |
| utility | 3.3% |
| bathroom | 3.2% |
| dining | 2.6% |
| freezer | 2.2% |
| hospital | 2.3% |
| rec | 2.1% |
| lab | 1.9% |
| barn | 1.7% |
| temple | 1.1% |
| crypt | 1.0% |
| nursery | 0.6% |

- `room_empty` is twice the probe's share, for two reasons:
  - rooms holding only ignored things (lamps, chairs, conduits) now count as empty, not untyped
  - rock now closes about 5,700 extra rooms in mountains, many of them small and unfurnished
- Freezers include wall-mounted cooler mods (`Cooler_Over`); passive coolers are excluded.
- Cell shares in train:
  - empty 79.4%, floor 7.5%, wall 5.2%, door 0.3%
  - cover 0.16%, trap 0.07%, turret 0.03%, embrasure, fence and wall_mounted about 0.045% each
  - all room types together 7.2%, with bedroom the largest at 1.4%
- ASCII spot checks look right: rows of bedrooms, rooms against rock, dining hall with kitchen and freezer, turret rows behind embrasures.
