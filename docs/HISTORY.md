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
