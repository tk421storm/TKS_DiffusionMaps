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
