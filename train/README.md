# v1 layout model: masked discrete diffusion

This folder trains a model that generates 48x48 category grids (empty / wall / door / floor / bridge /
fence / wall_mounted) from the window sets made by `tools/windows.py`. The approach is MaskGIT-style
masked discrete diffusion.

| File | What it does |
|---|---|
| `data.py` | Loads a window set onto the GPU once, applies the 8 rotations and flips on the GPU, and computes class weights. |
| `model.py` | U-Net, conditioned on the masked fraction. Attention runs at 12x12. Uses only ONNX-friendly ops. |
| `diffusion.py` | Masked training loss (class-weighted, skips `loss_mask`) and the iterative sampler (inpainting via `fixed`, `logit_bias`). |
| `metrics.py` | Structural checks: stray walls, door and wall-mounted placement, rooms, room size, floor inside rooms. |
| `render.py` | Text and PNG sheets. |
| `train.py` | Training loop. Saves `runs/<name>/` with checkpoints, `log.jsonl`, and samples and metrics every `eval_every` epochs. |
| `sample.py` | Samples from a checkpoint. `--adjust` sets the logit-adjustment strength. |
| `sweep_adjust.sh` | Compares adjustment strengths on CPU. |
| `export_onnx.py` | Exports to ONNX. Writes `model.onnx` and `model_meta.json`, then runs three checks: logits match PyTorch, deterministic sampling matches the numpy + ORT sampler, and CPU timing. |

## Running (from Git Bash, project folder)

GPU jobs go through `tools/run_wsl_gpu.sh`. On Windows 10 + ROCDXG the process hangs in the kernel after it
finishes. The runner waits for the job to print `done`, then runs `wsl --shutdown`. Everything is saved before that point.
Quote `~` paths, or Git Bash expands them first.

```
bash tools/run_wsl_gpu.sh runs/v1_mdm.log '~/venvs/dm/bin/python' -u train/train.py configs/train_v1.json
```

CPU-only jobs (sampling, export) can run directly in WSL, even next to GPU training, with the GPU hidden:

```
HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ~/venvs/dm/bin/python -u train/sample.py runs/v1_mdm/ckpt_best.pt --adjust 0.6
HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ~/venvs/dm/bin/python -u train/export_onnx.py runs/v1_mdm/ckpt_best.pt --adjust 0.6
```

## Logit adjustment

Class-weighted training, with weights from frequency^-0.5 capped at 20, makes the model over-predict rare categories.
On the raw model this means far too many walls and floors: 25 rooms per window against 2.75 in real data.

At sampling time, `logit_bias = -adjust * log(class_weight)` corrects this:
- 1.0 undoes the weighting fully, which gives too sparse a layout.
- 0.6 gave the best match for v1.

`export_onnx.py --adjust` stores the bias in `model_meta.json` for the in-game sampler.

## v1 results (2026-10-27, `configs/train_v1.json`)

- Model: 16.5M parameters, 150 epochs, about 20 s per epoch on the RX 7900 XTX (about 1 hour total including evaluation).
- Best validation loss: 0.407. At 50% masked, 98% of hidden cells are predicted correctly.

64 samples, 16 steps, adjust 0.6, compared with 1,024 clean real windows:

| Metric | Real | Generated |
|---|---|---|
| Empty / wall / floor cells | 89% / 3.4% / 6.8% | 79% / 5.1% / 14% |
| Stray wall pieces | 1.4% | 0.9% |
| Doors set in a wall line | 88% | 87% |
| Rooms per window | 2.75 | 3.3 |
| Median room size | 25 cells | 19 cells |
| Rooms with a door | 97% | 77% |
| Floor inside enclosed rooms | 48% | 21% |

ONNX check:
- PyTorch vs ONNX Runtime logits: max difference 1.2e-5.
- Deterministic sampling: 100% identical cells.
- CPU time, batch 1, 16 steps: 0.41 s with all threads, 1.33 s with one thread.
- `model.onnx` is 63 MB.

Known weaknesses:
- Too much outdoor floor: large paved areas.
- Rooms without doors.
- Some walls that don't close.
- Bridges appear as blobs, because the model has no idea where water is. That needs terrain conditioning.
- Wall-mounted objects are almost never generated.

Full v1 analysis, adjust sweep and saved run files: `../docs/HISTORY.md` and `../docs/history/v1_mdm/`.
