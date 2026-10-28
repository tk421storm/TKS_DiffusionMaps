#!/usr/bin/env bash
# Compare logit-adjustment strengths on one checkpoint (CPU only, so it can run beside GPU training).
# Usage (inside WSL, project folder): bash train/sweep_adjust.sh <ckpt> [n] [strengths...]
ckpt="$1"; n="${2:-32}"; shift 2 || true
strengths=("${@:-0 0.5 1.0}")
[ $# -eq 0 ] && strengths=(0 0.5 1.0)
for a in "${strengths[@]}"; do
  echo "== adjust $a"
  HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ~/venvs/dm/bin/python -u train/sample.py "$ckpt" --n "$n" --adjust "$a" 2>&1 \
    | grep -E "metric|freq_wall|freq_floor|freq_empty|stray|door_valid|rooms_per|room_area|rooms_with|floor_encl|wrote|Error|Trace"
done
