#!/usr/bin/env bash
# Install AMD's ROCm 7.2.1 PyTorch wheels (Python 3.10, Ubuntu 22.04) into a venv inside WSL.
# Prerequisites (see tools/README.md): Adrenalin 26.2.2+ on Windows, hsa-rocr 7.2.x, rocdxg-roct,
# rocminfo showing gfx1100 with HSA_ENABLE_DXG_DETECTION=1.
# Usage: bash tools/setup_wsl_torch.sh [venv dir, default ~/venvs/dm]
set -euo pipefail

VENV="${1:-$HOME/venvs/dm}"
WHEELS="$HOME/wheels/rocm-7.2.1"
BASE="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1"
FILES=(
  "torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp310-cp310-linux_x86_64.whl"
  "torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp310-cp310-linux_x86_64.whl"
  "triton-3.5.1+rocm7.2.1.gita272dfa8-cp310-cp310-linux_x86_64.whl"
)

[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
mkdir -p "$WHEELS"
for f in "${FILES[@]}"; do
  if [ ! -s "$WHEELS/$f" ]; then
    echo "downloading $f"
    wget -q -O "$WHEELS/$f.part" "$BASE/${f//+/%2B}"
    mv "$WHEELS/$f.part" "$WHEELS/$f"
  fi
done
ls -lh "$WHEELS"

"$VENV/bin/pip" install -q --upgrade pip wheel
"$VENV/bin/pip" install -q "${FILES[@]/#/$WHEELS/}" numpy scipy
"$VENV/bin/pip" list 2>/dev/null | grep -iE "^(torch|torchvision|triton|numpy|scipy) "
echo INSTALL_DONE
