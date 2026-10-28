#!/usr/bin/env bash
# Run a GPU job in WSL from Git Bash and clear the Windows 10 + ROCDXG exit hang afterwards.
#
# The job's stdout goes to <log> (a Windows path). Once the job prints a line starting with "done"
# (or a Traceback), the process gets <grace> seconds to exit; if it is still stuck in the GPU
# teardown (vmbus_teardown_gpadl), WSL is shut down. All outputs are written before "done".
#
# Usage (Git Bash, from the DiffusionMaps folder):
#   bash tools/run_wsl_gpu.sh <log file> <command run inside WSL from the project folder...>
# Example:
#   bash tools/run_wsl_gpu.sh runs/v1_mdm.log '~/venvs/dm/bin/python' -u train/train.py configs/train_v1.json
# Quote ~ paths: unquoted, Git Bash expands them to the Windows home before WSL sees them.
set -u
log="$1"; shift
grace=30
proj_wsl="/mnt/f/SteamLibrary/steamapps/common/RimWorld/Mods/DiffusionMaps"
mkdir -p "$(dirname "$log")"
: > "$log"
wsl --shutdown; sleep 3
# Inside WSL: run the job, appending its output to the log on the Windows drive.
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd "$proj_wsl" -- bash -c \
  "export HSA_ENABLE_DXG_DETECTION=1; $* >> '$proj_wsl/$log' 2>&1; echo \"[runner] job exited \$?\" >> '$proj_wsl/$log'" &
wpid=$!
finished=""
while kill -0 "$wpid" 2>/dev/null; do
  if [ -z "$finished" ] && grep -qE "^done|^Traceback" "$log" 2>/dev/null; then
    finished=$(date +%s)
  fi
  if [ -n "$finished" ] && [ $(( $(date +%s) - finished )) -ge $grace ]; then
    echo "[runner] job finished but did not exit within ${grace}s (WSL GPU teardown hang); shutting down WSL" >> "$log"
    wsl --shutdown
    break
  fi
  sleep 5
done
wait "$wpid" 2>/dev/null
echo "[runner] finished at $(date +%T)" >> "$log"
