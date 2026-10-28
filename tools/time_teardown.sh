#!/usr/bin/env bash
# Run a command and report how long the process lingers after it prints "done" (WSL GPU teardown).
# Usage: bash tools/time_teardown.sh <log file> <command...>
log="$1"; shift
start=$(date +%s)
"$@" > "$log" 2>&1 &
pid=$!
done_at=""
while kill -0 "$pid" 2>/dev/null; do
  if [ -z "$done_at" ] && grep -q "^done" "$log"; then done_at=$(date +%s); echo "done printed at +$((done_at - start))s"; fi
  sleep 2
done
end=$(date +%s)
[ -z "$done_at" ] && grep -q "^done" "$log" && done_at=$end
echo "process gone at +$((end - start))s; teardown took $((end - ${done_at:-$end}))s"
