#!/usr/bin/env bash
# Dry-run (no root, nothing changed) of moving the WSL ROCm packages from the 7.2 to the 7.2.1 apt repo.
# Uses a private copy of apt's lists with the ROCm sources swapped, then simulates the upgrade.
set -euo pipefail
VER="${1:-7.2.1}"
T=$(mktemp -d)
mkdir -p "$T/lists/partial" "$T/etc"
# Keep Ubuntu's lists, drop repo.radeon.com ones (they will be re-fetched for $VER).
find /var/lib/apt/lists -maxdepth 1 -type f ! -name "*radeon*" ! -name lock -exec cp {} "$T/lists/" \;
sed -E "s#repo.radeon.com/rocm/apt/[0-9.]+#repo.radeon.com/rocm/apt/$VER#; s#repo.radeon.com/graphics/[0-9.]+/#repo.radeon.com/graphics/$VER/#" \
  /etc/apt/sources.list.d/rocm.list > "$T/etc/rocm.list"
grep -h "^deb" /etc/apt/sources.list /etc/apt/sources.list.d/amdgpu.list 2>/dev/null >> "$T/etc/rocm.list" || true
echo "== sources used:"; cat "$T/etc/rocm.list"
OPTS=(-o Dir::State::Lists="$T/lists" -o Dir::Etc::SourceList="$T/etc/rocm.list" -o Dir::Etc::SourceParts=/dev/null
      -o Debug::NoLocking=1 -o Dir::Cache="$T/cache" -o APT::Get::List-Cleanup=0)
mkdir -p "$T/cache/archives/partial"
apt-get "${OPTS[@]}" update -qq 2>&1 | grep -vE "^W:|lock" || true
echo "== simulated full upgrade:"
apt-get "${OPTS[@]}" -s dist-upgrade 2>&1 | grep -E "^(Inst|Remv)|upgraded|E:" | sed -E 's/ \[[^]]*\]//; s/ \(([^ ]+) .*/ -> \1/'
rm -rf "$T"
