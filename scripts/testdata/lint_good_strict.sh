#!/usr/bin/env bash
# Fixture: the shape every Tessera script should have. Errexit, pipefail, and
# every failure either handled or propagated.
set -euo pipefail

count_driver_symbols() {
  local stub="$1"
  nm -D --defined-only "$stub" | grep -c ' T cu'
}

if [[ $# -eq 0 ]]; then
  echo "usage: ${0##*/} <libcuda.so>" >&2
  exit 2
fi

if ! count_driver_symbols "$1"; then
  echo "could not read the export list of $1" >&2
  exit 1
fi
