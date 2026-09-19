#!/usr/bin/env bash
# Fixture: a zero exit that is a success path, not a masked failure.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: lint_good_exit_paths.sh [--help]
USAGE
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

echo "doing the work"
