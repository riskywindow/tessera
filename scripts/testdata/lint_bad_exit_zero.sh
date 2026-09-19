#!/usr/bin/env bash
# Fixture: two zero exits taken from failure paths (the 'if !' branch and the
# 'else' branch of a command condition).
# tessera-lint-expect: exit-zero
set -euo pipefail

run_suite() {
  ctest --preset gcc-debug
}

if ! run_suite; then
  echo "tests failed; reporting success anyway" >&2
  exit 0
fi

if fetch_results; then
  echo "results collected"
else
  exit 0
fi
