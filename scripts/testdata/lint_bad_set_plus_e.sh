#!/usr/bin/env bash
# Fixture: errexit switched back off, so everything below can fail silently.
# tessera-lint-expect: set-plus-e
set -euo pipefail

set +e
cmake --build --preset gcc-debug -j3
