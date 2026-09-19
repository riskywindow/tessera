#!/usr/bin/env bash
# Fixture: a pipeline whose first stage can fail unnoticed.
# tessera-lint-expect: missing-pipefail
set -eu

nm -D --defined-only /usr/local/cuda/lib64/stubs/libcuda.so | grep -c ' T cu'
