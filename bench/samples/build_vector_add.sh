#!/usr/bin/env bash
# Build bench/samples/vector_add.cu with nvcc into a fat binary that covers the
# GPUs TESSERA_GPU can select, so one image works for L4 (sm_89), A100 (sm_80),
# A10G/A10 (sm_86) and H100 (sm_90) without a rebuild.
#
# Usage: build_vector_add.sh <output-binary-path>
#
# This deliberately does not go through CMake: the repo's CMake project has no
# CUDA language enabled (the dev host has no nvcc) and adding one would break
# every local preset. The sample is a standalone nvcc build run at image-build
# time, which is also when GPU seconds are not being spent.
set -euo pipefail

usage() {
  echo "usage: $0 <output-binary-path>" >&2
  exit 64
}

[[ $# -eq 1 ]] || usage
out="$1"

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
src="${here}/vector_add.cu"
[[ -f "$src" ]] || {
  echo "build_vector_add: source not found: $src" >&2
  exit 1
}

nvcc_bin="${NVCC:-nvcc}"
command -v "$nvcc_bin" >/dev/null 2>&1 || {
  echo "build_vector_add: nvcc not found (set NVCC=/path/to/nvcc)" >&2
  exit 1
}

# Architectures come from infra/versions.lock [cuda.gencode]. They are repeated
# here as literals because nvcc runs in the image build with no TOML parser on
# hand; scripts/check_versions_lock.py (WP-0.7's job if it wants one) can assert
# the two lists agree.
gencode_args=(
  -gencode "arch=compute_80,code=sm_80"
  -gencode "arch=compute_86,code=sm_86"
  -gencode "arch=compute_89,code=sm_89"
  -gencode "arch=compute_90,code=sm_90"
  # PTX for anything newer than sm_90, so a TESSERA_GPU we have not compiled
  # for JITs instead of failing with "no kernel image is available".
  -gencode "arch=compute_90,code=compute_90"
)

mkdir -p "$(dirname -- "$out")"

"$nvcc_bin" \
  -std=c++17 \
  -O2 \
  --generate-line-info \
  "${gencode_args[@]}" \
  -Xcompiler -Wall \
  -Xcompiler -Wextra \
  -o "$out" \
  "$src"

echo "build_vector_add: built $out"
"$nvcc_bin" --version | tail -n 2
