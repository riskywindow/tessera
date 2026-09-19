#!/usr/bin/env bash
# Everything that happens inside the Modal image build, after the repo has been
# copied in. Kept in the repo (rather than inlined into modal_gpu.py) so that
# the image build is one auditable script with one set of exit-code rules, and
# so infra/versions.lock stays the single source of truth: this script reads it
# instead of repeating any version.
#
# Run by infra/modal_gpu.py as a cached image layer. No GPU is attached, which
# is the point: the expensive seconds are GPU seconds, so the compiles happen
# here and the GPU functions only run binaries.
#
# Environment (all defaulted, all overridable):
#   TESSERA_REPO        repo root inside the image          (/opt/tessera)
#   TESSERA_DEPS        fetched deps                        (/opt/tessera-deps)
#   TESSERA_BUILD_DIR   build trees                         (/opt/tessera-build)
#   TESSERA_BIN         built sample binaries               (/opt/tessera-bin)
#   TESSERA_IMAGE_DIR   where the image manifest is written (/opt/tessera-image)
#   TESSERA_BUILD_JOBS  -j for the in-image builds          (4)
set -euo pipefail

TESSERA_REPO="${TESSERA_REPO:-/opt/tessera}"
TESSERA_DEPS="${TESSERA_DEPS:-/opt/tessera-deps}"
TESSERA_BUILD_DIR="${TESSERA_BUILD_DIR:-/opt/tessera-build}"
TESSERA_BIN="${TESSERA_BIN:-/opt/tessera-bin}"
TESSERA_IMAGE_DIR="${TESSERA_IMAGE_DIR:-/opt/tessera-image}"
TESSERA_BUILD_JOBS="${TESSERA_BUILD_JOBS:-4}"
export TESSERA_DEPS

VERSIONS="${TESSERA_REPO}/infra/versions.lock"

die() {
  echo "image_build: $*" >&2
  exit 1
}

[[ -f "$VERSIONS" ]] || die "versions.lock not found at $VERSIONS"

# Read one dotted key out of versions.lock. tomllib ships with python >= 3.11,
# which is the pinned interpreter.
lock() {
  python3 - "$VERSIONS" "$1" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as fh:
    doc = tomllib.load(fh)
node = doc
for part in sys.argv[2].split("."):
    node = node[part]
if isinstance(node, list):
    print(" ".join(str(x) for x in node))
else:
    print(node)
PY
}

gtest_version="$(lock test_deps.googletest.version)"
gtest_sha256="$(lock test_deps.googletest.sha256)"
gtest_url="$(lock test_deps.googletest.url)"
clang_major="$(lock compilers.clang.major)"
gcc_major="$(lock compilers.gcc.major)"

CC_GCC="gcc-${gcc_major}"
CXX_GCC="g++-${gcc_major}"
CC_CLANG="clang-${clang_major}"
CXX_CLANG="clang++-${clang_major}"

for tool in "$CC_GCC" "$CXX_GCC" "$CC_CLANG" "$CXX_CLANG" cmake ninja nvcc python3 curl; do
  command -v "$tool" >/dev/null 2>&1 || die "required tool missing from image: $tool"
done

# tomllib (used to read versions.lock) is stdlib only from 3.11. The image gets
# its python from modal's add_python, which lands in /usr/local/bin ahead of any
# distro python; if that ordering ever changed, this would be a silent downgrade
# rather than an error, so it is checked.
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' ||
  die "python3 on PATH is $(python3 -V 2>&1), but >= 3.11 is required for tomllib"

mkdir -p "$TESSERA_DEPS" "$TESSERA_BUILD_DIR" "$TESSERA_BIN" "$TESSERA_IMAGE_DIR"

# --- 1. CUDA driver headers + driver stub ----------------------------------
# The base image already carries a CUDA toolkit, but the repo's contract is
# that cuda.h comes from the SHA-256-verified fetch script, so the image and
# the dev host compile against the same bytes.
echo "image_build: fetching CUDA headers"
"${TESSERA_REPO}/scripts/fetch_cuda_headers.sh" x86_64

# --- 2. GoogleTest tarball, hash-checked here and again by CMake ------------
gtest_tarball="${TESSERA_DEPS}/googletest-${gtest_version}.tar.gz"
if [[ ! -f "$gtest_tarball" ]]; then
  echo "image_build: fetching googletest ${gtest_version}"
  curl -fsSL "$gtest_url" -o "${gtest_tarball}.part"
  echo "${gtest_sha256}  ${gtest_tarball}.part" | sha256sum -c --quiet -
  mv "${gtest_tarball}.part" "$gtest_tarball"
fi

# --- 3. Build the repo with BOTH minimum compilers --------------------------
# This is the brief's minimum-compiler claim: clang-18 and g++-13 must build the
# tree with -Wall -Wextra -Werror. The dev host runs gcc-15/clang-20, so this
# image is the only place that claim is actually tested.
build_with() {
  local tag="$1" cc="$2" cxx="$3"
  local dir="${TESSERA_BUILD_DIR}/${tag}"
  echo "image_build: configuring ${tag} (${cc} / ${cxx})"
  cmake -S "$TESSERA_REPO" -B "$dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER="$cc" \
    -DCMAKE_CXX_COMPILER="$cxx" \
    -DTESSERA_WERROR=ON \
    -DTESSERA_BUILD_TESTS=ON
  echo "image_build: building ${tag}"
  cmake --build "$dir" -j"$TESSERA_BUILD_JOBS"
}

build_with "gcc-${gcc_major}" "$CC_GCC" "$CXX_GCC"
build_with "clang-${clang_major}" "$CC_CLANG" "$CXX_CLANG"

# --- 4. The CUDA sample, fat-binary'd for the GPUs TESSERA_GPU can select ----
echo "image_build: building bench/samples/vector_add.cu"
"${TESSERA_REPO}/bench/samples/build_vector_add.sh" "${TESSERA_BIN}/vector_add"

# --- 5. Record what actually got installed ---------------------------------
# Measured, not declared: every version below is read back out of the image.
# Results JSON embeds this file so no number in a result is a recollection.
echo "image_build: writing ${TESSERA_IMAGE_DIR}/manifest.json"
python3 - "$VERSIONS" "${TESSERA_IMAGE_DIR}/manifest.json" \
  "$CC_GCC" "$CXX_GCC" "$CC_CLANG" "$CXX_CLANG" \
  "$TESSERA_DEPS" "$TESSERA_BUILD_DIR" "$TESSERA_BIN" <<'PY'
import datetime
import json
import os
import re
import subprocess
import sys
import tomllib

versions_path, out_path = sys.argv[1], sys.argv[2]
cc_gcc, cxx_gcc, cc_clang, cxx_clang = sys.argv[3:7]
deps_dir, build_dir, bin_dir = sys.argv[7:10]

with open(versions_path, "rb") as fh:
    lock = tomllib.load(fh)


def run(cmd):
    """Capture a version string. A missing or failing tool is fatal: an
    unrecorded toolchain version would make every downstream number
    unattributable (I-1)."""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return (proc.stdout + proc.stderr).strip()


def which(binary):
    return run(["bash", "-c", f"command -v {binary}"])


def first_line(text):
    return text.splitlines()[0].strip() if text else ""


manifest = {
    "kind": "tessera_image_manifest",
    "built_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    ),
    "base_image": lock["base_image"]["reference"],
    "base_image_digest": lock["base_image"]["digest"],
    "versions_lock_verified_on": lock["verified_on"],
    "os_release": next(
        (
            ln.split("=", 1)[1].strip().strip('"')
            for ln in open("/etc/os-release").read().splitlines()
            if ln.startswith("PRETTY_NAME=")
        ),
        "unknown",
    ),
    "toolchain": {
        "gcc": {
            "binary": cxx_gcc,
            "path": which(cxx_gcc),
            "version_line": first_line(run([cxx_gcc, "--version"])),
            "dumpfullversion": run([cc_gcc, "-dumpfullversion"]),
        },
        "clang": {
            "binary": cxx_clang,
            "path": which(cxx_clang),
            "version_line": first_line(run([cxx_clang, "--version"])),
        },
        "cmake": {
            "path": which("cmake"),
            "version_line": first_line(run(["cmake", "--version"])),
        },
        "ninja": {
            "path": which("ninja"),
            "version": first_line(run(["ninja", "--version"])),
        },
        "nvcc": {
            "path": which("nvcc"),
            "version": run(["nvcc", "--version"]).splitlines()[-1].strip(),
        },
        "python": sys.version.split()[0],
    },
    "cuda_headers": {},
    "python_packages": {},
    "paths": {
        "repo": os.environ.get("TESSERA_REPO", "/opt/tessera"),
        "deps": deps_dir,
        "build_dir": build_dir,
        "bin_dir": bin_dir,
        "build_trees": sorted(
            d for d in os.listdir(build_dir) if os.path.isdir(os.path.join(build_dir, d))
        ),
        "vector_add": os.path.join(bin_dir, "vector_add"),
    },
}

header = os.path.join(deps_dir, "cuda-12.6", "x86_64-linux", "include", "cuda.h")
with open(header) as fh:
    match = re.search(r"^#define CUDA_VERSION\s+(\d+)", fh.read(), re.M)
if not match:
    raise SystemExit(f"image_build: CUDA_VERSION not found in {header}")
manifest["cuda_headers"] = {
    "path": header,
    "cuda_version": int(match.group(1)),
    "package_version": lock["cuda"]["headers"]["package_version"],
}

import torch  # noqa: E402  (imported late; it is the slowest import here)

manifest["python_packages"] = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_git_version": torch.version.git_version,
    "pinned_torch": lock["python_packages"]["torch"]["version"],
    "pinned_vllm_for_m1": lock["python_packages"]["vllm"]["version"],
}
if manifest["python_packages"]["torch"].split("+")[0] != manifest["python_packages"]["pinned_torch"]:
    raise SystemExit(
        "image_build: installed torch {} does not match the pin {}".format(
            manifest["python_packages"]["torch"], manifest["python_packages"]["pinned_torch"]
        )
    )

with open(out_path, "w") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

echo "image_build: done"
