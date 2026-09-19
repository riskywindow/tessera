#!/usr/bin/env bash
# Fetch the CUDA driver-API headers (cuda.h, cudaTypedefs.h) and the driver
# stub library (stubs/libcuda.so, used only as the authoritative export list)
# from NVIDIA's apt repository, verify SHA-256 against the repository's
# Packages index, and extract them outside the repo. NVIDIA headers are never
# vendored into the repo.
#
# Usage: scripts/fetch_cuda_headers.sh [x86_64|aarch64 ...]
#   Default: the host architecture. Pass both to support cross builds.
# Output: ${TESSERA_DEPS:-$HOME/.cache/tessera-deps}/cuda-<ver>/<target>/{include,lib/stubs}
#   where <target> is x86_64-linux or sbsa-linux (NVIDIA's naming).
set -euo pipefail

CUDA_SERIES="12-6"
CUDA_DOTTED="12.6"
CUDA_PKG_VERSION="12.6.77-1"
REPO_DISTRO="ubuntu2404"
DEPS_ROOT="${TESSERA_DEPS:-$HOME/.cache/tessera-deps}"
DEST_ROOT="${DEPS_ROOT}/cuda-${CUDA_DOTTED}"

die() { echo "fetch_cuda_headers: $*" >&2; exit 1; }

for tool in curl sha256sum gunzip tar; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
done

extract_deb() {
  local deb="$1" out="$2"
  mkdir -p "$out"
  if command -v dpkg-deb >/dev/null 2>&1; then
    dpkg-deb -x "$deb" "$out"
  else
    command -v ar >/dev/null 2>&1 || die "need dpkg-deb or ar to unpack $deb"
    local tmp
    tmp="$(mktemp -d)"
    (cd "$tmp" && ar x "$deb")
    local data
    data="$(find "$tmp" -maxdepth 1 -name 'data.tar.*' -print -quit)"
    [[ -n "$data" ]] || die "no data.tar.* in $deb"
    tar -xf "$data" -C "$out"
    rm -rf "$tmp"
  fi
}

fetch_arch() {
  local arch="$1" repo_arch deb_arch target
  case "$arch" in
    x86_64 | amd64) repo_arch="x86_64" deb_arch="amd64" target="x86_64-linux" ;;
    aarch64 | arm64) repo_arch="sbsa" deb_arch="arm64" target="sbsa-linux" ;;
    *) die "unsupported arch: $arch" ;;
  esac

  local dest="${DEST_ROOT}/${target}"
  local stamp="${dest}/.tessera-stamp"
  local want_stamp="${CUDA_PKG_VERSION}"
  if [[ -f "$stamp" ]] && [[ "$(cat "$stamp")" == "$want_stamp" ]] &&
    [[ -f "${dest}/include/cuda.h" ]] && [[ -f "${dest}/lib/stubs/libcuda.so" ]]; then
    echo "fetch_cuda_headers: ${target} already present at ${dest}"
    return 0
  fi

  local base="https://developer.download.nvidia.com/compute/cuda/repos/${REPO_DISTRO}/${repo_arch}"
  local work
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' RETURN

  curl -fsSL "${base}/Packages.gz" -o "${work}/Packages.gz"
  gunzip "${work}/Packages.gz"

  local pkg
  for pkg in "cuda-cudart-dev-${CUDA_SERIES}" "cuda-driver-dev-${CUDA_SERIES}"; do
    # Pull Filename and SHA256 for exactly this package+version from the index.
    local record filename sha
    record="$(awk -v p="$pkg" -v v="$CUDA_PKG_VERSION" '
      BEGIN { RS = ""; FS = "\n" }
      {
        name = ""; ver = "";
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^Package: /) name = substr($i, 10);
          if ($i ~ /^Version: /) ver = substr($i, 10);
        }
        if (name == p && ver == v) { print; exit }
      }' "${work}/Packages")"
    [[ -n "$record" ]] || die "${pkg}=${CUDA_PKG_VERSION} not found in ${base}/Packages.gz"
    filename="$(printf '%s\n' "$record" | sed -n 's/^Filename: //p')"
    sha="$(printf '%s\n' "$record" | sed -n 's/^SHA256: //p')"
    [[ -n "$filename" && -n "$sha" ]] || die "incomplete index record for ${pkg}"

    local deb="${work}/${pkg}_${deb_arch}.deb"
    curl -fsSL "${base}/${filename#./}" -o "$deb"
    echo "${sha}  ${deb}" | sha256sum -c --quiet - || die "SHA-256 mismatch for ${pkg}"
    extract_deb "$deb" "${work}/root"
    echo "fetch_cuda_headers: ${pkg} ${CUDA_PKG_VERSION} (${deb_arch}) sha256=${sha}"
  done

  local src="${work}/root/usr/local/cuda-${CUDA_DOTTED}/targets/${target}"
  [[ -f "${src}/include/cuda.h" ]] || die "cuda.h missing after extraction"
  [[ -f "${src}/lib/stubs/libcuda.so" ]] || die "stubs/libcuda.so missing after extraction"

  rm -rf "$dest"
  mkdir -p "${dest}/lib/stubs"
  cp -a "${src}/include" "${dest}/include"
  cp -a "${src}/lib/stubs/libcuda.so" "${dest}/lib/stubs/libcuda.so"
  echo "$want_stamp" >"$stamp"
  echo "fetch_cuda_headers: installed ${target} into ${dest}"
}

if [[ $# -eq 0 ]]; then
  set -- "$(uname -m)"
fi
for a in "$@"; do
  fetch_arch "$a"
done
