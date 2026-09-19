#!/usr/bin/env bash
# Run a command with libtessera masquerading as the CUDA driver.
#
#   infra/with_shim.sh <command> [args...]
#
# The masquerade strategy (ADR-001): a directory that contains only
# libcuda.so.1 and libcuda.so - both the shim - is put first on
# LD_LIBRARY_PATH, so every loader lookup for the driver soname resolves to the
# shim, including dlopen("libcuda.so.1") on an explicit handle, which LD_PRELOAD
# cannot reach. The shim then needs the real driver, whose absolute path is
# handed to it in TESSERA_REAL_LIBCUDA.
#
# Inputs (all optional, all with loud failures rather than silent defaults):
#   TESSERA_MASQ_DIR       the masquerade directory. Default:
#                          ${TESSERA_BUILD_DIR}/shim/masq, and if that is unset,
#                          <repo>/build/<TESSERA_PRESET:-gcc-release>/shim/masq.
#   TESSERA_REAL_LIBCUDA   the real driver. Default: discovered with ldconfig.
#
# Nothing here is masked: set -euo pipefail, and the command is exec'd so its
# exit code is this script's exit code (I-9).
set -euo pipefail

die() {
  echo "with_shim: $*" >&2
  exit 1
}

[[ $# -ge 1 ]] || {
  echo "usage: $0 <command> [args...]" >&2
  exit 64
}

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${here}/.." && pwd)"

# --- 1. The masquerade directory -------------------------------------------
if [[ -n "${TESSERA_MASQ_DIR:-}" ]]; then
  masq="${TESSERA_MASQ_DIR}"
elif [[ -n "${TESSERA_BUILD_DIR:-}" ]]; then
  masq="${TESSERA_BUILD_DIR}/shim/masq"
else
  masq="${repo_root}/build/${TESSERA_PRESET:-gcc-release}/shim/masq"
fi

[[ -d "$masq" ]] || die "masquerade directory not found: $masq
  Build the shim first, or set TESSERA_MASQ_DIR / TESSERA_BUILD_DIR."
masq="$(cd -- "$masq" && pwd)"

for soname in libcuda.so.1 libcuda.so; do
  [[ -e "${masq}/${soname}" ]] ||
    die "masquerade directory ${masq} has no ${soname}"
done

# The directory must contain nothing else: anything extra would be injected
# into every tenant's library search path ahead of the system libraries.
extra="$(find "$masq" -mindepth 1 -maxdepth 1 \
  ! -name libcuda.so.1 ! -name libcuda.so -printf '%f\n' | sort)"
[[ -z "$extra" ]] || die "masquerade directory ${masq} must contain only
  libcuda.so.1 and libcuda.so; it also contains:
$(printf '  %s\n' "$extra")"

# --- 2. The real driver -----------------------------------------------------
real=""
if [[ -n "${TESSERA_REAL_LIBCUDA:-}" ]]; then
  real="${TESSERA_REAL_LIBCUDA}"
else
  command -v ldconfig >/dev/null 2>&1 || die "ldconfig not found; cannot locate
  the real driver. Set TESSERA_REAL_LIBCUDA to its absolute path."
  ldconfig_cache=""
  ldconfig_cache="$(ldconfig -p)" || die "'ldconfig -p' failed"
  # Cache lines look like:
  #   libcuda.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcuda.so.1
  # Take the first entry whose path is absolute and outside the masquerade
  # directory, so a stale cache entry for the shim can never be chosen.
  while IFS= read -r line; do
    case "$line" in
      *libcuda.so.1*) ;;
      *) continue ;;
    esac
    path="${line#*=> }"
    if [[ "$path" != /* ]]; then
      continue
    fi
    if [[ "$(dirname -- "$path")" == "$masq" ]]; then
      continue
    fi
    real="$path"
    break
  done <<<"$ldconfig_cache"
fi

[[ -n "$real" ]] || die "no real libcuda.so.1 found by ldconfig.
  On a GPU host the driver is usually /usr/lib/x86_64-linux-gnu/libcuda.so.1.
  Set TESSERA_REAL_LIBCUDA explicitly if it lives somewhere else."
[[ -e "$real" ]] || die "TESSERA_REAL_LIBCUDA does not exist: $real"
real="$(readlink -f -- "$real")" || die "cannot resolve $real"
[[ -f "$real" ]] || die "resolved real driver is not a regular file: $real"

# T6 (misconfiguration): pointing the shim at itself must not be possible here.
if [[ "$(dirname -- "$real")" == "$masq" ]]; then
  die "TESSERA_REAL_LIBCUDA resolves into the masquerade directory ($real);
  that would make the shim forward to itself."
fi
for soname in libcuda.so.1 libcuda.so; do
  if [[ -e "${masq}/${soname}" ]] &&
    [[ "$(readlink -f -- "${masq}/${soname}")" == "$real" ]]; then
    die "masquerade ${soname} resolves to the same file as the real driver
  ($real); the masquerade directory is not holding the shim."
  fi
done

# --- 3. Environment and exec ------------------------------------------------
export TESSERA_REAL_LIBCUDA="$real"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${masq}:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${masq}"
fi

if [[ -n "${TESSERA_WITH_SHIM_VERBOSE:-}" ]]; then
  echo "with_shim: LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >&2
  echo "with_shim: TESSERA_REAL_LIBCUDA=${TESSERA_REAL_LIBCUDA}" >&2
fi

exec "$@"
