#!/usr/bin/env bash
# clang-tidy over a preset's compile database. Any diagnostic fails the run:
# the policy in .clang-tidy sets WarningsAsErrors '*', and this script also
# passes it explicitly so the gate does not depend on a config file a future
# edit might relax by accident.
#
#   scripts/tidy.sh                       clang-debug, every first-party TU
#   scripts/tidy.sh --preset clang-release
#   scripts/tidy.sh --jobs 1              serial, for readable output
#   scripts/tidy.sh --fix                 apply the fixes clang-tidy suggests
#   scripts/tidy.sh -- tests/common/tenant_page_test.cc
#
# The preset must be a clang preset: the compile database of a gcc or cross
# preset contains flags clang does not accept. Override the binary with
# TESSERA_CLANG_TIDY=/path/to/clang-tidy.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

die() {
  printf 'tidy: %s\n' "$*" >&2
  exit 2
}

find_clang_tidy() {
  local cand
  if [[ -n "${TESSERA_CLANG_TIDY:-}" ]]; then
    command -v "${TESSERA_CLANG_TIDY}" >/dev/null 2>&1 ||
      die "TESSERA_CLANG_TIDY=${TESSERA_CLANG_TIDY} is not executable"
    printf '%s' "${TESSERA_CLANG_TIDY}"
    return 0
  fi
  for cand in clang-tidy-20 clang-tidy-19 clang-tidy-18 clang-tidy; do
    if command -v "$cand" >/dev/null 2>&1; then
      printf '%s' "$cand"
      return 0
    fi
  done
  die "no clang-tidy on PATH"
}

# Translation units from the compile database that belong to Tessera: inside
# the repo and outside the build tree (which holds the fetched GoogleTest).
list_translation_units() {
  local db="$1"
  python3 - "$db" "$repo_root" <<'PY'
import json
import sys

db_path, root = sys.argv[1], sys.argv[2].rstrip("/")
with open(db_path, encoding="utf-8") as handle:
    entries = json.load(handle)

seen = set()
for entry in entries:
    path = entry["file"]
    if not path.startswith(root + "/"):
        continue
    rel = path[len(root) + 1 :]
    if rel.startswith("build/") or "/_deps/" in path:
        continue
    if path not in seen:
        seen.add(path)
        print(path)
PY
}

main() {
  local preset="clang-debug" jobs=3 fix=0
  local -a paths=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --preset)
        [[ $# -ge 2 ]] || die "--preset needs a value"
        preset="$2"
        shift
        ;;
      --preset=*) preset="${1#--preset=}" ;;
      --jobs | -j)
        [[ $# -ge 2 ]] || die "--jobs needs a value"
        jobs="$2"
        shift
        ;;
      --jobs=*) jobs="${1#--jobs=}" ;;
      --fix) fix=1 ;;
      -h | --help)
        sed -n '2,16p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//'
        return 0
        ;;
      --)
        shift
        paths+=("$@")
        break
        ;;
      -*) die "unknown option: $1" ;;
      *) paths+=("$1") ;;
    esac
    shift
  done

  command -v python3 >/dev/null 2>&1 || die "python3 is required to read the compile database"

  local build_dir="${repo_root}/build/${preset}"
  local db="${build_dir}/compile_commands.json"
  if [[ ! -f "$db" ]]; then
    printf 'tidy: %s not found; configuring preset %s\n' "$db" "$preset"
    cmake --preset "$preset" -S "$repo_root"
  fi
  [[ -f "$db" ]] || die "preset ${preset} produced no compile_commands.json"

  local ct version
  ct="$(find_clang_tidy)"
  version="$("$ct" --version | sed -n 's/.*LLVM version \([0-9.]*\).*/\1/p' | head -n1)"
  [[ -n "$version" ]] || version="unknown"

  if [[ ${#paths[@]} -eq 0 ]]; then
    local tu
    while IFS= read -r tu; do
      paths+=("$tu")
    done < <(list_translation_units "$db")
  fi

  if [[ ${#paths[@]} -eq 0 ]]; then
    die "no first-party translation units in ${db}; nothing would be checked"
  fi

  # Anchor the header filter at this checkout so headers under
  # build/<preset>/_deps are never analysed, whatever the clone is called.
  local header_filter="^${repo_root}/(common|shim|daemon|cli|fake_driver|bench|tests)/"

  local -a args=(
    -p "$build_dir"
    --quiet
    --warnings-as-errors=*
    "--header-filter=${header_filter}"
    --extra-arg=-Wno-unknown-warning-option
  )
  if [[ "$fix" -eq 1 ]]; then
    args+=(--fix --fix-errors --format-style=file)
  fi

  printf 'tidy: %s (LLVM %s), preset %s, %d translation unit(s), -j%s\n' \
    "$ct" "$version" "$preset" "${#paths[@]}" "$jobs"

  local rc=0
  if ! printf '%s\0' "${paths[@]}" |
    xargs -0 -r -n1 -P"$jobs" "$ct" "${args[@]}"; then
    rc=1
  fi

  if [[ "$rc" -ne 0 ]]; then
    printf 'tidy: FAIL (%s, preset %s)\n' "$ct" "$preset" >&2
  else
    printf 'tidy: PASS (%d translation unit(s))\n' "${#paths[@]}"
  fi
  return "$rc"
}

main "$@"
