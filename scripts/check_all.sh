#!/usr/bin/env bash
# The full local gate: everything that must be green before a WP is committed.
#
#   1. scripts/lint_exit_codes.sh --self-test   (the linter is itself tested)
#   2. scripts/lint_exit_codes.sh               (I-9 across the repo)
#   3. scripts/format.sh --check
#   4. every preset, one at a time: configure, build, ctest
#        native x86-64: gcc-debug gcc-release clang-debug clang-release
#                       gcc-asan clang-asan clang-tsan
#        cross arm64:   arm64-gcc arm64-clang   (ctest runs under qemu-aarch64)
#   5. scripts/tidy.sh
#
# Presets are never built in parallel: several agents share this 8-core,
# 15 GiB host and the build trees live on tmpfs.
#
#   scripts/check_all.sh
#   scripts/check_all.sh --json results/m0/local_matrix.json
#   scripts/check_all.sh --jobs 2 --presets gcc-debug,clang-asan
#
# Exit status is non-zero if any step failed. --json writes a machine-readable
# summary: per-preset status, test counts, compiler versions, the git commit,
# and the exact command line of every step.
set -euo pipefail
# Decimal points in EPOCHREALTIME and in awk's output must not follow the
# locale, or the JSON durations stop being JSON numbers.
export LC_ALL=C

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

DEFAULT_PRESETS="gcc-debug gcc-release clang-debug clang-release gcc-asan clang-asan clang-tsan arm64-gcc arm64-clang"
TIDY_PRESET="clang-debug"

die() {
  printf 'check_all: %s\n' "$*" >&2
  exit 2
}

# --- JSON helpers ----------------------------------------------------------
json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\t'/\\t}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\n'/\\n}"
  printf '%s' "$s"
}

json_str() {
  printf '"%s"' "$(json_escape "$1")"
}

# --- step bookkeeping ------------------------------------------------------
declare -a ROW_NAME=() ROW_STATUS=() ROW_TESTS=() ROW_SECONDS=() ROW_DETAIL=()
declare -a STEP_JSON=() PRESET_JSON=()
OVERALL_RC=0

elapsed_since() {
  local start="$1" now
  now="${EPOCHREALTIME:-$(date +%s).0}"
  printf '%.1f' "$(awk -v a="$start" -v b="$now" 'BEGIN { printf "%.3f", b - a }')"
}

# run_logged <logfile> <command...>: echoes the command into the log, streams
# the output, and returns the command's status unchanged.
run_logged() {
  local log="$1"
  shift
  local rc=0
  mkdir -p -- "$(dirname -- "$log")"
  printf '+ %s\n' "$*" | tee -- "$log"
  "$@" 2>&1 | tee -a -- "$log" || rc=$?
  return "$rc"
}

record_row() {
  ROW_NAME+=("$1")
  ROW_STATUS+=("$2")
  ROW_TESTS+=("$3")
  ROW_SECONDS+=("$4")
  ROW_DETAIL+=("$5")
}

# run_simple_step <name> <detail> <command...>
run_simple_step() {
  local name="$1" detail="$2"
  shift 2
  local log="${repo_root}/build/check_all/${name}.log"
  local start="${EPOCHREALTIME:-$(date +%s).0}"
  local rc=0

  printf '\n=== %s ===\n' "$name"
  run_logged "$log" "$@" || rc=$?

  local seconds status
  seconds="$(elapsed_since "$start")"
  if [[ "$rc" -eq 0 ]]; then status="PASS"; else status="FAIL"; OVERALL_RC=1; fi

  record_row "$name" "$status" "-" "$seconds" "$detail"
  STEP_JSON+=("$(
    printf '{"name": %s, "status": %s, "exit_code": %d, "seconds": %s, "command": %s, "log": %s}' \
      "$(json_str "$name")" "$(json_str "$status")" "$rc" "$seconds" \
      "$(json_str "$*")" "$(json_str "$log")"
  )")
}

cache_value() {
  local cache="$1" key="$2"
  [[ -f "$cache" ]] || return 0
  sed -n "s|^${key}:[^=]*=||p" "$cache" | head -n1
}

compiler_info() {
  # $1 = build dir, $2 = C or CXX; sets CI_PATH, CI_ID, CI_VERSION, CI_LINE
  local build_dir="$1" lang="$2" info
  CI_PATH=""
  CI_ID=""
  CI_VERSION=""
  CI_LINE=""
  # CMake<lang>Compiler.cmake is the authoritative record: a toolchain file
  # that sets CMAKE_CXX_COMPILER as a normal variable (both cross toolchains
  # do) never writes it into CMakeCache.txt, so the cache is only a fallback.
  info="$(find "${build_dir}/CMakeFiles" -maxdepth 2 -name "CMake${lang}Compiler.cmake" -print -quit)"
  if [[ -n "$info" && -f "$info" ]]; then
    CI_PATH="$(sed -n "s/^set(CMAKE_${lang}_COMPILER \"\\(.*\\)\")$/\\1/p" "$info" | head -n1)"
    CI_ID="$(sed -n "s/^set(CMAKE_${lang}_COMPILER_ID \"\\(.*\\)\")$/\\1/p" "$info" | head -n1)"
    CI_VERSION="$(sed -n "s/^set(CMAKE_${lang}_COMPILER_VERSION \"\\(.*\\)\")$/\\1/p" "$info" | head -n1)"
  fi
  if [[ -z "$CI_PATH" ]]; then
    CI_PATH="$(cache_value "${build_dir}/CMakeCache.txt" "CMAKE_${lang}_COMPILER")"
  fi
  # CMAKE_<lang>_COMPILER may be a bare name ("g++") or a full path; resolve it
  # so the recorded version line comes from the binary that was actually used.
  local resolved out
  if [[ -n "$CI_PATH" ]] && resolved="$(command -v -- "$CI_PATH")"; then
    CI_PATH="$resolved"
    out="$("$resolved" --version 2>/dev/null)"
    CI_LINE="${out%%$'\n'*}"
  fi
}

# run_preset <preset> <jobs>
run_preset() {
  local preset="$1" jobs="$2"
  local build_dir="${repo_root}/build/${preset}"
  local logdir="${repo_root}/build/check_all"
  local start="${EPOCHREALTIME:-$(date +%s).0}"

  printf '\n=== preset %s ===\n' "$preset"

  local cfg_cmd=(cmake --preset "$preset")
  local bld_cmd=(cmake --build --preset "$preset" -j "$jobs")
  local tst_cmd=(ctest --preset "$preset" --parallel "$jobs")

  local cfg_rc=0 bld_rc=0 tst_rc=0
  local cfg_s bld_s tst_s t0

  t0="${EPOCHREALTIME:-$(date +%s).0}"
  run_logged "${logdir}/${preset}.configure.log" "${cfg_cmd[@]}" || cfg_rc=$?
  cfg_s="$(elapsed_since "$t0")"

  if [[ "$cfg_rc" -eq 0 ]]; then
    t0="${EPOCHREALTIME:-$(date +%s).0}"
    run_logged "${logdir}/${preset}.build.log" "${bld_cmd[@]}" || bld_rc=$?
    bld_s="$(elapsed_since "$t0")"
  else
    bld_rc=125
    bld_s="0.0"
  fi

  local tests_total="" tests_failed="" tests_passed=""
  if [[ "$bld_rc" -eq 0 ]]; then
    t0="${EPOCHREALTIME:-$(date +%s).0}"
    run_logged "${logdir}/${preset}.test.log" "${tst_cmd[@]}" || tst_rc=$?
    tst_s="$(elapsed_since "$t0")"
    local tlog="${logdir}/${preset}.test.log"
    tests_failed="$(sed -n 's/^.*tests passed, \([0-9][0-9]*\) tests failed out of \([0-9][0-9]*\).*$/\1/p' "$tlog" | tail -n1)"
    tests_total="$(sed -n 's/^.*tests passed, \([0-9][0-9]*\) tests failed out of \([0-9][0-9]*\).*$/\2/p' "$tlog" | tail -n1)"
    if [[ -n "$tests_total" && -n "$tests_failed" ]]; then
      tests_passed=$((tests_total - tests_failed))
    fi
  else
    tst_rc=125
    tst_s="0.0"
  fi

  local status="PASS"
  if [[ "$cfg_rc" -ne 0 || "$bld_rc" -ne 0 || "$tst_rc" -ne 0 ]]; then
    status="FAIL"
    OVERALL_RC=1
  fi

  compiler_info "$build_dir" CXX
  local cxx_path="$CI_PATH" cxx_id="$CI_ID" cxx_ver="$CI_VERSION" cxx_line="$CI_LINE"
  compiler_info "$build_dir" C
  local c_path="$CI_PATH" c_id="$CI_ID" c_ver="$CI_VERSION"

  local build_type cuda_inc cuda_stub sanitize
  build_type="$(cache_value "${build_dir}/CMakeCache.txt" CMAKE_BUILD_TYPE)"
  cuda_inc="$(cache_value "${build_dir}/CMakeCache.txt" TESSERA_CUDA_INCLUDE_DIR)"
  cuda_stub="$(cache_value "${build_dir}/CMakeCache.txt" TESSERA_CUDA_STUB_LIBCUDA)"
  sanitize="$(cache_value "${build_dir}/CMakeCache.txt" TESSERA_SANITIZE)"

  local tests_cell="-"
  if [[ -n "$tests_total" ]]; then
    tests_cell="${tests_passed}/${tests_total}"
  elif [[ "$status" == "FAIL" ]]; then
    tests_cell="n/a"
  fi

  local total_s
  total_s="$(elapsed_since "$start")"
  record_row "$preset" "$status" "$tests_cell" "$total_s" "${cxx_id} ${cxx_ver}"

  local tests_json='null'
  if [[ -n "$tests_total" ]]; then
    tests_json="$(printf '{"total": %d, "passed": %d, "failed": %d}' \
      "$tests_total" "$tests_passed" "$tests_failed")"
  fi

  PRESET_JSON+=("$(
    printf '{"preset": %s, "status": %s, "seconds": %s, "build_dir": %s, "build_type": %s, "sanitizers": %s, "jobs": %d,' \
      "$(json_str "$preset")" "$(json_str "$status")" "$total_s" \
      "$(json_str "$build_dir")" "$(json_str "$build_type")" "$(json_str "$sanitize")" "$jobs"
    printf ' "cxx": {"path": %s, "id": %s, "version": %s, "version_line": %s},' \
      "$(json_str "$cxx_path")" "$(json_str "$cxx_id")" "$(json_str "$cxx_ver")" "$(json_str "$cxx_line")"
    printf ' "cc": {"path": %s, "id": %s, "version": %s},' \
      "$(json_str "$c_path")" "$(json_str "$c_id")" "$(json_str "$c_ver")"
    printf ' "cuda": {"include_dir": %s, "stub_libcuda": %s},' \
      "$(json_str "$cuda_inc")" "$(json_str "$cuda_stub")"
    printf ' "tests": %s,' "$tests_json"
    printf ' "configure": {"command": %s, "exit_code": %d, "seconds": %s},' \
      "$(json_str "${cfg_cmd[*]}")" "$cfg_rc" "$cfg_s"
    printf ' "build": {"command": %s, "exit_code": %d, "seconds": %s},' \
      "$(json_str "${bld_cmd[*]}")" "$bld_rc" "$bld_s"
    printf ' "test": {"command": %s, "exit_code": %d, "seconds": %s}}' \
      "$(json_str "${tst_cmd[*]}")" "$tst_rc" "$tst_s"
  )")
}

print_summary() {
  local i
  printf '\n'
  printf '%-18s %-6s %-9s %9s  %s\n' CONFIGURATION STATUS TESTS SECONDS DETAIL
  printf '%-18s %-6s %-9s %9s  %s\n' '------------------' '------' '---------' '---------' '------'
  for i in "${!ROW_NAME[@]}"; do
    printf '%-18s %-6s %-9s %9s  %s\n' \
      "${ROW_NAME[$i]}" "${ROW_STATUS[$i]}" "${ROW_TESTS[$i]}" "${ROW_SECONDS[$i]}" "${ROW_DETAIL[$i]}"
  done
  printf '\n'
  if [[ "$OVERALL_RC" -eq 0 ]]; then
    printf 'check_all: PASS (%d configurations)\n' "${#ROW_NAME[@]}"
  else
    printf 'check_all: FAIL\n' >&2
  fi
}

write_json() {
  local out="$1" jobs="$2" presets="$3"
  local commit branch dirty uname_s cmake_v ninja_v now

  commit=""
  branch=""
  dirty="false"
  if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    commit="$(git -C "$repo_root" rev-parse HEAD)"
    branch="$(git -C "$repo_root" rev-parse --abbrev-ref HEAD)"
    if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
      dirty="true"
    fi
  fi
  uname_s="$(uname -srm)"
  cmake_v="$(cmake --version | head -n1)"
  ninja_v="$(ninja --version)"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  mkdir -p -- "$(dirname -- "$out")"
  {
    printf '{\n'
    printf '  "schema": "tessera.local_matrix/1",\n'
    printf '  "generated_utc": %s,\n' "$(json_str "$now")"
    printf '  "overall": %s,\n' "$(json_str "$([[ "$OVERALL_RC" -eq 0 ]] && printf PASS || printf FAIL)")"
    printf '  "git": {"commit": %s, "branch": %s, "dirty": %s},\n' \
      "$(json_str "$commit")" "$(json_str "$branch")" "$dirty"
    printf '  "host": {"uname": %s, "nproc": %s, "cmake": %s, "ninja": %s},\n' \
      "$(json_str "$uname_s")" "$(nproc)" "$(json_str "$cmake_v")" "$(json_str "$ninja_v")"
    printf '  "jobs_per_build": %s,\n' "$jobs"
    printf '  "presets_requested": ['
    local first=1 p
    for p in $presets; do
      [[ "$first" -eq 1 ]] || printf ', '
      first=0
      json_str "$p"
    done
    printf '],\n'
    printf '  "steps": [\n'
    local i
    for i in "${!STEP_JSON[@]}"; do
      printf '    %s' "${STEP_JSON[$i]}"
      [[ $((i + 1)) -eq ${#STEP_JSON[@]} ]] || printf ','
      printf '\n'
    done
    printf '  ],\n'
    printf '  "presets": [\n'
    for i in "${!PRESET_JSON[@]}"; do
      printf '    %s' "${PRESET_JSON[$i]}"
      [[ $((i + 1)) -eq ${#PRESET_JSON[@]} ]] || printf ','
      printf '\n'
    done
    printf '  ]\n'
    printf '}\n'
  } >"$out"

  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$out"
    printf 'check_all: wrote valid JSON to %s\n' "$out"
  else
    printf 'check_all: wrote %s (python3 absent, JSON not validated)\n' "$out" >&2
  fi
}

main() {
  local json_out="" jobs=3 presets="$DEFAULT_PRESETS"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --json)
        [[ $# -ge 2 ]] || die "--json needs a path"
        json_out="$2"
        shift
        ;;
      --json=*) json_out="${1#--json=}" ;;
      --jobs | -j)
        [[ $# -ge 2 ]] || die "--jobs needs a value"
        jobs="$2"
        shift
        ;;
      --jobs=*) jobs="${1#--jobs=}" ;;
      --presets)
        [[ $# -ge 2 ]] || die "--presets needs a comma-separated list"
        presets="${2//,/ }"
        shift
        ;;
      --presets=*) presets="${1#--presets=}"; presets="${presets//,/ }" ;;
      --no-presets) presets="" ;;
      -h | --help)
        sed -n '2,25p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//'
        return 0
        ;;
      *) die "unknown option: $1" ;;
    esac
    shift
  done

  [[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
  if [[ "$jobs" -gt 3 ]]; then
    printf 'check_all: WARNING -j%s exceeds the -j3 shared-host limit in CLAUDE.md\n' "$jobs" >&2
  fi

  # Every 'cmake --preset' / 'ctest --preset' invocation resolves
  # CMakePresets.json relative to the working directory.
  cd -- "$repo_root"
  mkdir -p -- "${repo_root}/build/check_all"

  printf 'check_all: repo %s\n' "$repo_root"
  printf 'check_all: presets [%s] at -j%s, one at a time\n' "$presets" "$jobs"

  run_simple_step "lint-self-test" "fixtures under scripts/testdata" \
    "${repo_root}/scripts/lint_exit_codes.sh" --self-test
  run_simple_step "lint" "I-9 across tracked and new files" \
    "${repo_root}/scripts/lint_exit_codes.sh"
  run_simple_step "format-check" "clang-format --dry-run -Werror" \
    "${repo_root}/scripts/format.sh" --check

  local p
  for p in $presets; do
    run_preset "$p" "$jobs"
  done

  run_simple_step "tidy" "preset ${TIDY_PRESET}" \
    "${repo_root}/scripts/tidy.sh" --preset "$TIDY_PRESET" --jobs "$jobs"

  print_summary

  if [[ -n "$json_out" ]]; then
    write_json "$json_out" "$jobs" "$presets"
  fi

  return "$OVERALL_RC"
}

main "$@"
