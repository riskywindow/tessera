#!/usr/bin/env bash
# clang-format over every first-party C/C++ source in the repo.
#
#   scripts/format.sh            rewrite files in place
#   scripts/format.sh --check    dry run; prints the diff and fails if any
#                                file is not already formatted
#   scripts/format.sh [--check] <path>...   restrict to these paths
#
# The pinned version is clang-format 19.1.7, which is what CI installs (see
# .github/workflows/ci.yml). clang-format's output changes between major
# releases, so a different major version here would produce a diff CI does not
# agree with; the script says so rather than quietly reformatting the tree.
# Override the binary with TESSERA_CLANG_FORMAT=/path/to/clang-format.
set -euo pipefail

TESSERA_CLANG_FORMAT_VERSION="19"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

die() {
  printf 'format: %s\n' "$*" >&2
  exit 2
}

find_clang_format() {
  local cand
  if [[ -n "${TESSERA_CLANG_FORMAT:-}" ]]; then
    command -v "${TESSERA_CLANG_FORMAT}" >/dev/null 2>&1 ||
      die "TESSERA_CLANG_FORMAT=${TESSERA_CLANG_FORMAT} is not executable"
    printf '%s' "${TESSERA_CLANG_FORMAT}"
    return 0
  fi
  for cand in "clang-format-${TESSERA_CLANG_FORMAT_VERSION}" clang-format; do
    if command -v "$cand" >/dev/null 2>&1; then
      printf '%s' "$cand"
      return 0
    fi
  done
  die "no clang-format on PATH (want ${TESSERA_CLANG_FORMAT_VERSION}.x)"
}

# Tracked files plus anything new that is not ignored: a formatting gate that
# only sees committed content cannot gate the commit that is being prepared.
list_sources() {
  local f
  if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$repo_root" ls-files --cached --others --exclude-standard
  else
    (cd -- "$repo_root" && find . -type f -printf '%P\n' | sort)
  fi | while IFS= read -r f; do
    case "$f" in
      build/* | */_deps/* | third_party/*) continue ;;
      *.c | *.cc | *.cpp | *.cxx | *.h | *.hh | *.hpp | *.hxx | *.cu | *.cuh) printf '%s\n' "$f" ;;
    esac
  done
}

main() {
  local check=0
  local -a paths=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --check | -n) check=1 ;;
      -h | --help)
        sed -n '2,13p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//'
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

  local cf version major
  cf="$(find_clang_format)"
  version="$("$cf" --version)"
  major="$(printf '%s' "$version" | sed -n 's/.*clang-format version \([0-9][0-9]*\).*/\1/p')"
  [[ -n "$major" ]] || die "cannot parse version from: ${version}"
  if [[ "$major" != "$TESSERA_CLANG_FORMAT_VERSION" ]]; then
    printf 'format: WARNING %s is major %s; CI pins %s.x, so its diff may differ\n' \
      "$cf" "$major" "$TESSERA_CLANG_FORMAT_VERSION" >&2
  fi

  if [[ ${#paths[@]} -eq 0 ]]; then
    while IFS= read -r f; do
      paths+=("${repo_root}/${f}")
    done < <(list_sources)
  fi
  if [[ ${#paths[@]} -eq 0 ]]; then
    printf 'format: no C/C++ sources found under %s\n' "$repo_root" >&2
    return 0
  fi

  printf 'format: %s (%s) over %d file(s)\n' "$cf" "$version" "${#paths[@]}"

  if [[ "$check" -eq 1 ]]; then
    local rc=0 f
    for f in "${paths[@]}"; do
      if ! "$cf" --style=file --dry-run -Werror "$f"; then
        rc=1
      fi
    done
    if [[ "$rc" -ne 0 ]]; then
      printf 'format: files above are not formatted; run scripts/format.sh\n' >&2
    fi
    return "$rc"
  fi

  "$cf" --style=file -i "${paths[@]}"
  printf 'format: formatted %d file(s)\n' "${#paths[@]}"
}

main "$@"
