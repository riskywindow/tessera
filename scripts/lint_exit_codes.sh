#!/usr/bin/env bash
# I-9 enforcement: exit codes are never masked.
#
# Scans tracked shell scripts, CI YAML, and Python runner scripts for patterns
# that swallow a failure, prints "file:line: [rule] message" for each, and
# exits non-zero if anything was found.
#
# Rules (see --rules for the authoritative list):
#   or-true               a failure discarded with '|| true' or '|| :'
#   set-plus-e            errexit/pipefail turned back off mid-script
#   exit-zero             a zero exit taken from inside a failure-handling block
#   missing-set-e         a shell script that never enables errexit
#   missing-pipefail      a shell script with a pipeline but no pipefail
#   subprocess-no-check   subprocess.run() with neither check=True nor a
#                         returncode inspection
#   unchecked-call        APIs that return a status nobody can be forced to
#                         read: subprocess.call, subprocess.getoutput,
#                         os.system
#   except-pass           an except handler whose entire body is pass/continue
#   modal-masked          a 'modal run' whose failure is discarded (GPU spend
#                         that reports success is worse than a red build)
#   ci-continue-on-error  'continue-on-error: true' in a workflow
#   bad-allow-directive   an allow directive that names no known rule or gives
#                         no justification
#
# Allowlisting. A finding is suppressed only by an explicit, justified comment
# on the same line:
#
#     rm -f "$f" || true  # tessera-lint: allow-or-true best-effort cleanup
#
# The justification is mandatory (four characters or more) and the rule name
# must exist, so a legitimate exception stays visible in the diff and in grep
# instead of disappearing. The two file-scoped rules (missing-set-e,
# missing-pipefail) accept their directive on any line of the file, because
# the line they are reported at may not be able to carry a comment.
#
# Usage:
#   scripts/lint_exit_codes.sh [path ...]   scan the repo, or just these paths
#   scripts/lint_exit_codes.sh --self-test  run the fixtures under scripts/testdata
#   scripts/lint_exit_codes.sh --rules      print the rule table
#
# Fixtures under scripts/testdata/ are deliberately broken and are excluded
# from the normal scan; --self-test is the only thing that looks at them.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fixture_dir="${repo_root}/scripts/testdata"

die() {
  printf 'lint_exit_codes: %s\n' "$*" >&2
  exit 2
}

# ---------------------------------------------------------------------------
# The scanner. One awk pass per file; the whole file is buffered so the rules
# can look both backwards (enclosing block) and forwards (multi-line calls).
# ---------------------------------------------------------------------------
lint_awk="$(
  cat <<'AWK_PROGRAM'
function strip_comment(s,   i, c, out, inq, q, prev) {
  inq = 0
  q = ""
  out = ""
  prev = " "
  for (i = 1; i <= length(s); i++) {
    c = substr(s, i, 1)
    if (inq) {
      out = out c
      if (c == q) { inq = 0 }
      prev = c
      continue
    }
    if (c == "'" || c == "\"") {
      inq = 1
      q = c
      out = out c
      prev = c
      continue
    }
    if (c == "#" && (i == 1 || prev == " " || prev == "\t")) { break }
    out = out c
    prev = c
  }
  return out
}

function trim(s) {
  gsub(/^[ \t\r]+/, "", s)
  gsub(/[ \t\r]+$/, "", s)
  return s
}

function indent_of(s,   i, c, w) {
  w = 0
  for (i = 1; i <= length(s); i++) {
    c = substr(s, i, 1)
    if (c == " ") { w += 1 }
    else if (c == "\t") { w += 8 }
    else { break }
  }
  return w
}

# Returns the justification text of a valid allow directive for <rule> on <s>,
# or "" when there is none.
function directive_reason(s, rule,   rest) {
  if (match(s, "#[ \t]*tessera-lint:[ \t]*allow-" rule "[ \t]+")) {
    rest = trim(substr(s, RSTART + RLENGTH))
    if (length(rest) >= 4) { return rest }
  }
  return ""
}

function allowed(lineno, rule,   i) {
  if (directive_reason(RAW[lineno], rule) != "") { return 1 }
  if (rule in FILESCOPED) {
    for (i = 1; i <= n; i++) {
      if (directive_reason(RAW[i], rule) != "") { return 1 }
    }
  }
  return 0
}

function report(lineno, rule, msg) {
  if (allowed(lineno, rule)) { return }
  printf "%s:%d: [%s] %s\n", file, lineno, rule, msg
  findings++
}

# --- shared line rules (shell syntax, which also covers YAML 'run:' blocks) --

function failure_marker(s) {
  if (s ~ /\|\|/) { return 1 }
  if (s ~ /^[ \t]*(if|elif|while|until)[ \t]*!/) { return 1 }
  if (s ~ /\$\?/) { return 1 }
  if (s ~ /-ne[ \t]+0/) { return 1 }
  if (s ~ /![=][ \t]*0/) { return 1 }
  if (s ~ /^[ \t]*trap[ \t].*ERR/) { return 1 }
  return 0
}

function command_condition(s,   rest) {
  if (!match(s, /^[ \t]*(if|elif)[ \t]+/)) { return 0 }
  rest = substr(s, RSTART + RLENGTH)
  if (rest ~ /^(\[|test[ \t]|!)/) { return 0 }
  return 1
}

function opens_block(s) {
  if (s ~ /(^|[ \t;&])(then|do)[ \t]*$/) { return 1 }
  if (s ~ /\{[ \t]*$/) { return 1 }
  if (s ~ /(^|[ \t;&])in[ \t]*$/ && s ~ /(^|[ \t;&])case[ \t]/) { return 1 }
  return 0
}

function closes_block(s) {
  if (s ~ /(^|[ \t;&])(fi|done|esac)([ \t;&)]|$)/) { return 1 }
  if (s ~ /^[ \t]*\}/) { return 1 }
  return 0
}

function has_errexit(s) {
  if (s ~ /(^[ \t]*|[;&][ \t]*)set[ \t]+(-[a-zA-Z]+[ \t]+)*-o[ \t]+errexit/) { return 1 }
  if (s ~ /(^[ \t]*|[;&][ \t]*)set[ \t]+-[a-zA-Z]*e/) { return 1 }
  return 0
}

# Accepts every spelling that turns pipefail on ("set -o pipefail",
# "set -euo pipefail", "set -eo pipefail") and rejects the one that turns it
# back off ("set +o pipefail").
function has_pipefail(s) {
  if (s ~ /(^[ \t]*|[;&][ \t]*)set[ \t]+/ && s ~ /pipefail/ && s !~ /\+o[ \t]+pipefail/) {
    return 1
  }
  return 0
}

function is_pipeline(s,   t) {
  t = s
  gsub(/\|\|/, "", t)
  if (t ~ /[ \t]\|/) { return 1 }
  return 0
}

function scan_directives(   i, raw, rule, rest) {
  for (i = 1; i <= n; i++) {
    raw = RAW[i]
    if (!match(raw, /#[ \t]*tessera-lint:[ \t]*allow-[A-Za-z0-9_-]+/)) { continue }
    rule = substr(raw, RSTART, RLENGTH)
    sub(/^#[ \t]*tessera-lint:[ \t]*allow-/, "", rule)
    rest = trim(substr(raw, RSTART + RLENGTH))
    if (!(rule in KNOWN)) {
      report(i, "bad-allow-directive", "allow directive names unknown rule '" rule "'")
    } else if (length(rest) < 4) {
      report(i, "bad-allow-directive",
             "allow-" rule " needs a justification on the same line")
    }
  }
}

function scan_shell_lines(   i, raw, s, depth, fail_active, d) {
  depth = 0
  for (i = 1; i <= n; i++) {
    raw = RAW[i]
    s = strip_comment(raw)
    if (trim(s) == "") { continue }

    if (s ~ /\|\|[ \t]*(true|:|\/bin\/true|\/usr\/bin\/true)([ \t]*[;&)}]|[ \t]*$)/) {
      report(i, "or-true", "the failure is discarded by a trailing true/no-op; handle it or allowlist it")
    }
    if (s ~ /(^[ \t]*|[;&(][ \t]*)set[ \t]+(-[a-zA-Z]+[ \t]+)*\+[a-zA-Z]*e/ ||
        s ~ /(^[ \t]*|[;&(][ \t]*)set[ \t]+\+o[ \t]+(errexit|pipefail)/) {
      report(i, "set-plus-e", "errexit/pipefail disabled; the rest of the script cannot fail")
    }
    if (s ~ /modal[ \t]+run/ &&
        (s ~ /\|\|/ || s ~ /;[ \t]*(true|:)([ \t]*[;&)}]|[ \t]*$)/)) {
      report(i, "modal-masked", "a failed 'modal run' must not report success (GPU spend)")
    }
    if (kind == "yaml" && s ~ /^[ \t-]*continue-on-error:[ \t]*['"]?true['"]?[ \t]*$/) {
      report(i, "ci-continue-on-error", "continue-on-error hides a red step from the gate")
    }

    fail_active = 0
    for (d = 1; d <= depth; d++) {
      if (STK[d]) { fail_active = 1 }
    }
    if (s ~ /(^|[^A-Za-z0-9_])exit[ \t]+0([ \t]*[;&)}]|[ \t]*$)/ &&
        (fail_active || failure_marker(s))) {
      report(i, "exit-zero", "zero exit taken from a failure path")
    }

    if (s ~ /^[ \t]*(else|elif)([ \t;&]|$)/) {
      if (depth > 0 && (STK[depth] || STKCMD[depth])) { STK[depth] = 1 }
    }
    if (closes_block(s) && depth > 0) { depth-- }
    if (opens_block(s) && s !~ /^[ \t]*(else|elif)([ \t;&]|$)/) {
      depth++
      STK[depth] = failure_marker(s) ? 1 : 0
      STKCMD[depth] = command_condition(s)
    }
  }
}

function scan_shell_file(   i, s, saw_e, saw_pipefail, first_pipe) {
  saw_e = 0
  saw_pipefail = 0
  first_pipe = 0
  for (i = 1; i <= n; i++) {
    s = strip_comment(RAW[i])
    if (has_errexit(s)) { saw_e = 1 }
    if (has_pipefail(s)) { saw_pipefail = 1 }
    if (first_pipe == 0 && is_pipeline(s)) { first_pipe = i }
  }
  if (!saw_e) {
    report(1, "missing-set-e", "shell script never enables errexit (want: set -euo pipefail)")
  }
  if (first_pipe > 0 && !saw_pipefail) {
    report(first_pipe, "missing-pipefail",
           "pipeline in a script without 'set -o pipefail'; only the last stage can fail")
  }
}

# --- Python rules ----------------------------------------------------------

# Collects the text of a parenthesised call starting at RAW[start] from column
# <col>, leaving it in CALL_TEXT, and returns the line the call ends on.
function py_call_span(start, col,   i, j, s, depth, ch) {
  depth = 0
  CALL_TEXT = ""
  for (i = start; i <= n; i++) {
    s = strip_comment(i == start ? substr(RAW[i], col) : RAW[i])
    CALL_TEXT = CALL_TEXT " " s
    for (j = 1; j <= length(s); j++) {
      ch = substr(s, j, 1)
      if (ch == "(") { depth++ }
      else if (ch == ")") {
        depth--
        if (depth <= 0) { return i }
      }
    }
  }
  return n
}

function scan_python(   i, s, last, k, checked, exc_indent, exc_line, exc_body,
                        exc_open, ind) {
  exc_open = 0
  exc_indent = 0
  exc_line = 0
  exc_body = 0
  for (i = 1; i <= n; i++) {
    s = strip_comment(RAW[i])
    ind = indent_of(s)

    if (exc_open && trim(s) != "" && ind <= exc_indent) {
      if (exc_body > 0) {
        report(exc_line, "except-pass", "except handler swallows the error (pass/continue only)")
      }
      exc_open = 0
    }
    if (exc_open && trim(s) != "") {
      if (trim(s) ~ /^(pass|continue)$/) { exc_body++ }
      else { exc_body = -1000 }
      if (s ~ /(^|[^A-Za-z0-9_.])(sys\.exit|os\._exit)\([ \t]*0[ \t]*\)/ ||
          s ~ /(^|[^A-Za-z0-9_.])exit\([ \t]*0[ \t]*\)/) {
        report(i, "exit-zero", "except handler exits 0; the failure is invisible to the caller")
      }
    }
    if (s ~ /^[ \t]*except([ \t(:]|$)/) {
      exc_open = 1
      exc_indent = ind
      exc_line = i
      exc_body = 0
      continue
    }

    if (match(s, /subprocess\.run[ \t]*\(/)) {
      last = py_call_span(i, RSTART + RLENGTH - 1)
      if (CALL_TEXT !~ /check[ \t]*=[ \t]*True/) {
        checked = 0
        for (k = i; k <= last + 5 && k <= n; k++) {
          if (RAW[k] ~ /returncode/) { checked = 1 }
        }
        if (!checked) {
          report(i, "subprocess-no-check",
                 "subprocess.run() with neither check=True nor a returncode check")
        }
      }
    }
    if (s ~ /(^|[^A-Za-z0-9_.])(subprocess\.call|subprocess\.getoutput|os\.system)[ \t]*\(/ &&
        s !~ /[!=]=[ \t]*0/) {
      report(i, "unchecked-call",
             "this API returns a status that nothing forces you to read; use subprocess.run(check=True)")
    }
    if (s ~ /modal[ \t]+run/ && s ~ /\|\|/) {
      report(i, "modal-masked", "a failed 'modal run' must not report success (GPU spend)")
    }
  }
  if (exc_open && exc_body > 0) {
    report(exc_line, "except-pass", "except handler swallows the error (pass/continue only)")
  }
}

BEGIN {
  findings = 0
  split("missing-set-e missing-pipefail", fs_, " ")
  for (k_ in fs_) { FILESCOPED[fs_[k_]] = 1 }
  split("or-true set-plus-e exit-zero missing-set-e missing-pipefail " \
        "subprocess-no-check unchecked-call except-pass modal-masked " \
        "ci-continue-on-error bad-allow-directive", kn_, " ")
  for (k_ in kn_) { KNOWN[kn_[k_]] = 1 }
}

{ RAW[NR] = $0 }

END {
  n = NR
  scan_directives()
  if (kind == "py") {
    scan_python()
  } else {
    scan_shell_lines()
    if (kind == "sh") { scan_shell_file() }
  }
  exit(findings > 0 ? 1 : 0)
}
AWK_PROGRAM
)"

# ---------------------------------------------------------------------------
# File classification and discovery
# ---------------------------------------------------------------------------
detect_kind() {
  local path="$1" first=""
  case "$path" in
    *.sh | *.bash) printf 'sh'; return 0 ;;
    *.yml | *.yaml) printf 'yaml'; return 0 ;;
    *.py) printf 'py'; return 0 ;;
  esac
  if [[ -f "$path" && -r "$path" ]] && IFS= read -r first <"$path"; then
    case "$first" in
      '#!'*bash*) printf 'sh'; return 0 ;;
      '#!'*sh) printf 'sh'; return 0 ;;
      '#!'*sh\ *) printf 'sh'; return 0 ;;
      '#!'*python*) printf 'py'; return 0 ;;
    esac
  fi
  printf 'none'
}

is_fixture() {
  case "$1" in
    scripts/testdata/* | */scripts/testdata/*) return 0 ;;
    *) return 1 ;;
  esac
}

scan_one() {
  local path="$1" kind="$2" display="${3:-$1}"
  awk -v file="$display" -v kind="$kind" "$lint_awk" "$path"
}

list_repo_files() {
  if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$repo_root" ls-files --cached --others --exclude-standard
  else
    (cd -- "$repo_root" && find . -type f -printf '%P\n' | sort)
  fi
}

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
print_rules() {
  sed -n '/^# Rules (see --rules/,/^# Allowlisting\./p' "${BASH_SOURCE[0]}" |
    sed -e 's/^# \{0,1\}//' -e '$d'
}

scan_paths() {
  local -a paths=("$@")
  local rc=0 scanned=0 path kind rel out

  for path in "${paths[@]}"; do
    rel="$path"
    case "$rel" in
      "${repo_root}"/*) rel="${rel#"${repo_root}"/}" ;;
    esac
    if is_fixture "$rel"; then
      continue
    fi
    [[ -f "$path" ]] || continue
    kind="$(detect_kind "$path")"
    [[ "$kind" != "none" ]] || continue
    scanned=$((scanned + 1))
    if out="$(scan_one "$path" "$kind" "$rel")"; then
      continue
    fi
    printf '%s\n' "$out"
    rc=1
  done

  printf 'lint_exit_codes: scanned %d file(s)\n' "$scanned" >&2
  return "$rc"
}

scan_repo() {
  local -a files=()
  local f
  while IFS= read -r f; do
    files+=("${repo_root}/${f}")
  done < <(list_repo_files)
  [[ ${#files[@]} -gt 0 ]] || die "no files found under ${repo_root}"
  scan_paths "${files[@]}"
}

# Every fixture declares the rules it must trigger:
#   # tessera-lint-expect: <rule>
# A bad fixture passes the self-test only when the scanner reports exactly that
# set of rules, so a fixture cannot be "rejected for the wrong reason".
expected_rules() {
  sed -n 's/^[^A-Za-z0-9]*tessera-lint-expect:[ \t]*//p' "$1" | tr -s ' ' '\n' |
    sed '/^$/d' | sort -u | tr '\n' ' '
}

reported_rules() {
  sed -n 's/^[^:]*:[0-9]*: \[\([a-z-]*\)\].*$/\1/p' | sort -u | tr '\n' ' '
}

self_test() {
  local rc=0 f kind out status want got n_bad=0 n_good=0
  [[ -d "$fixture_dir" ]] || die "fixture directory missing: ${fixture_dir}"

  shopt -s nullglob
  local -a bad=("${fixture_dir}"/lint_bad_*) good=("${fixture_dir}"/lint_good_*)
  shopt -u nullglob

  [[ ${#bad[@]} -gt 0 ]] || die "no lint_bad_* fixtures in ${fixture_dir}"
  [[ ${#good[@]} -gt 0 ]] || die "no lint_good_* fixtures in ${fixture_dir}"

  for f in "${bad[@]}"; do
    n_bad=$((n_bad + 1))
    kind="$(detect_kind "$f")"
    want="$(expected_rules "$f")"
    [[ -n "$want" ]] || die "fixture ${f##*/} declares no 'tessera-lint-expect:' rule"
    status=0
    if out="$(scan_one "$f" "$kind" "${f##*/}" 2>&1)"; then
      status=0
    else
      status=$?
    fi
    got="$(printf '%s\n' "$out" | reported_rules)"
    if [[ "$status" -eq 0 ]]; then
      printf 'SELF-TEST FAIL %-34s accepted a bad fixture (want: %s)\n' "${f##*/}" "$want"
      rc=1
    elif [[ "$got" != "$want" ]]; then
      printf 'SELF-TEST FAIL %-34s reported [%s], expected [%s]\n' "${f##*/}" "${got% }" "${want% }"
      rc=1
    else
      printf 'SELF-TEST ok   %-34s rejected: %s\n' "${f##*/}" "${want% }"
    fi
  done

  for f in "${good[@]}"; do
    n_good=$((n_good + 1))
    kind="$(detect_kind "$f")"
    status=0
    if out="$(scan_one "$f" "$kind" "${f##*/}" 2>&1)"; then
      status=0
    else
      status=$?
    fi
    if [[ "$status" -ne 0 || -n "$out" ]]; then
      printf 'SELF-TEST FAIL %-34s rejected a good fixture:\n%s\n' "${f##*/}" "$out"
      rc=1
    else
      printf 'SELF-TEST ok   %-34s accepted\n' "${f##*/}"
    fi
  done

  printf 'lint_exit_codes: self-test %d bad + %d good fixture(s)\n' "$n_bad" "$n_good"
  return "$rc"
}

# ---------------------------------------------------------------------------
main() {
  local mode="scan"
  local -a paths=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --self-test) mode="self-test" ;;
      --rules) mode="rules" ;;
      -h | --help)
        sed -n '2,40p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//'
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

  case "$mode" in
    rules) print_rules ;;
    self-test) self_test ;;
    scan)
      if [[ ${#paths[@]} -gt 0 ]]; then
        scan_paths "${paths[@]}"
      else
        scan_repo
      fi
      ;;
  esac
}

main "$@"
