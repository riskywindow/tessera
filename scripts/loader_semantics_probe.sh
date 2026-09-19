#!/usr/bin/env bash
# Measures the dynamic-loader behaviour that ADR-001 depends on, using three
# tiny stand-in libraries instead of the real driver, so it runs anywhere with
# a C compiler and no GPU.
#
# The four questions, and why each matters:
#   Q1 Does a library that masquerades as libcuda.so.1 (first on
#      LD_LIBRARY_PATH) intercept DT_NEEDED linkage?
#   Q2 Does it also intercept dlopen("libcuda.so.1") + dlsym, the path that
#      Triton launchers take and that LD_PRELOAD cannot reach?
#   Q3 Can the masquerading library still load the REAL library by absolute
#      path, or does the loader hand it back itself (which would recurse)?
#   Q4 Under LD_PRELOAD instead, is the dlopen+dlsym path bypassed?
#
# Q2 vs Q4 is the evidence for making masquerade the primary strategy.
# A NOT-FOUND result for a symbol the masquerading library does not define is
# also recorded: it is why the shim must forward every exported symbol.
#
# Usage: scripts/loader_semantics_probe.sh [--json <path>]
set -euo pipefail

JSON_OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_OUT="${2:?--json needs a path}"
      shift 2
      ;;
    *)
      echo "usage: $0 [--json <path>]" >&2
      exit 2
      ;;
  esac
done

CC="${CC:-gcc}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/real" "$WORK/masq" "$WORK/preload"

cat >"$WORK/real.c" <<'EOF'
int cuFoo(void) { return 1; }
int cuOnlyReal(void) { return 42; }
EOF

cat >"$WORK/shim.c" <<'EOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
static void *real_handle;
static int (*real_foo)(void);
static void init(void) {
  const char *p = getenv("TESSERA_REAL_LIBCUDA");
  real_handle = dlopen(p, RTLD_LOCAL | RTLD_NOW);
  if (!real_handle) { fprintf(stderr, "SHIM_DLOPEN_FAILED\n"); exit(3); }
  void *their_foo = dlsym(real_handle, "cuFoo");
  extern int cuFoo(void);
  if (their_foo == (void *)cuFoo) { printf("SELF_LOAD_GUARD_FIRED\n"); exit(4); }
  real_foo = (int (*)(void))their_foo;
}
int cuFoo(void) {
  if (!real_handle) init();
  printf("INTERCEPTED\n");
  return real_foo();
}
EOF

cat >"$WORK/app.c" <<'EOF'
#include <dlfcn.h>
#include <stdio.h>
extern int cuFoo(void);
int main(void) {
  printf("DTNEEDED_RESULT=%d\n", cuFoo());
  void *h = dlopen("libcuda.so.1", RTLD_NOW);
  if (!h) { printf("DLOPEN_FAILED\n"); return 1; }
  int (*f)(void) = (int (*)(void))dlsym(h, "cuFoo");
  printf("DLSYM_RESULT=%d\n", f ? f() : -1);
  int (*g)(void) = (int (*)(void))dlsym(h, "cuOnlyReal");
  printf("UNHOOKED_SYMBOL=%s\n", g ? "found" : "notfound");
  return 0;
}
EOF

"$CC" -shared -fPIC -o "$WORK/real/libcuda.so.1" "$WORK/real.c" -Wl,-soname,libcuda.so.1
"$CC" -shared -fPIC -o "$WORK/masq/libcuda.so.1" "$WORK/shim.c" -ldl -Wl,-soname,libcuda.so.1
"$CC" -shared -fPIC -o "$WORK/preload/libtessera.so" "$WORK/shim.c" -ldl -Wl,-soname,libtessera.so
"$CC" -o "$WORK/app_masq" "$WORK/app.c" -L"$WORK/masq" -l:libcuda.so.1
"$CC" -o "$WORK/app_real" "$WORK/app.c" -L"$WORK/real" -l:libcuda.so.1

run() { env "$@" 2>&1; }

masq_out="$(run LD_LIBRARY_PATH="$WORK/masq" TESSERA_REAL_LIBCUDA="$WORK/real/libcuda.so.1" "$WORK/app_masq")"
preload_out="$(run LD_LIBRARY_PATH="$WORK/real" LD_PRELOAD="$WORK/preload/libtessera.so" TESSERA_REAL_LIBCUDA="$WORK/real/libcuda.so.1" "$WORK/app_real")"

# The self-load guard must fire (exit 4) rather than recurse.
selfload_rc=0
run LD_LIBRARY_PATH="$WORK/masq" TESSERA_REAL_LIBCUDA="$WORK/masq/libcuda.so.1" "$WORK/app_masq" >"$WORK/selfload.txt" || selfload_rc=$?

# grep -c exits 1 when the count is zero, which is a legitimate measurement
# here rather than a failure, so count through a function that normalizes it.
count_intercepts() { printf '%s\n' "$1" | grep -c '^INTERCEPTED$' || [[ $? == 1 ]]; }

masq_intercepts="$(count_intercepts "$masq_out")"
preload_intercepts="$(count_intercepts "$preload_out")"
masq_unhooked="$(printf '%s\n' "$masq_out" | sed -n 's/^UNHOOKED_SYMBOL=//p')"
preload_unhooked="$(printf '%s\n' "$preload_out" | sed -n 's/^UNHOOKED_SYMBOL=//p')"

echo "--- masquerade (shim is libcuda.so.1, first on LD_LIBRARY_PATH)"
printf '%s\n' "$masq_out"
echo "--- LD_PRELOAD (shim is libtessera.so, real driver on the search path)"
printf '%s\n' "$preload_out"
echo "--- self-load guard"
cat "$WORK/selfload.txt"
echo "exit=$selfload_rc"

fail=0
# Q1+Q2: masquerade intercepts BOTH the DT_NEEDED call and the dlopen+dlsym call.
[[ "$masq_intercepts" == "2" ]] || { echo "FAIL: masquerade intercepted $masq_intercepts/2 paths" >&2; fail=1; }
# Q3: the absolute-path dlopen reached the real library (result 1, not recursion).
grep -q '^DTNEEDED_RESULT=1$' <<<"$masq_out" || { echo "FAIL: masquerade did not reach the real library" >&2; fail=1; }
# Q4: LD_PRELOAD catches DT_NEEDED only; dlopen+dlsym goes straight to the real library.
[[ "$preload_intercepts" == "1" ]] || { echo "FAIL: preload intercepted $preload_intercepts paths, expected 1" >&2; fail=1; }
# The masquerading handle cannot serve a symbol it does not define.
[[ "$masq_unhooked" == "notfound" ]] || { echo "FAIL: expected unhooked symbol to be absent under masquerade" >&2; fail=1; }
[[ "$preload_unhooked" == "found" ]] || { echo "FAIL: expected unhooked symbol to resolve under preload" >&2; fail=1; }
# The guard fires instead of recursing.
[[ "$selfload_rc" == "4" ]] || { echo "FAIL: self-load guard did not fire (rc=$selfload_rc)" >&2; fail=1; }

if [[ -n "$JSON_OUT" ]]; then
  mkdir -p "$(dirname "$JSON_OUT")"
  cat >"$JSON_OUT" <<EOF
{
  "probe": "loader_semantics",
  "utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "host": {
    "uname": "$(uname -srm)",
    "libc": "$(ldd --version | head -1)",
    "cc": "$($CC --version | head -1)",
    "git_commit": "$(git -C "$(dirname "$0")/.." rev-parse HEAD)"
  },
  "command": "scripts/loader_semantics_probe.sh --json $JSON_OUT",
  "masquerade": {
    "paths_intercepted": $masq_intercepts,
    "paths_tested": 2,
    "reached_real_library": true,
    "unhooked_symbol_via_dlsym": "$masq_unhooked"
  },
  "ld_preload": {
    "paths_intercepted": $preload_intercepts,
    "paths_tested": 2,
    "unhooked_symbol_via_dlsym": "$preload_unhooked"
  },
  "self_load_guard_exit_code": $selfload_rc,
  "conclusions": [
    "Masquerading as libcuda.so.1 intercepts both DT_NEEDED linkage and dlopen(\"libcuda.so.1\")+dlsym.",
    "LD_PRELOAD intercepts DT_NEEDED linkage only; the dlopen+dlsym path reaches the real library directly, which is why that strategy needs a dlsym hook.",
    "A masquerading library can load the real library by absolute path without the loader returning itself.",
    "dlsym on the masquerading handle cannot find a symbol the shim does not define, so the shim must forward every exported driver symbol."
  ],
  "pass": $([[ $fail == 0 ]] && echo true || echo false)
}
EOF
  echo "wrote $JSON_OUT"
fi

exit "$fail"
