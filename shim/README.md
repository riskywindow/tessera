# `libtessera`: the CUDA driver-API shim

Tessera can only schedule what it can see. This library is how it sees: it gets
between an unmodified CUDA application and `libcuda.so.1`, forwards everything,
and counts the calls that matter. Version 0 **only** forwards and counts —
there is no gating yet, that is M2 — so what has to be right now is
*interception completeness* and *safety*, because M1's measurements and every
later milestone stand on both.

The strategy and the reasoning behind it are `docs/decisions/ADR-001-interposition.md`.
This file is how the code implements it.

## Build outputs

```
<build>/shim/libtessera.so             SONAME libtessera.so    (LD_PRELOAD)
<build>/shim/masq/libcuda.so.1         SONAME libcuda.so.1     (masquerade)
<build>/shim/masq/libcuda.so        -> libcuda.so.1
```

`masq/` contains nothing else, because `tessera run` puts exactly that
directory first on `LD_LIBRARY_PATH`; anything else in it would be on the
application's library path too. Both libraries are linked from one set of
objects. The only difference is `src/dlsym_hook.cc`, which is compiled into the
preload copy alone, and the version script: `exports.map` exports `cu*` and
`tessera_*`, `exports_preload.map` is the same list plus `dlsym` and `dlvsym`.
A masquerading `libcuda.so.1` that also exported `dlsym` would interpose on
every `dlsym` call in the process for no reason.

CMake targets: `tessera::shim`, `tessera::shim_masq`, and `tessera::shim_api`
for the header alone.

## 1. Forwarding everything

When the shim *is* `libcuda.so.1`, a symbol it does not define does not exist as
far as the application is concerned — that was measured, not assumed
(`results/m0/loader_semantics.json`, row 4). So the shim exports every symbol
the real driver exports: 640 on x86-64 and 641 on sbsa in CUDA 12.6.77, and the
two lists genuinely differ.

`tools/gen_symbols.sh` reads that list with `nm` from the driver stub for the
**target** architecture at configure time, and writes three files into
`<build>/shim/generated/`:

| File | Contents |
| --- | --- |
| `tessera_symbol_index.h` | `TESSERA_SYM_<name>` — the index each symbol's counter and slot live at, and `TESSERA_SYMBOL_COUNT` |
| `tessera_symbols.inc` | `TESSERA_SYMBOL(index, "name", hooked)`, sorted by name so a lookup can binary-search |
| `tessera_trampolines.inc` | one `TESSERA_TRAMP name, index` per **unhooked** symbol |

Missing `nm` fails the configure loudly: without an export list there is no
forwarding. The index space is the union of the stub's exports and the hooked
names, so a hooked symbol a given stub does not export still has an index (and
the configure says so).

Unhooked symbols are generic assembly trampolines (`src/trampoline_x86_64.S`,
`src/trampoline_aarch64.S`) that need no knowledge of any signature, because
they never touch the arguments. Each one:

1. counts the call with a single atomic add on its own slot of
   `tessera_shim_counts` — which is what makes `forwarded_unhooked` a per-call
   number rather than a count of resolutions;
2. loads its entry in `tessera_shim_slots` and, if it is set, **tail-jumps**
   there: no frame, no call, the driver returns straight to the application;
3. otherwise saves every register that can carry an argument (the integer
   argument registers, `%rax` / `x8`, and `%xmm0-7` / `v0-v7`), calls the
   resolver with its index, restores them, and tail-jumps to whatever comes
   back. The resolver always returns something callable, so this path cannot
   fall through.

Both arrays are hidden symbols in the same object, which is what lets the
trampolines reach them with a PC-relative reference inside a shared library.
CMake picks exactly one trampoline source per architecture and fails the
configure on an architecture that has none — an unforwarded build must not be
possible to produce by accident.

## 2. Hooked symbols

`src/hook_table.inc` is the one source of truth: exported name, base name,
counter kind, parameter list, argument list. It is `#include`d twice with
`TESSERA_WRAP` defined differently — once in `hooks.cc` to *define* the
wrappers, once in `proc_address.cc` to build the variant table — so a hooked
symbol cannot be in one and not the other.

The signatures are the ones `cuda.h` declares with `__CUDA_API_VERSION_INTERNAL`
defined (see `src/cuda_api.h`), which is how the driver's own sources see the
header: every variant under its real exported name, and the renaming macros
off. The compiler therefore checks all 84 signatures against NVIDIA's own
declarations — a wrong parameter list is a build error, not a silent ABI
mismatch that would corrupt a tenant's arguments.

What is hooked: the launch family including `cuLaunch`/`cuLaunchGrid`/
`cuLaunchGridAsync`; the memory family in **both** v1 and `_v2` forms, because
a quota that only covers `_v2` is a quota with a hole; the virtual-memory API;
stream creation, destruction and capture; `cuCtxCreate` v1 through v4;
`cuDevicePrimaryCtxRetain`; event records; the asynchronous copy family; and
`cuGetProcAddress(_v2)`. To add one, add a row to `hook_table.inc` — nothing
else, unless it needs behaviour of its own, in which case it becomes a
`TESSERA_HOOK_MANUAL` row and a hand-written wrapper.

## 3. `cuGetProcAddress` is the main event

cudart does not call the exported `cu*` symbols at all. It resolves function
pointers once, by name plus a `cudaVersion` plus flags — including resolving
`cuGetProcAddress` itself that way — and calls through those pointers
afterwards. Hook the lookup, or the shim sees nothing from a cudart
application, which is nearly all of them.

The shim answers by **identity**, never by a hand-maintained version table:

1. forward the request to the real driver;
2. compare the pointer it returned against the real address of each variant
   this build knows for that base name (resolved once, in `on_driver_loaded`);
3. whichever matches tells us exactly which ABI the caller is about to use, so
   return the corresponding wrapper.

`cudaVersion` and the `CU_GET_PROC_ADDRESS_{DEFAULT,LEGACY_STREAM,PER_THREAD_DEFAULT_STREAM}`
flags are honoured by construction: the driver already applied them when it
chose what to return. The `_v2` query result is the driver's and is passed
through untouched. Answers are cached in a lock-free direct-mapped table keyed
by the returned pointer; each entry is a single pointer to an immutable row, so
it is published with one atomic store and there is no lock for `fork()` to
catch.

Two outcomes are not a match:

- **The driver handed back one of our own wrappers.** Not a bypass —
  interposition working. Under masquerade the shim is the object called
  `libcuda.so.1` and sits in the application's global scope, so the real
  driver's reference to its *own* exported symbol resolves to us unless it was
  linked with `-Bsymbolic`. Measured here: with the shim as `libcuda.so.1`, the
  fake driver's `tf_symbol_address("cuLaunchKernel")` returns the shim's
  wrapper, not its own entry point. The pointer goes back unchanged and the
  caller ends up in the wrapper anyway.
- **Nothing matches at all** — a newer driver returning a variant this build
  has never heard of. The **real** pointer goes back unchanged and
  `bypassed_lookups` is incremented. Guessing at an ABI would corrupt the
  tenant's arguments; under-counting is recoverable and visible, and that
  counter is how H1 tells "we captured everything" from "we captured what we
  knew about".

## 4. The `dlsym` hook (preload build only)

`dlsym` on an explicit handle returns the symbol from *that* object, so
`LD_PRELOAD` does not affect it — measured in
`results/m0/loader_semantics.json`, row 2, which is the empirical reason
masquerade is primary and this hook is the repair.

`dlsym` and `dlvsym` are interposed. A lookup of a hooked symbol on a real
handle returns the shim's wrapper, but only when the object really provides the
symbol: a handle that does not have `cuLaunchKernel` must not suddenly appear
to. Everything else is **tail-called** to the real `dlsym`, so the return
address stays the caller's and `RTLD_NEXT` keeps its caller-relative meaning.

Residual limitations, stated plainly:

- The tail call is forced with the `musttail` attribute where the compiler has
  it (clang 13+, gcc 15+; both local toolchains do). On an older compiler —
  gcc 13 in the minimum-version CI job — it degrades to an ordinary call, and
  an application's `RTLD_NEXT`/`RTLD_DEFAULT` lookups would then resolve
  relative to `libtessera.so` instead of the caller.
- Requests with `RTLD_NEXT` or `RTLD_DEFAULT` are always passed straight
  through, never intercepted, precisely because we cannot answer them correctly
  from inside the hook.

Getting hold of the real `dlsym` is its own problem, solved in
`src/native_dlsym.cc`. Calling `dlvsym` by name would land back in our own
`dlvsym`, and calling anything a sanitizer interposes on (`dl_iterate_phdr`,
`dladdr`, `strcmp`) would run an interceptor before ASan has finished
initialising, because the first caller of an interposed `dlsym` in a sanitized
process is the sanitizer's runtime, not the application. Both failures were
observed, not theorised. So the bootstrap reads the loader's own structures —
`_r_debug.r_map` for the link map, `_DYNAMIC` to know which entry is us — and
does one GNU-hash lookup of `"dlvsym"` by hand. From there it is ADR-001's
recipe verbatim: `dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34")`, falling back to
the architecture's base version (`GLIBC_2.2.5` on x86-64, `GLIBC_2.17` on
aarch64). `native_dlsym.cc` and `dlsym_hook.cc` are the two files compiled
`-fno-sanitize=all` for exactly this reason. **Every internal lookup in the
shim goes through this pointer**, never through the interposed `dlsym`, or the
shim would resolve its own wrappers as the "real" driver and recurse.

## 5. Lifecycle and safety

- **Lazy init.** Nothing runs from a static constructor. A library masquerading
  as `libcuda.so.1` is constructed at times we do not control, and calling the
  driver from there is how shims deadlock. The first hooked call or trampoline
  hit initialises the shim.
- **Loading the driver.** `TESSERA_REAL_LIBCUDA` (absolute path), with
  `RTLD_LOCAL | RTLD_NOW`: local so the driver's symbols cannot interpose on
  us, now so a broken driver surfaces here rather than at the first forwarded
  call. If the variable is unset, the documented fallback is
  `/usr/lib/<multiarch>/libcuda.so.1`, `/usr/lib64/libcuda.so.1`,
  `/usr/lib/libcuda.so.1`, then the bare SONAME. An explicit
  `TESSERA_REAL_LIBCUDA` that does not work is **not** followed by a fallback:
  a misconfiguration should be loud.
- **Self-load guard.** If the object we opened is this very library — which is
  what a `TESSERA_REAL_LIBCUDA` pointing at the shim produces, and what the
  bare-SONAME fallback finds under masquerade — the shim refuses it rather than
  recursing. The check compares the module its `cuInit` lives in against our
  own, so it holds whatever the file was called.
- **Fail open (I-5).** Every failure returns a CUDA error and never aborts:
  `CUDA_ERROR_NOT_INITIALIZED` when there is no driver,
  `CUDA_ERROR_NOT_FOUND` when the driver does not export the symbol. A
  trampoline with nothing to jump to jumps to a stub that returns one of those.
- **Fork.** `pthread_atfork` handlers take the shim's one lock before the fork
  and release it on both sides, so the child can never inherit it held by a
  thread that did not survive. The child resets its counters and does not write
  the exit stats file. Everything else in the shim is lock-free by
  construction, including the `dlsym` bootstrap and the `cuGetProcAddress`
  cache, so there is nothing else for a fork to catch.
- **No exceptions cross a `cu*` entry point.** The entry points cannot be
  marked `noexcept` — `cuda.h` declares them without it and in C++17 that is
  part of the type — so the whole library is compiled `-fno-exceptions`.

## 6. The hot path (I-4)

Counting is O(1) with no string hashing, no map lookup, no lock, no allocation
and no syscall. Each wrapper holds a **compile-time** index into a fixed array
of relaxed atomics, and that is the entire cost. In a release build,
`cuLaunchKernel` compiles to three `lock addq` at fixed addresses, a load of
its slot, a test, and a tail `jmp` to the driver. M2 adds gating here and
nowhere else.

`forwarded_unhooked` is deliberately not a field: it is the sum of the
per-symbol counters of the symbols this build does not hook, computed when
stats are read, which keeps the trampoline fast path to a single atomic add on
one cache line instead of two.

## 7. Observability

`include/tessera/shim/stats.h` is the ABI other components code against:

```c
int      tessera_abi_version(void);
int      tessera_get_stats_v1(tessera_stats_v1* out);   /* launches,
                                                           graph_launches,
                                                           hooked_calls,
                                                           forwarded_unhooked,
                                                           bypassed_lookups */
uint64_t tessera_get_symbol_count(const char* exported_symbol_name);
void     tessera_reset_stats(void);
int      tessera_dump_stats(const char* path);          /* JSON */
```

`launches` counts launch-API calls; **graph launches are counted separately**,
because a graph launch is one call but many kernels, which would otherwise
corrupt the G9 comparison. Per-symbol counts are per *exported variant*:
`cuLaunchKernel` and `cuLaunchKernel_ptsz` are counted apart, as the driver's
own entry points are, and they work for forwarded symbols too.

Environment:

| Variable | Meaning |
| --- | --- |
| `TESSERA_REAL_LIBCUDA` | absolute path of the real driver to forward to |
| `TESSERA_STATS_FILE` | where to write `tessera_dump_stats` JSON at process exit — how a test whose application runs in a child process gets the child's counters |
| `TESSERA_LOG` | non-empty and not `0`: lifecycle events to stderr. Never logged from the per-launch path |

## 8. What v0 does not do

No gating, no quotas, no credits, no communication with `tesserad`: v0
forwards and counts. The capture hooks are wired up and counted but do not act,
which is where M2's "never gate during capture" rule will live.

## Tests

`tests/shim/smoke_*` are the shim's own smoke tests: it loads under both
strategies, forwards, and the counters move. The full acceptance matrix
(T1–T7 in `docs/gates/M0.md`) is written independently.

```
cmake --preset gcc-debug && cmake --build --preset gcc-debug -j3 && ctest --preset gcc-debug
cmake --preset clang-asan ...    # ASan + UBSan
cmake --preset clang-tsan ...    # TSan
cmake --preset arm64-gcc  ...    # cross build, tests under qemu-aarch64
```

`shim_smoke_fork_test` is a binary of its own because ThreadSanitizer kills a
child that starts threads after a multi-threaded fork unless `die_after_fork`
is off, and recreating its worker thread in the child is exactly what the fake
driver does on purpose.
