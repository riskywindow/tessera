# ADR-001: How Tessera gets between an application and the driver

- Status: proposed (HC-0 decides)
- Date: 2026-09-19
- Invariants: I-2 (unmodified tenants), I-3 (bitwise-identical outputs), I-4 (hot path), I-5 (fail open)
- Validated by: WP-0.5 tests T1–T7 on CPU; H1 (capture completeness) on GPU in M1

## Context

Tessera can only schedule what it can see. A CUDA application reaches
`libcuda.so.1` through three different paths, and a shim that misses any one of
them silently under-counts launches, which would make every later number wrong.

1. **Direct linkage.** The binary or one of its libraries has `DT_NEEDED
   libcuda.so.1` and calls `cuLaunchKernel` through the PLT.
2. **`dlopen` + `dlsym`.** Triton's launcher stubs, some PyTorch paths, and
   vLLM custom ops open the driver explicitly and look symbols up by name.
   `dlsym` on an explicit handle returns the symbol from *that object*, so
   `LD_PRELOAD` does not affect it. A preload-only shim is bypassed here.
3. **`cuGetProcAddress` / `cuGetProcAddress_v2`.** The CUDA runtime does not
   call the exported `cu*` symbols at all. It resolves function pointers once,
   by name plus a `cudaVersion` plus flags, and calls through those pointers
   afterwards — including resolving `cuGetProcAddress` itself this way. A shim
   that exports `cuLaunchKernel` but does not hook the lookup sees nothing from
   a cudart-based application, which is nearly all of them.

Two further complications are structural rather than incidental:

- **Versioned and per-thread variants are distinct symbols.** `cuMemAlloc` and
  `cuMemAlloc_v2` take different argument types; `cuLaunchKernel` and
  `cuLaunchKernel_ptsz` differ in how the default stream is resolved.
  `cuGetProcAddress` chooses between them using `cudaVersion` and the
  `CU_GET_PROC_ADDRESS_{LEGACY,PER_THREAD_DEFAULT}_STREAM` flags. Returning the
  wrong one is either a silent miscount or an ABI mismatch.
- **We do not own the whole API.** The CUDA 12.6.77 driver stub exports 640
  symbols on x86-64 and 641 on sbsa (measured with `nm -D` on
  `stubs/libcuda.so` from `cuda-driver-dev-12-6`). Tessera hooks a few dozen.
  Everything else still has to work.

## Decision

**Primary strategy: masquerade.** Tessera builds a shared object whose SONAME
is `libcuda.so.1`, installs it in a private directory, and `tessera run` puts
that directory first on `LD_LIBRARY_PATH`. Every path above then finds the shim
first: direct linkage resolves `DT_NEEDED libcuda.so.1` to it, `dlopen("libcuda.so.1")`
finds it by search path, and cudart's `cuGetProcAddress` is the shim's.

The shim loads the real driver itself, by absolute path, from
`TESSERA_REAL_LIBCUDA`, which `tessera run` discovers via `ldconfig`. Loading by
absolute path matters: glibc matches an already-loaded object by name or SONAME
only for non-path lookups, so an absolute path loads the real driver as a
separate object rather than finding the shim again. Guards:

- open with `RTLD_LOCAL | RTLD_NOW`, so the real driver's symbols do not enter
  the global scope and cannot interpose on the shim;
- refuse to proceed if the loaded object's `cuInit` is the shim's own, which
  catches a misconfigured `TESSERA_REAL_LIBCUDA` pointing back at the shim
  instead of recursing forever (test T6);
- if the real driver cannot be opened, return a CUDA error rather than
  aborting. Tessera may slow a tenant; it may never break one (I-5).

**Secondary strategy: `LD_PRELOAD` plus a `dlsym` hook.** For environments where
the search path cannot be controlled, the same object is also built as
`libtessera.so` for preloading, which additionally interposes `dlsym` (and
`dlvsym`) so that path 2 cannot escape. Both strategies are implemented in M0;
**M1's H1 measurement chooses the default**, and the loser stays supported.

The preload strategy has a known limitation worth stating plainly: a hooked
`dlsym` receives `RTLD_NEXT` requests whose meaning is relative to the *caller's*
object, and glibc computes that from the return address. Forwarding normally
would resolve relative to the shim instead. Tessera tail-calls the real `dlsym`
for non-intercepted lookups so the return address stays the caller's, and the
real `dlsym` is obtained with `dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34")` with a
fallback to the architecture's base version. This is one reason masquerade is
primary.

**Forwarding everything else.** The shim exports every symbol the real driver
exports. Symbols Tessera does not hook are generic assembly trampolines that
jump to the real function pointer, so no signature knowledge is needed; the
pointers are resolved lazily on first use, never from a static constructor
(a library masquerading as `libcuda.so.1` gets constructed at unpredictable
times, and calling the driver from there is how shims deadlock). The export list
is generated per architecture from the matching driver stub at configure time,
because the two lists genuinely differ.

**Hooked symbols** are ordinary typed C++ wrappers, compiled against `cuda.h`
with `__CUDA_API_VERSION_INTERNAL` defined so that both legacy and `_v2`/`_ptsz`
prototypes are visible and the header's renaming macros are disabled.

**Resolving a `cuGetProcAddress` request** proceeds by identity rather than by a
hand-maintained version table: the shim forwards the request to the real driver,
then compares the returned pointer against `dlsym` of each variant name it knows
for that base symbol. Whichever matches tells the shim exactly which ABI the
caller is about to use, so it returns the corresponding wrapper. If no known
variant matches — a newer driver returning a variant this build has never heard
of — the shim returns the real pointer unchanged and increments a
`bypassed_lookups` counter. That counter is a first-class signal: it is how H1
distinguishes "we captured everything" from "we captured what we knew about".

## Measured, not assumed

The loader behaviour this decision rests on was measured on the dev host with
three stand-in libraries and no GPU, by `scripts/loader_semantics_probe.sh`
(artifact: `results/m0/loader_semantics.json`). Four results:

| Question | Masquerade | `LD_PRELOAD` |
|---|---|---|
| `DT_NEEDED libcuda.so.1` call intercepted | yes | yes |
| `dlopen("libcuda.so.1")` + `dlsym` intercepted | **yes** | **no — reached the real library directly** |
| Real library loadable by absolute path without self-recursion | yes | yes |
| `dlsym` for a symbol the shim does not define | **not found** | found |

Row 2 is the empirical reason masquerade is primary: under `LD_PRELOAD`, the
`dlopen`+`dlsym` path — Triton's path — goes straight past the shim, which is
what a `dlsym` hook has to repair. Row 4 is the empirical reason the shim must
forward every exported symbol rather than only the hooked ones: when the shim
*is* the handle, a symbol it does not define does not exist as far as the
application is concerned. A self-load guard was also exercised: pointing
`TESSERA_REAL_LIBCUDA` at the shim itself exits through the guard instead of
recursing.

These are properties of glibc's loader, not of CUDA, so they hold before any
GPU is involved; H1 still has to confirm that the driver paths a real cudart
takes are the ones measured here.

## Alternatives rejected

- **CUPTI callbacks.** The profiling interface can observe and even block at API
  entry points, but there may be only one subscriber per process, so Tessera
  would take the slot that profilers need, and CUPTI's overhead is unsuited to a
  path on every launch. CUPTI is still used in `bench/` as an *independent*
  ground truth for H1, which is exactly the role it should have.
- **`LD_AUDIT`.** `la_symbind` sees PLT binding, which covers path 1 but not
  pointers handed out by `cuGetProcAddress`, which is the dominant path.
- **Preload-only.** Loses path 2, which is where Triton lives.
- **Patching the framework** (a PyTorch allocator plugin, a vLLM fork). Violates
  I-2, and would have to be redone per framework.
- **A modified driver or kernel module.** Out of scope (I-6) and not deployable
  as a drop-in.

## Consequences

- The shim must be exhaustive about exports, and its export list is tied to a
  driver version. A newer driver may export symbols this build does not, which
  would fail at symbol resolution; the mitigation is to generate from the newest
  stub we support and to keep `bypassed_lookups` visible. (Risk R-3.)
- Because cudart caches the pointers it resolves at init, anything Tessera fails
  to hook at lookup time is unhookable for that process's lifetime. There is no
  second chance, which is why T1 tests the lookup path explicitly.
- Static initialization order is a hazard we accept and mitigate by lazy init;
  T5 asserts that loading the shim causes zero driver calls.
- Stream capture is a correctness cliff: gating or injecting an event while a
  capture is active corrupts the graph. The shim checks capture state on every
  launch and never gates during capture, gating `cuGraphLaunch` instead.

## What would overturn this

H1 in M1 compares shim-counted launches against CUPTI-counted launches for a
PyTorch matmul loop, a Triton kernel, and vLLM decode with CUDA graphs enabled.
If the counts disagree and the gap is not explained by a known bypass, the
masquerade strategy is not sufficient and this ADR is revised before any policy
is built on top of it.
