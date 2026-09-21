# ADR-004: Build the shim against CUDA 13.x, and let the driver pick the variant

- Status: accepted at HC-0 (2026-09-21), decided by the human
- Date: 2026-09-21
- Supersedes: the CUDA 12.6 header pin in `infra/versions.lock` (for the shim only)
- Invariants: I-1, I-5 (the load-time refusal below is a deliberate, bounded
  interaction with "fail open"), I-7 (this ADR precedes the code)
- Validated by: H1 in M1, and the `bypassed_lookups` counter

## Context

M0 measured a three-way version split on the target platform:

| Component | Version | Source |
|---|---|---|
| Installed driver on the L4 | 580.95.05, reporting CUDA **13000** | `results/m0/platform.json` |
| Shim's headers | CUDA **12060** | `infra/versions.lock` |
| Tenant runtime (torch 2.10 wheel) | CUDA **12.8** bundled | same |

The gap is not theoretical: the driver exports **67 `cu*` symbols the 12.6
headers do not know** (`results/m0/shim_diag.json`). A symbol the shim does not
know is a symbol it cannot wrap, and under masquerade one it does not export at
all.

## Decision

**1. The shim builds against CUDA 13.x headers.** It should know every symbol
and struct version the installed driver exports. The shim's header pin and the
tenant's toolkit pin are now explicitly independent: the tenant keeps whatever
vLLM's wheel expects (12.8 today), because the shim sits at the driver ABI and
does not care what runtime the tenant brought.

**2. `cuGetProcAddress` is resolve-then-substitute, through a pointer-keyed
table.** Forward the query to the real driver, take the pointer it returns, and
look that pointer up in a table keyed by address to find our hook. Version and
per-thread-stream variant selection is then **the driver's job, not ours**: we
never decide which variant a `(symbol, cudaVersion, flags)` triple means, we
only recognise what the driver decided.

This is a refinement of what M0 shipped, which compared the returned pointer
against `dlsym` of each known variant name in a loop. The pointer-keyed table is
O(1), and it states the intent directly: the shim's job is recognition, not
version arithmetic.

**3. A load-time version check.** On initialisation, call `cuDriverGetVersion`
and refuse to load with a clear error if it is **below** the header version the
shim was built against.

## Consequences

- **The refusal is a deliberate, bounded exception to I-5.** I-5 says Tessera
  may never hang a tenant; it does not say Tessera must always run. Refusing at
  load with a legible error is a fast, loud failure, not a hang, and it is
  preferable to a shim that silently mis-parses a struct whose layout changed
  between driver versions. The error message must name both versions and the
  fix. This is the one place Tessera stops a tenant from starting, and it is
  written down here so it is never a surprise.
- **A driver newer than the shim is still not detected.** The check catches
  too-old drivers. Too-new drivers remain R-3, mitigated only by
  `bypassed_lookups` being visible.
- **Green-context entry points arrive.** 13.x headers include the
  `cuGreenCtx*`/`cuDevSmResourceSplitByCount` surface that M1's H6 and H9 need;
  M0 confirmed by `grep` and `nm` that `cuDevSmResourceSplit` (no `ByCount`)
  does **not** exist in 12.6 and that `CU_DEV_RESOURCE_TYPE_WORKQUEUE` is absent
  there, so some of what the docs describe was simply unreachable from the old
  pin.
- **`scripts/fetch_cuda_headers.sh` changes its pin**, and the fake driver's
  symbol table (generated from the same headers) grows. The fake must still
  export only symbols the real driver has, which M0 asserts as a test.

## What would overturn this

If a 13.x header set turns out to be incompatible with the jammy-based image's
toolchain, or if the pointer-keyed table proves unable to recognise a pointer
the driver returns (for instance an internal trampoline that no exported name
resolves to), the fallback is the M0 behaviour — pass through and count a
bypassed lookup — and H1 is where that would show up as a launch-count gap.
