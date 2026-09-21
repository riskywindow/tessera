# Decisions

Architecture decisions for Tessera. Each entry is a short form; the long form
lives in `docs/decisions/`. Per I-7, a scope change needs an ADR **before** the
code, and per the gate protocol, a gate criterion is never edited to make it
pass — an ADR explains why a criterion was wrong, and the human approves it at
the next checkpoint.

| ADR | Title | Status | Decided at |
|---|---|---|---|
| [ADR-001](decisions/ADR-001-interposition.md) | How Tessera gets between an application and the driver | **accepted** | HC-0 |
| [ADR-002](decisions/ADR-002-credits.md) | Credits, and why the hot path is one atomic | proposed | HC-0 |
| [ADR-003](decisions/ADR-003-deployment-modes.md) | Two deployment modes, both first-class | proposed | HC-0 |
| [ADR-004](decisions/ADR-004-driver-headers-and-variant-resolution.md) | Build the shim against CUDA 13.x, and let the driver pick the variant | **accepted** | HC-0 |

## ADR-001 — Interposition

Masquerade as `libcuda.so.1` on `LD_LIBRARY_PATH` (primary) and support
`LD_PRELOAD` plus a `dlsym` hook (secondary); M1's H1 picks the default. The
shim loads the real driver by absolute path from `TESSERA_REAL_LIBCUDA`, forwards
all 640+ exported symbols through lazily resolved trampolines, and hooks
`cuGetProcAddress(_v2)` because cudart resolves everything through it — including
`cuGetProcAddress` itself. A requested symbol's exact variant is identified by
comparing the real driver's returned pointer against the variants the shim
knows, so `cudaVersion` and the per-thread-default-stream flag are honoured
without a hand-maintained table; unknown variants pass through and increment a
visible `bypassed_lookups` counter.

## ADR-002 — Credits

One 4 KiB shared page per tenant. The launch fast path is a single
`fetch_sub` on `credits_us`: no syscall, no lock, no allocation (I-4). Shortfall
falls back to a bounded futex wait; a daemon epoch that stops advancing for
100 ms makes the shim fail open (I-5). Kernel costs come from an EMA over
sampled event pairs, never from instrumenting every launch. Gating is suspended
during stream capture and charged at `cuGraphLaunch` instead. Deviations from
the brief's struct sketch (atomic policy fields, a magic and layout version, an
epoch timestamp, cache-line separation) are listed in the ADR.

## ADR-003 — Deployment modes

`solo` (no MPS: gating and memory quotas) and `shared` (on top of MPS: adds
stream priorities, active-thread percentage, and green contexts if H6 permits)
are both first-class and both appear in the M3 matrix. The daemon advertises
which levers are active and refuses policies a mode cannot honour. Stream
priorities are assumed inert across processes without MPS; that assumption is
measured by H3 and H4 before anything depends on it.

## ADR-004 — CUDA 13.x headers, driver-decided variants

Decided by the human at HC-0. The shim builds against CUDA 13.x so it knows
every symbol and struct version the installed driver exports; the tenant's
toolkit stays pinned to whatever vLLM's wheel expects, because the two are
independent. `cuGetProcAddress` becomes resolve-then-substitute through a
pointer-keyed table, so variant selection is the driver's job. A load-time
`cuDriverGetVersion` check refuses to load, with a legible error, when the
driver is older than the headers — a deliberate and bounded exception to I-5,
which forbids hanging a tenant, not failing one loudly.

M0's measurements are why: the L4's driver reports CUDA 13000 while the shim
was built against 12060, and it exports 67 `cu*` symbols those headers do not
know.

## HC-0 decisions of record (2026-09-21)

1. **Delta: not approved as written**; replaced verbatim in `docs/RELATED.md`,
   and **H9** added to M1 as a hard hypothesis. The delta now rests on a
   bandwidth-pressure claim that H9 is allowed to falsify.
2. **ADR-001 accepted**, with the export list to be generated from the installed
   driver rather than the stub, and no driver calls from constructors. The
   platform wrinkle (the driver is only present when a GPU is attached) is
   recorded in that ADR.
3. **M1 budget stays $25.**
4. **Disk cleanup approved and done**: journald 2.1 GB to 165 MB, apt 600 MB
   freed, Hugging Face cache (10 repos, 4.9 GB) removed after listing. 4545 MB
   free; the human resizes the VM if it drops under 2 GB.
5. **The repository is public from day one**:
   `https://github.com/riskywindow/tessera`. **M0 cannot be GREEN until CI has
   run green once** — added to the M0 gate as G10.
6. **ADR-004**, above.
