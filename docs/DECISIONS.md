# Decisions

Architecture decisions for Tessera. Each entry is a short form; the long form
lives in `docs/decisions/`. Per I-7, a scope change needs an ADR **before** the
code, and per the gate protocol, a gate criterion is never edited to make it
pass — an ADR explains why a criterion was wrong, and the human approves it at
the next checkpoint.

| ADR | Title | Status | Decided at |
|---|---|---|---|
| [ADR-001](decisions/ADR-001-interposition.md) | How Tessera gets between an application and the driver | proposed | HC-0 |
| [ADR-002](decisions/ADR-002-credits.md) | Credits, and why the hot path is one atomic | proposed | HC-0 |
| [ADR-003](decisions/ADR-003-deployment-modes.md) | Two deployment modes, both first-class | proposed | HC-0 |

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
