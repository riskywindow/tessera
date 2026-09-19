# ADR-002: Credits, and why the hot path is one atomic

- Status: proposed (HC-0 decides)
- Date: 2026-09-19
- Invariants: I-4 (hot path), I-5 (fail open), I-1 (cost estimates are measured, not assumed)
- Validated by: WP-0.5 T7 (no syscalls, no allocation); M2 gating accuracy; H7 (the preemption floor)

## Context

A running CUDA kernel cannot be preempted from userspace. The only moment
Tessera controls is *before* a launch is submitted. So the scheduling primitive
is admission control: decide, per launch, whether it goes now or waits.

That decision sits on the path of every kernel launch in every tenant. PyTorch
inference launches thousands of kernels per second per process, so the cost of
deciding has to be near zero when the answer is yes, which it is almost always.
A syscall, a lock, or an allocation per launch would show up directly in the
overhead budget that H2 measures.

## Decision

**One 4 KiB shared page per tenant**, created by `tesserad` under
`/dev/shm/tessera/<gpu>/<tenant>` and mapped `MAP_SHARED` by the shim. The
layout is `common/include/tessera/common/tenant_page.h`.

**The fast path is a single `fetch_sub`.** The shim estimates the cost `c` of
this launch in GPU microseconds, does `credits_us.fetch_sub(c)`, and if the
value it got back was at least `c`, forwards the launch. That is all: no
syscall, no lock, no allocation, no extra driver call. Credits are allowed to
go negative, because a launch commits its estimate before anyone knows the true
cost; the next refill repays the debt rather than losing it.

**The slow path is a bounded futex wait.** If the prior balance was short, the
shim waits on `futex_word` with a 1 ms timeout in a loop, re-checking credits
each time. The daemon bumps `futex_word` on every refill.

**Fail open is a timeout, not a heartbeat.** The page carries `daemon_epoch`
(bumped every tick) and `epoch_mono_ns` (the daemon's `CLOCK_MONOTONIC` stamp at
that tick). If the epoch has not advanced for 100 ms, the shim stops gating and
runs unthrottled (I-5). Carrying the timestamp as well as the counter means a
shim that maps the page mid-run can judge staleness immediately instead of
waiting to observe two ticks — which matters precisely in the case that
motivates I-5, a daemon that died before the tenant started.

**Costs are measured, not guessed.** The cost of a kernel key (`CUfunction` plus
grid dimensions, or `CUgraphExec`) comes from a per-process EMA table fed by
sampled event pairs: one launch in N per stream is bracketed by timing-enabled
events from a preallocated ring, and a background thread polls them with
`cuEventQuery` and never blocks the application. Unknown keys start at a default
and converge. Sampling, not instrumenting every launch, is deliberate: event
injection has a cost, and paying it per launch would distort the thing being
measured.

**Capture safety overrides gating.** While a stream is capturing, the shim does
not gate and does not inject events: doing either corrupts the graph or
deadlocks the capture. The cost is charged at `cuGraphLaunch` instead, with the
whole graph as the key.

### Deviations from the brief's sketch

| Brief | Here | Why |
|---|---|---|
| `uint32_t policy_flags; uint32_t stream_priority;` (plain) | atomic | They are written by the daemon and read by the shim in another process; plain loads of shared memory are a data race. Relaxed atomic loads cost nothing on x86-64 or arm64. |
| — | `magic`, `layout_version` | A shim must be able to reject a stale or foreign mapping rather than interpret arbitrary bytes as credits. |
| — | `epoch_mono_ns` | Staleness without needing two observations (above). |
| — | cache-line separation | Shim-written telemetry is kept off the line holding daemon-written credits, so refills do not invalidate the counter line on every tick. |

## Alternatives rejected

- **Ask the daemon per launch** (socket or shared ring with a wakeup). Correct
  and simple, but a syscall per launch violates I-4 and would dominate H2's
  overhead budget.
- **A semaphore per tenant.** `sem_wait` is a syscall under contention and gives
  no way to express cost-weighted admission, only counted admission.
- **Gate on kernel count rather than estimated time.** This is roughly what
  HAMi-core's rate limiting does, and it is simpler; it also means a tenant
  issuing long kernels and a tenant issuing short ones get wildly different GPU
  time for the same quota. Credits in microseconds is the unit the SLO
  controller actually needs.
- **Trust CUPTI timing for costs.** One subscriber per process, and an overhead
  profile unsuited to production use.

## Consequences

- Estimates are wrong at first, and wrong forever for kernels whose duration
  depends on data rather than on launch geometry. The controller in M3 closes a
  loop on measured interference precisely so that estimate error becomes a
  disturbance it rejects, not an error it accumulates.
- The effective preemption latency of this lever is bounded below by the longest
  in-flight kernel or graph of the throttled tenant. H7 measures that floor for
  the batch workload; it is the number that decides whether launch gating alone
  can hold a tail-latency SLO, and it is why the lever ladder exists.
- Negative credit makes a tenant's debt visible but also means a burst can
  overshoot by up to one kernel's cost per stream.

## What would overturn this

If H2 shows the fast path costs more than 1.0 µs at the median, the design is
too heavy and the estimate lookup (not the atomic) is the suspect. If H7 shows
the batch workload's kernels are long relative to the interactive tenant's SLO,
gating alone cannot deliver and the spatial levers stop being optional.
