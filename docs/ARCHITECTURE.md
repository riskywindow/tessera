# Architecture

Status markers are literal: **built** means it exists and its tests run in CI
configurations today; **planned (Mn)** means it is designed and scheduled. No
performance claims appear here. Numbers live in `docs/mechanisms.md` and
`results/`, each tied to a run.

## The problem, stated precisely

Several unmodified CUDA processes want one GPU. One of them is interactive and
has a tail-latency objective; the others are batch and want throughput. The
hardware gives userspace no way to preempt a kernel that has already been
submitted, and the stock driver's time-slicing has no policy interface. MPS
shares a context with static SM percentages; MIG partitions hardware statically
on datacenter parts only.

So Tessera schedules at the only point it controls: **the moment before a launch
is submitted**, plus whatever spatial partitioning the driver will grant it. The
design question is not "how do we preempt" (we cannot) but "what can be built
out of admission control, stream priorities, MPS thread percentages, and green
contexts, and how far does that get a p99".

## Shape

```
  tenant process (unmodified)                      tesserad (one per GPU)
  ┌─────────────────────────────┐                 ┌──────────────────────────┐
  │ PyTorch / vLLM / Triton     │                 │ epoll event loop         │
  │   └─ cudart ──┐             │                 │  ├─ control socket       │
  │ dlopen/dlsym ─┤             │                 │  │   registration,       │
  │ DT_NEEDED ────┤             │                 │  │   config, status      │
  │               ▼             │                 │  ├─ timerfd tick (1 kHz) │
  │      ┌──────────────────┐   │   shm page      │  │   credit refill       │
  │      │   libtessera     │◄──┼─────────────────┼──┤   per policy          │
  │      │  (libcuda.so.1)  │   │  credits_us     │  ├─ NVML telemetry       │
  │      └────────┬─────────┘   │  futex_word     │  └─ SLO controller (M3)  │
  │               │ forward     │  daemon_epoch   │      measured slowdown   │
  │               ▼             │  mem_quota      │        ↓ actuators       │
  │      real libcuda.so.1      │                 │      credit rate,        │
  └─────────────────────────────┘                 │      partition size      │
                                                  └──────────────────────────┘
```

The shim is the only thing in the tenant's address space. The daemon never sits
on the launch path; it only writes the shared page.

## The shim (`libtessera`) — planned (M0 WP-0.5)

**Interposition** is ADR-001. Summary: the shim is installed as `libcuda.so.1`
ahead of the real driver on `LD_LIBRARY_PATH`, so direct linkage,
`dlopen`+`dlsym`, and cudart's `cuGetProcAddress` all land on it; it loads the
real driver by absolute path. An `LD_PRELOAD` + `dlsym`-hook build exists as a
fallback. Symbols Tessera does not hook are forwarded through generic
trampolines, resolved lazily, never from a constructor.

**The launch path** is ADR-002. When credits suffice, a launch costs one
`fetch_sub` on shared memory and the forwarded driver call — no syscall, no lock,
no allocation (I-4, verified by test). When credits do not suffice, the shim
waits on a futex with a 1 ms timeout and re-checks, and if the daemon's epoch
stops advancing for 100 ms it stops gating entirely (I-5). A tenant whose daemon
died is a tenant that runs at full speed, never a tenant that hangs.

**Cost estimation** samples: one launch in N per stream is bracketed with
timing events from a preallocated ring, polled by a background thread with
`cuEventQuery`. The per-key EMA feeds the next launch's estimate.

**Memory virtualization** (M2) reports the tenant's quota through
`cuMemGetInfo_v2` and `cuDeviceTotalMem_v2`, so applications that size
themselves from free memory — vLLM's KV cache is the motivating case — fit the
quota by construction instead of crashing into it. Allocations past the quota
return `CUDA_ERROR_OUT_OF_MEMORY`, which PyTorch's caching allocator already
knows how to survive.

**Safety rails**: gating is suspended during stream capture and charged at
`cuGraphLaunch` instead; `pthread_atfork` handlers reset per-process state so a
forked dataloader cannot inherit a held lock.

## The daemon (`tesserad`) — planned (M2)

A single-threaded event loop: a unix control socket for registration and status,
a 1 kHz timerfd tick that refills credits according to policy, NVML telemetry,
and two tenant classes. `interactive` is protected: effectively unbounded
credits, highest stream priority, and its measured slowdown is the controller's
objective. `batch` is modulated: its credit rate and partition are the
actuators. Every actuation is logged.

The SLO controller (M3) starts as proportional-integral with anti-windup and a
deadband, and graduates only if oscillation is measured.

## Levers, and what each one can actually do

| # | Lever | Mode | Bounded by | Measured in |
|---|---|---|---|---|
| 1 | Launch gating | solo and shared | the longest in-flight kernel or graph of the throttled tenant | H7 |
| 2 | Stream priorities | shared only (assumed inert across contexts) | whether MPS honours cross-client priority | H3, H4 |
| 3 | MPS active thread % | shared | static per-client, set at tenant start | H5 |
| 4 | Green contexts | shared, if usable | SM granularity; address-space and MPS behaviour unknown | H6 |
| 5 | Memory quotas | solo and shared | applications that ignore free-memory queries | M2 |

The ladder exists because lever 1 has a floor: gating cannot make a kernel that
is already running finish sooner. If the batch tenant's kernels are long
relative to the interactive tenant's latency budget, spatial levers are the only
remaining answer. M1 measures that floor before M2 or M3 builds on it.

## Built today (M0)

- `common/include/tessera/common/tenant_page.h`: the shm contract, with layout
  rules enforced by `static_assert` and cross-process behaviour (shared mapping,
  credit debit, futex wake, staleness) covered by tests.
- The build system: nine configurations including ASan+UBSan, TSan, and arm64
  cross builds whose tests run under qemu.
- `scripts/fetch_cuda_headers.sh`: pinned, hash-verified CUDA headers and driver
  stub, fetched outside the repo.

## Deliberately out of scope (I-6)

One GPU, one node, Linux, CUDA 12.4+. No multi-GPU, no NCCL, no ROCm. A
Kubernetes device plugin is an M4 stretch and only with HC-3 approval.
