# Risks

Owners are work packages. "Resolved by" names the test or measurement that
closes the risk; a risk is only closed when that artifact exists.

Status: `open`, `mitigated` (a mechanism is in place, not yet proven on
hardware), `closed` (an artifact proves it), `accepted` (we will live with it).

## Interposition and correctness

| ID | Risk | Impact | Owner | Mitigation | Resolved by | Status |
|---|---|---|---|---|---|---|
| R-1 | cudart resolves driver entry points through `cuGetProcAddress`, not the exported `cu*` symbols. A shim that exports symbols but does not hook the lookup sees nothing. | Every launch invisible; all scheduling inert. | WP-0.5 | Hook `cuGetProcAddress` and `cuGetProcAddress_v2`, including a lookup of `cuGetProcAddress` itself. | T1, then H1 | open |
| R-2 | `dlsym` on an explicit handle ignores `LD_PRELOAD`; Triton launchers and some PyTorch/vLLM ops take that path. | Silent under-counting for exactly the tenants we care about. | WP-0.5 | Masquerade as `libcuda.so.1` (ADR-001) so the handle itself is the shim; `dlsym` hook in the preload strategy. | T1, H1 | open |
| R-3 | A newer driver returns symbol variants this build's `cuda.h` 12.6 does not know. **Measured, not hypothetical:** the L4 target runs driver 580.95.05 reporting CUDA driver version **13000**, while the shim is built against **12060** headers, and torch 2.10 ships a 12.8 runtime on top. Three versions, none of them equal. | Launches pass through unhooked and uncounted. | WP-0.5 | Identify variants by pointer identity, pass unknown ones through, and expose `bypassed_lookups` so the gap is visible rather than silent. Decide at HC-0 whether to build the shim against CUDA 13 headers to match the driver. | H1, `results/m0/platform.json` | open |
| R-4 | `_ptsz`/`_ptds` variants are separate symbols. Missing them loses a subset of launches. | Under-counting proportional to how much of the app uses per-thread default streams. | WP-0.5 | Hook every variant; honour `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM`. | T2 | open |
| R-5 | Gating or injecting events during stream capture corrupts the graph or deadlocks the capture. | Tenant hangs or produces wrong results. Violates I-3 and I-5. | WP-0.5, M2 | Check capture state on every launch; never gate or inject while capturing; charge at `cuGraphLaunch`. | M2 capture-safety test | open |
| R-6 | Redirecting the legacy default stream changes implicit synchronization semantics. | Silent wrong answers in tenant workloads. | M2 | Either emulate legacy semantics with events or do not redirect; prove with bitwise outputs. | H6, I-3 suite | open |
| R-7 | A library masquerading as `libcuda.so.1` runs static initializers at unpredictable times; calling the driver from a constructor deadlocks or crashes. | Tenant fails to start. | WP-0.5 | Lazy initialization on the first hooked call; nothing in constructors. | T5 | open |
| R-8 | `pthread_atfork` handling is wrong and a forked child (PyTorch dataloaders) deadlocks on a shim lock held at fork time. | Tenant hangs. Violates I-5. | WP-0.5 | atfork handlers reset per-process state and locks. | T3 | open |
| R-9 | `dlsym(RTLD_NEXT)` in the preload strategy resolves relative to the shim rather than the caller. | Wrong symbol returned to an app that uses `RTLD_NEXT`. | WP-0.5 | Tail-call the real `dlsym` so the return address stays the caller's; documented limitation in ADR-001; masquerade is primary. | T1, T4 | open |
| R-10 | vLLM sizes its KV cache from free device memory at startup; unvirtualized `cuMemGetInfo` turns a quota into a crash instead of a smaller cache. | Quotas unusable for the headline workload. | M2 | Virtualize `cuMemGetInfo_v2` and `cuDeviceTotalMem_v2` to report the quota. | M2 quota test | open |
| R-11 | Event injection for cost sampling perturbs what it measures. | Overhead budget (H2) blown; costs biased. | M2 | Sample one launch in N per stream; poll with `cuEventQuery` on a background thread; never block the app. | H2, M2 | open |

## Found while building the shim (WP-0.5)

| ID | Risk | Impact | Owner | Mitigation | Resolved by | Status |
|---|---|---|---|---|---|---|
| R-21 | The `cuGetProcAddress` identity check assumes the pointer the driver returns belongs to the driver. Under masquerade the shim is in the application's global scope, so a driver not linked `-Bsymbolic` can return **the shim's own** wrapper (measured with the fake driver), and a driver can also return an internal address no exported name resolves to. | The third case is an unhookable symbol for the life of the process: launches pass through uncounted. | WP-0.5, M1 | The self-returned case is recognised and passed through. The unrecognised case increments `bypassed_lookups`, so the loss is visible rather than silent. | H1 on real hardware | open |
| R-22 | `src/native_dlsym.cc` and `src/dlsym_hook.cc` do raw ELF parsing and are compiled `-fno-sanitize=all`, because the first caller of an interposed `dlsym` in a sanitized process is ASan's own runtime during init (measured: instrumenting them killed every `clang-asan` preload run). | A memory error in the shim's trickiest code is invisible to the sanitizers that cover everything else. | WP-0.7 | Every preload test in every configuration exercises them; a human reviewer reads them by hand. | Adversarial review (WP-0.7) | open |
| R-23 | The ELF walk adds the load bias to a `DT_*` value below the object's base, a heuristic about whether ld.so relocated the dynamic section in place. Holds for glibc on x86-64 and aarch64 as measured; would break for an object loaded at a very low address. `_r_debug` is a glibc interface. | Preload strategy breaks on a non-glibc loader. | WP-0.5 | I-6 scopes Tessera to Linux/glibc; masquerade (the primary strategy) does not use this path. | accepted within I-6 | accepted |
| R-24 | `gcc-13` (the CI minimum-version job) has no `musttail`, so the `dlsym` forwarding path degrades from a tail jump to an ordinary call, and `RTLD_NEXT` then resolves relative to the shim. | A preload-strategy app using `RTLD_NEXT` gets the wrong symbol when built by the minimum compiler. | WP-0.5 | Documented in the source and README; masquerade is primary; clang 13+ and gcc 15+ emit the tail jump (verified in disassembly). | M1 default-strategy decision | open |
| R-25 | The exit-time stats file is written from a library destructor, so a tenant that calls `_exit()` or crashes leaves none. | Telemetry silently missing for crashed tenants; a test that reads it could hang or read stale data. | WP-0.5 | Tests let children exit normally. M2's daemon-side telemetry does not depend on it. | M2 | open |
| R-26 | Counting uses one atomic add on a shared cache line (three per hooked launch). Contention across many threads is unmeasured. | H2's overhead budget could be blown by contention rather than by logic. | M1 | Measure in H2 before M2 adds gating to the same path. | H2 | open |

## Mechanisms (assumptions that must not become claims)

| ID | Risk | Impact | Owner | Mitigation | Resolved by | Status |
|---|---|---|---|---|---|---|
| R-12 | Stream priorities do not cross context boundaries, so lever 2 is inert without MPS. | A whole lever silently does nothing; discovered in M3 instead of M1. | M1 | Say so in the docs now as an assumption; measure it. | H3, H4 | open |
| R-13 | Green contexts may not share the primary context's address space, may not work under MPS, and may have a coarse SM granularity. | Lever 4 unusable; the ladder loses its spatial rung. | M1 | No policy depends on green contexts before H6 returns. | H6 | open |
| R-14 | Launch gating's preemption latency is bounded below by the longest in-flight kernel or graph of the throttled tenant. | If batch kernels are long, gating alone cannot hold an interactive SLO. | M1 | Measure the distribution first, then choose levers. | H7 | open |
| R-15 | MPS may not start inside a Modal container, and `nvidia-smi compute-policy` may be unavailable. | H4/H5 unmeasurable there; `shared` mode untestable. | WP-0.6, M1 | Probe in M0 pre-flight; if unavailable, run those rows on a root-capable hourly instance and charge the same budget. Never skip a row silently. | H8, M0 platform probe | open |

## Environment (found during M0 recon)

| ID | Risk | Impact | Owner | Mitigation | Resolved by | Status |
|---|---|---|---|---|---|---|
| R-16 | The dev host's root filesystem is 100% full; only root-reserved blocks remain (about 4 GiB). The rest is the human's other projects, which are not ours to delete. | Builds fail, or worse, the machine's other work breaks. | orchestrator | Build trees on tmpfs via the `build/` symlink; apt archives to tmpfs; CUDA deps in `~/.cache` are about 8 MB; nothing outside the repo is deleted. Raised at HC-0. | HC-0 decision | open |
| R-17 | The dev host is x86-64, not the arm64 the brief assumed, so I-6's arm64 coverage has no native machine. | arm64 support asserted without evidence. | WP-0.3 | Cross-compile with gcc and clang and run the CPU tests under `qemu-aarch64`; a CI workflow targets native `ubuntu-24.04-arm` once a remote exists. | `results/m0/local_matrix.json` | mitigated |
| R-18 | The CI workflow cannot run: the repo has no git remote. | "CI is green" would be an unbacked claim. | orchestrator | The workflow is written but unexecuted, and said to be so. HC-0 asks whether to create a remote. | HC-0 decision | open |
| R-19 | GPU runs cost real money against a hard cap (I-8). An agent could leave a container running. | Budget burned with nothing to show. | WP-0.6 | Every Modal function has a timeout; no `--detach`; a spend ledger line per launch; `modal app list` checked after each run. | `results/spend/ledger.jsonl` | mitigated |
| R-20 | Numbers quoted from prior art can drift into sounding like ours (I-1). | The project's credibility, which is the whole deliverable. | WP-0.1 | Independent verification of every claim in `RELATED.md`; paper numbers attributed inline; no Tessera numbers before they are measured. | `docs/RELATED.md` Method section | open |
