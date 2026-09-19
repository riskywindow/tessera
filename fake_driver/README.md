# `libcuda_fake`: a CPU-only CUDA driver

The development host has no GPU. Everything Tessera's shim does before it
reaches hardware — intercepting `cu*` symbols, resolving them through
`cuGetProcAddress`, telling a `_ptsz` entry point from its legacy twin,
surviving `fork()`, forwarding what it does not hook — has to be provable
without one, deterministically, under ASan/UBSan and TSan, on x86-64 and
arm64. `libcuda_fake` is that substrate: a simulation of the driver subset the
shim touches, with a control API (`tessera_fake/control.h`) that lets a test
drive the simulated GPU and read back exactly what the "application" did.

Its fidelity to the real driver's *semantics* matters more than its
completeness. Where it deliberately differs, the difference is listed under
[Deviations](#deviations).

## Build outputs

```
<build>/fake_driver/libcuda_fake.so              SONAME libcuda_fake.so
<build>/fake_driver/fake_masq/libcuda.so.1       SONAME libcuda.so.1
<build>/fake_driver/fake_masq/libcuda.so      -> libcuda.so.1
```

Both are built from one set of objects. `fake_masq/` contains nothing else, so
a test can put exactly that directory on `LD_LIBRARY_PATH` and have
`dlopen("libcuda.so.1")` — or a `DT_NEEDED libcuda.so.1` — find the fake and
only the fake. The two copies are separate shared objects with separate state:
a process loads one or the other, never both.

The exported symbol set is the `cu*` entry points listed in
`src/table.inc` plus the `tf_*` control API. Everything else is hidden. The
linker version script is *generated* from `src/table.inc` at configure time, so
the library cannot export a driver symbol that `cuGetProcAddress` does not know
about, or know about one it does not export. The version node is anonymous:
like the real `libcuda.so.1`, the exported symbols carry no version tag, so a
binary linked against the masquerading copy can later run against the real
driver or against Tessera's shim.

Every entry point in the table is a symbol the real CUDA 12.6 driver also
exports; that was checked against `stubs/libcuda.so` from
`cuda-driver-dev-12-6`.

## Using it

```c
#include <cuda.h>                        /* the normal driver API          */
#include <tessera_fake/control.h>        /* tf_* simulation control        */
#include <tessera_fake/cuda_variants.h>  /* named _ptsz/_ptds entry points */
```

CMake: link `tessera::cuda_fake` (or `tessera::cuda_fake_masq` for the
masquerading copy). Both carry the include directory.

A minimal test:

```c
tf_reset();                              /* known state                    */
cuInit(0);
cuDeviceGet(&dev, 0);
cuCtxCreate(&ctx, 0, dev);
tf_set_time_mode(1);                     /* manual clock: deterministic    */

CUfunction k = tf_register_kernel("add", add_body, /*duration_us=*/100);
tf_set_kernel_param_count(k, 3);

cuStreamCreate(&s, 0);
cuLaunchKernel(k, 1,1,1, 1,1,1, 0, s, params, NULL);
cuStreamSynchronize(s);                  /* jumps the clock to +100 us     */
```

## What is simulated

**Device.** One device, reported as `Tessera Fake Device` with 16 GiB of memory,
58 SMs and compute capability 8.9 by default. `tf_set_device_total_mem` and
`tf_set_device_sm_count` change the first two; the memory size is a real
budget, so an allocation past it fails with `CUDA_ERROR_OUT_OF_MEMORY` and
`cuMemGetInfo` tracks what is outstanding.

**Contexts.** A per-thread context stack, as CUDA has: `cuCtxCreate` pushes,
`cuCtxPushCurrent`/`cuCtxPopCurrent` push and pop, `cuCtxSetCurrent` replaces
the top. The primary context is reference counted. Calls that need a context
and have none return `CUDA_ERROR_INVALID_CONTEXT`; calls before `cuInit` return
`CUDA_ERROR_NOT_INITIALIZED` (except `cuGetProcAddress`, `cuDriverGetVersion`
and the error-string functions, which work before initialisation on a real
driver too, because that is how cudart bootstraps).

**Streams.** FIFOs. Each operation occupies its stream from
`max(now, stream's tail)` for its duration, so work on one stream completes in
submission order and work on separate streams overlaps. A single worker thread
completes operations in deadline order and runs their effects. `cuStreamWaitEvent`
pushes the waiting stream's tail out to the event's completion time.

**Default streams.** Each context has a legacy default stream; each *thread*
gets its own per-thread default stream in each context, created on first use. A
`NULL` stream argument resolves to the legacy one at a legacy entry point and to
the per-thread one at a `_ptsz`/`_ptds` entry point;
`CU_STREAM_LEGACY` and `CU_STREAM_PER_THREAD` override that either way.
`tf_last_stream_was_per_thread_default()` reports which kind the most recent
call resolved, which is how a test proves which entry point ran.

**Time.** Two modes, chosen with `tf_set_time_mode`:

* *scaled-real* (the default): simulated microseconds track `CLOCK_MONOTONIC`
  one for one, and a blocking call really blocks.
* *manual*: the clock stands still until `tf_advance_time_us` moves it, and a
  blocking call (`cuStreamSynchronize`, `cuCtxSynchronize`,
  `cuEventSynchronize`, any synchronous copy or fill) **jumps the clock forward
  to the moment the work it waits for completes** instead of waiting on the
  wall clock. That is what makes a test both deterministic and fast: a five
  simulated second kernel is synchronized in microseconds of real time. Most
  tests should use manual mode.

  `tf_advance_time_us(0)` is the useful special case: it does not move the
  clock, it just waits for the worker to catch up with it, which makes a
  zero-duration operation observable without synchronizing anything.

**Memory.** Device allocations are real host buffers, 256-byte aligned and
zeroed, and `CUdeviceptr` *is* the host address. A simulated kernel is a CPU
callback that really computes into them, so a test can compare outputs bit for
bit — which is how invariant I-3 ("a tenant's outputs are identical with and
without Tessera") gets tested on a machine with no GPU. `cuMemcpy*` and
`cuMemset*` do the obvious thing and are ordered on their stream; the device
side of every copy and fill is range-checked against a live allocation, so a
stray pointer is rejected the way a real driver rejects it.

The virtual-memory API (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`,
`cuMemUnmap`, `cuMemSetAccess`, `cuMemRelease`, `cuMemAddressFree`) uses real
`mmap`: a reservation is a `PROT_NONE` mapping and `cuMemMap` maps a `memfd`
over it, so two mappings of one allocation really do alias. The allocation stays
alive until both the user's reference and every mapping are gone, as on a real
driver.

**Events.** `cuEventRecord` queues a zero-duration marker; `cuEventQuery`
returns `CUDA_ERROR_NOT_READY` until the stream reaches it,
`cuEventSynchronize` waits for it, and `cuEventElapsedTime` reports the
difference between two completed records in simulated milliseconds.

**Kernels.** `tf_register_kernel(name, body, duration_us)` returns a
`CUfunction`. The body runs on the worker thread when the launch completes and
receives the `kernelParams` array the application passed.
`cuModuleLoadData` accepts any image; `cuModuleGetFunction` returns the kernel
registered under the requested name, or creates one on demand with a null body
and the duration set by `tf_set_default_kernel_duration_us`.

**Graphs.** Minimal and deliberately so: `cuStreamBeginCapture` records the
operations submitted to a stream instead of running them, `cuStreamEndCapture`
hands back the list as a `CUgraph`, and `cuGraphLaunch` replays the list onto a
stream. Capturing the legacy default stream is refused and synchronizing a
capturing stream returns `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`, both as on a
real driver.

**Fork.** `pthread_atfork` handlers hold the simulation lock across `fork()`, so
the child never inherits it locked by a thread that no longer exists. The worker
thread does not survive fork; the child abandons it (it is never joined) and
creates a new one on its next call, so the child's first `cuStreamSynchronize`
completes instead of waiting forever on work nobody is left to do. The child's
per-symbol counters start at zero and the child does not write the exit stats
file. With `tf_set_fork_mode(1)` every `cu*` call in a forked child instead
returns `CUDA_ERROR_NOT_INITIALIZED`, the way a real CUDA context becomes
unusable across fork.

## `cuGetProcAddress`

`cuGetProcAddress` and `cuGetProcAddress_v2` resolve `(symbol, cudaVersion,
flags)` against the single table in `src/table.inc`:

* the table is keyed by **base name** (`"cuMemAlloc"`), as the real driver's
  is; asking for an exported variant name (`"cuMemAlloc_v2"`) is a
  `SYMBOL_NOT_FOUND`;
* among the rows for that base name whose stream kind matches the flags, the
  newest row whose introduction version is at most `cudaVersion` wins — so
  `"cuMemAlloc"` at 3000 is the v1 entry point and at 3020 or later is
  `cuMemAlloc_v2`, and `"cuCtxCreate"` walks v1 → `_v2` → `_v3` → `_v4` at
  2000, 3020, 11040 and 12050;
* `CU_GET_PROC_ADDRESS_LEGACY_STREAM` selects the legacy entry point and
  `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM` the `_ptsz`/`_ptds` one;
  `CU_GET_PROC_ADDRESS_DEFAULT` means legacy, because it is `cuda.h`, not the
  driver, that turns a per-thread-default-stream compilation into the explicit
  flag. Passing both flags is `CUDA_ERROR_INVALID_VALUE`;
* the `_v2` form sets `CUdriverProcAddressQueryResult` to `SUCCESS`,
  `SYMBOL_NOT_FOUND` (no such base name, or none with the requested stream
  kind) or `VERSION_NOT_SUFFICIENT` (the base name exists but every entry point
  for it is newer than `cudaVersion`), and returns `CUDA_ERROR_NOT_FOUND` for
  the latter two;
* the version `tf_set_driver_version` reports is a ceiling: an entry point
  introduced after it cannot be resolved, however new a `cudaVersion` the
  caller asks for;
* **looking up `"cuGetProcAddress"` itself works**, and returns the v1 entry
  point below CUDA 12000 and `cuGetProcAddress_v2` at or above it. That is what
  cudart does, and a shim that does not hook it sees nothing.

The introduction versions are not invented. Each one is the
`PFN_<base>_v<version>` typedef CUDA 12.6's `cudaTypedefs.h` declares for that
entry point.

The table is queryable through the control API — `tf_symbol_table_size`,
`tf_symbol_name_at`, `tf_symbol_base_name_at`, `tf_symbol_version_at`,
`tf_symbol_stream_kind_at`, `tf_symbol_index`, `tf_symbol_address` — so a test
can enumerate every row rather than spot-check a few, which
`tests/fake_driver/proc_address_test.cc` does.

## Control API

Declared and documented in `fake_driver/include/tessera_fake/control.h`. In
brief:

| Function | What it does |
| --- | --- |
| `tf_reset()` | Tears down and rebuilds the simulation. Call it at the start of every test. |
| `tf_symbol_count(name)` | Calls to one **exported variant** (`"cuLaunchKernel_ptsz"` is counted apart from `"cuLaunchKernel"`). |
| `tf_dump_stats(path)` | Writes `{"symbol": count, ...}`; also written at exit to `$TESSERA_FAKE_STATS_FILE`. |
| `tf_register_kernel(name, body, us)` | Registers a named kernel with a CPU body and a duration. |
| `tf_set_kernel_duration_us(f, us)` | Changes a kernel's duration. |
| `tf_set_kernel_param_count(f, n)` | How many `kernelParams` pointers to copy at launch (default 0). |
| `tf_set_default_kernel_duration_us(us)` | Duration for kernels `cuModuleGetFunction` creates on demand. |
| `tf_set_time_mode(manual)` | 0 scaled-real, 1 manual. |
| `tf_advance_time_us(us)` | Moves the manual clock; `0` just waits for the worker. |
| `tf_now_us()` | Current simulated time. |
| `tf_set_result(name, r)` | Forces one exported symbol to return `r`; `CUDA_SUCCESS` clears it. |
| `tf_clear_results()` | Clears every injection. |
| `tf_set_fork_mode(emulate_real)` | 0 the child keeps working, 1 the child fails like real CUDA. |
| `tf_stream_launch_count(s)` | Kernel launches enqueued on a stream. |
| `tf_last_stream_was_per_thread_default()` | Proof of which entry point resolved the last stream. |
| `tf_legacy_default_stream()`, `tf_per_thread_default_stream()` | The two default streams of the current context. |
| `tf_symbol_table_size()` and the `tf_symbol_*_at` accessors | Enumerate the `cuGetProcAddress` table. |
| `tf_symbol_index(name)`, `tf_symbol_address(name)` | Look up one exported symbol by name. |
| `tf_set_driver_version(v)` | What `cuDriverGetVersion` reports, and the `cuGetProcAddress` ceiling. |
| `tf_set_device_total_mem(b)`, `tf_set_device_sm_count(n)` | Device properties. |
| `tf_control_failure_count()` | How many `tf_*` calls have failed (in practice: could not allocate). A `tf_*` function returning `void` has no other channel, so it counts the failure instead of swallowing it. |

Counters are plain relaxed atomics outside the simulation lock, so counting a
call takes no lock. They survive into a stats file at process exit whenever
`TESSERA_FAKE_STATS_FILE` names a path, which is how a shim test that runs its
"application" in a child process gets the child's counts back.

## Deviations

Places where the fake knowingly differs from a real driver, and why.

1. **`cuMemcpy*` and `cuMemset*` take zero simulated time.** Only kernels have a
   duration. Modelling bandwidth would add a number nobody measured; a test that
   needs a copy to take time puts a kernel next to it.
2. **A blocking call jumps the manual clock.** A real driver cannot fast-forward
   time. Without this, manual mode would deadlock on the first synchronous copy.
3. **`kernelParams` values are not copied.** The fake copies the array of
   pointers (using `tf_set_kernel_param_count`, default 0) but cannot know the
   argument sizes, so the pointed-to values must stay alive until the launch
   completes. Real CUDA copies them at launch time.
4. **Version-1 device pointers live in their own space.** The v1 entry points
   take 32-bit device pointers, which cannot hold a host address, so each v1
   allocation gets a 32-bit token the v1 entry points translate back. A v1 token
   means nothing to `cuMemAlloc_v2` and friends, and vice versa. The tokens are
   never 256-byte aligned, so one can never be mistaken for a real allocation.
5. **`cuModuleGetFunction` never fails on an unknown name.** The module image is
   opaque to the fake, so an unregistered name creates a kernel on demand rather
   than returning `CUDA_ERROR_NOT_FOUND`.
6. **A graph is a flat list, not a DAG.** Capture records the operations
   submitted to the stream in order and replay re-submits them in order. There
   are no explicit nodes, dependencies or cross-stream capture.
7. **`cuStreamDestroy` drains the stream.** A real driver returns immediately
   and destroys the stream once its work finishes. Draining is simpler and
   observationally equivalent for these tests.
8. **`cuDeviceGetAttribute` answers 0 for attributes the simulation has no
   opinion about**, rather than failing, so a caller enumerating the whole enum
   still works.
9. **A kernel body must not call back into the driver.** The worker holds the
   simulation lock while it runs the body, which is what gives a test that reads
   a buffer after `cuStreamSynchronize` a happens-before edge to the kernel that
   wrote it.
10. **`cuMemMap` falls back to an anonymous mapping** when `memfd_create` is
    unavailable; two mappings of one allocation then do not alias.

## Layout

```
fake_driver/
  include/tessera_fake/
    control.h          the tf_* control API (C linkage, documented)
    cuda_variants.h    the _ptsz/_ptds entry points and the v1 signatures,
                       declared under their real names without cuda.h's macros
  src/
    table.inc          the one table: exported name, base name, version,
                       stream kind. Drives the symbol enum, the
                       cuGetProcAddress table, the control-API enumeration and
                       the generated version script.
    counters.h         per-exported-symbol counters and the TF_ENTER macro
    sim.h / sim.cc     the simulated GPU: handles, clock, streams, worker
                       thread, memory, fork and exit handling
    entry_points.cc    every exported cu* symbol, and the table itself
    control.cc         the tf_* functions
  exports.map.in       template for the generated linker version script
  .clang-tidy          one scoped option on top of the repository policy
```

`fake_driver/.clang-tidy` inherits the repository policy and changes exactly one
option: `misc-non-private-member-variables-in-classes` is told to ignore records
whose member variables are all public. `Sim` is the library's own state, not a
public abstraction — it lives in a header private to `fake_driver/src`, every
symbol in it is hidden, and `entry_points.cc` manipulates it directly under
`Sim::mu` rather than through a wall of one-line accessors that would each have
to restate the same locking rule. The `cppcoreguidelines-` spelling of this
check defaults that option on and the `misc-` spelling defaults it off; this
asks for the former behaviour, which still catches the half-encapsulated types
the check exists for. Nothing else in the policy is relaxed, and `scripts/tidy.sh`
passes over `fake_driver/` and `tests/fake_driver/` with no findings.

## Tests

`tests/fake_driver/` covers stream FIFO ordering and durations, event timing,
manual versus scaled clock, memory allocation, copies and frees with real data,
graph capture and replay, error injection, every row of the `cuGetProcAddress`
resolution table, `_ptsz` versus legacy dispatch, the stats-file round trip
through a child process, the exported symbol set, and thread safety under
concurrent launches on several streams. The fork tests are a separate binary
because they run with `TSAN_OPTIONS=die_after_fork=0`: ThreadSanitizer's default
is to kill a child that starts threads after a multi-threaded fork, which is
exactly what the fake does on purpose.

```
cmake --preset gcc-debug && cmake --build --preset gcc-debug -j3 && ctest --preset gcc-debug
cmake --preset clang-asan  ...    # ASan + UBSan
cmake --preset clang-tsan  ...    # TSan
cmake --preset arm64-gcc   ...    # cross build, tests under qemu-aarch64
```
