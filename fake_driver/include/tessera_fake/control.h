/* libcuda_fake control API.
 *
 * libcuda_fake is a CPU-only simulation of the CUDA driver. It exports the
 * cu* entry points that Tessera's shim hooks, plus the tf_* functions declared
 * here, which let a test drive and inspect the simulation. Everything else in
 * the library is hidden.
 *
 * All tf_* functions have C linkage and are safe to call from any thread.
 * None of them throw; nothing in this library lets an exception cross a C ABI
 * boundary.
 *
 * Terminology used throughout:
 *
 *   exported symbol name   The name of one entry point in the .so, for example
 *                          "cuLaunchKernel_ptsz". Counters and forced results
 *                          are per exported symbol, so "cuLaunchKernel" and
 *                          "cuLaunchKernel_ptsz" are counted separately.
 *   base name              The name an application passes to cuGetProcAddress,
 *                          for example "cuLaunchKernel". Several exported
 *                          symbols share one base name.
 *   simulated microseconds The simulation's own clock. See tf_set_time_mode.
 */
#ifndef TESSERA_FAKE_CONTROL_H
#define TESSERA_FAKE_CONTROL_H

#include <stdint.h>

#include <cuda.h>

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* A generic function pointer. Returned instead of void* so that callers do not
 * have to cast between function and object pointers, which is not portable and
 * which -Wpedantic rejects. */
typedef void (*tf_fn_ptr)(void);

/* ---- stream kinds, as reported by tf_symbol_stream_kind_at ---------------- */
#define TF_STREAM_KIND_NONE 0       /* no stream argument, or no _ptsz twin   */
#define TF_STREAM_KIND_LEGACY 1     /* NULL stream -> legacy default stream   */
#define TF_STREAM_KIND_PER_THREAD 2 /* NULL stream -> per-thread default      */

/* ==========================================================================
 * Lifecycle
 * ========================================================================== */

/* Tears the simulation down and builds a fresh one: every context, stream,
 * event, module, graph and allocation is destroyed, all per-symbol counters go
 * to zero, all forced results are cleared, the clock returns to scaled-real
 * time at 0 us, the fork mode returns to 0, the device properties and driver
 * version return to their defaults, and the driver becomes uninitialised again
 * (so cuInit must be called before the next cu* call).
 *
 * Call it at the start of every test. It is the only way to get a known state.
 * Do not call it while another thread is inside a cu* entry point. */
void tf_reset(void);

/* ==========================================================================
 * Call counting and the stats file
 * ========================================================================== */

/* The number of times the entry point exported under this exact name has been
 * entered since the last tf_reset(). Counting happens on entry, before any
 * argument validation and before any forced result is applied, so a call that
 * fails still counts. Returns 0 for a name that is not an exported symbol;
 * use tf_symbol_index() to tell "not called" from "not a symbol". */
uint64_t tf_symbol_count(const char* exported_symbol_name);

/* Writes the counters to `path` as a JSON object mapping exported symbol name
 * to call count, for example {"cuInit":1,"cuLaunchKernel_ptsz":4}. Symbols
 * with a zero count are omitted. The file is written and flushed before the
 * call returns. Returns 0 on success, or -1 with errno set.
 *
 * The same file is written automatically at process exit when the environment
 * variable TESSERA_FAKE_STATS_FILE names a path, which is how a test that runs
 * its "application" in a child process gets the child's counters. The exit
 * dump happens in the process that first called into the library; a process
 * forked from it does not write the file (see tf_set_fork_mode). */
int tf_dump_stats(const char* path);

/* ==========================================================================
 * Kernels
 * ========================================================================== */

/* Registers a kernel under `name` and returns its CUfunction handle.
 *
 *   body          runs on the simulation's worker thread when the launch
 *                 completes, receiving the kernelParams array the application
 *                 passed to cuLaunchKernel. May be NULL for a kernel that only
 *                 burns simulated time. It must not call back into any cu* or
 *                 tf_* function: the worker holds the simulation lock while it
 *                 runs.
 *   duration_us   how long each launch occupies its stream, in simulated
 *                 microseconds. May be 0.
 *
 * Registering the same name twice updates the existing kernel and returns the
 * same handle. cuModuleGetFunction returns the kernel registered under the
 * requested name; if no kernel is registered under it, one is created with a
 * NULL body and the duration set by tf_set_default_kernel_duration_us. */
CUfunction tf_register_kernel(const char* name, void (*body)(void** kernel_params),
                              uint64_t duration_us);

/* Changes how long each launch of `f` occupies its stream. */
void tf_set_kernel_duration_us(CUfunction f, uint64_t us);

/* How many pointers the fake copies out of the kernelParams array at launch
 * time. Defaults to 0, which means the body is handed a NULL params array.
 * The fake cannot know a kernel's signature, so a kernel whose body reads its
 * parameters must declare the count here. Only the array of pointers is
 * copied: the values they point at must stay alive until the launch
 * completes, exactly as real CUDA requires for an asynchronous launch. */
void tf_set_kernel_param_count(CUfunction f, uint32_t count);

/* The duration given to kernels that cuModuleGetFunction creates on demand.
 * Defaults to 0 simulated microseconds. */
void tf_set_default_kernel_duration_us(uint64_t us);

/* ==========================================================================
 * The clock
 * ========================================================================== */

/* Chooses how simulated time advances.
 *
 *   0 (default)  scaled real time: simulated microseconds track
 *                CLOCK_MONOTONIC one for one. Blocking calls really block.
 *   1            manual: simulated time stands still until tf_advance_time_us
 *                moves it. A blocking call (cuStreamSynchronize,
 *                cuCtxSynchronize, cuEventSynchronize, or any synchronous
 *                copy or fill) jumps the clock forward to the moment the work
 *                it waits for completes, so it returns without ever waiting on
 *                wall-clock time. This is what makes tests deterministic.
 *
 * Switching modes preserves the current simulated time. */
void tf_set_time_mode(int manual);

/* Moves simulated time forward by `us` microseconds. In manual mode the call
 * returns only once the worker thread has completed every operation that is
 * due at the new time, so the effects are visible on return. Passing 0 is the
 * useful special case: it does not move the clock, it just waits for the
 * worker to catch up with it, which is how a test makes a zero-duration
 * operation observable without synchronizing a stream. In scaled-real mode it
 * shifts the clock's origin forward and returns immediately. */
void tf_advance_time_us(uint64_t us);

/* The current simulated time in microseconds. */
uint64_t tf_now_us(void);

/* ==========================================================================
 * Error injection
 * ========================================================================== */

/* Forces the entry point exported under `exported_symbol_name` to return `r`
 * immediately, after counting the call but before doing any work. Passing
 * CUDA_SUCCESS clears the injection for that symbol rather than forcing an
 * empty success, because returning success without filling the out-parameters
 * would hand the caller uninitialised memory. An unknown symbol name is
 * ignored. */
void tf_set_result(const char* exported_symbol_name, CUresult r);

/* Clears every forced result. */
void tf_clear_results(void);

/* ==========================================================================
 * Self-check
 * ========================================================================== */

/* How many tf_* calls have failed since the process started, which in practice
 * means "could not allocate". None of these functions may let an exception
 * cross the C ABI, and the ones that return void have nothing to report a
 * failure on, so they count it here instead of swallowing it. A test can
 * assert this stayed at zero. It is not reset by tf_reset(). */
uint64_t tf_control_failure_count(void);

/* ==========================================================================
 * Fork
 * ========================================================================== */

/* Chooses what happens to cu* calls made in a process forked from the one that
 * initialised the library.
 *
 *   0 (default)     the child keeps working. The worker thread, which does not
 *                   survive fork, is recreated on the child's next call; the
 *                   child's per-symbol counters start at zero; the child does
 *                   not write the exit stats file.
 *   1               every cu* entry point in the child returns
 *                   CUDA_ERROR_NOT_INITIALIZED, the way a real CUDA context
 *                   becomes unusable across fork.
 *
 * Either way the fake never deadlocks in the child: it installs pthread_atfork
 * handlers that hold the simulation lock across the fork. */
void tf_set_fork_mode(int emulate_real);

/* ==========================================================================
 * Streams
 * ========================================================================== */

/* How many kernel launches have been enqueued on this stream since the last
 * tf_reset(). Pass NULL for the current context's legacy default stream.
 * Returns 0 for a handle that is not a live stream. */
uint64_t tf_stream_launch_count(CUstream s);

/* 1 when the stream resolved by the most recent cu* call that takes a stream
 * was a per-thread default stream, 0 otherwise. This is how a test proves that
 * a _ptsz entry point, not its legacy twin, was the one that ran. The value is
 * process-wide and reflects the most recent resolution by any thread, so it is
 * meaningful in single-threaded tests. */
int tf_last_stream_was_per_thread_default(void);

/* The current context's legacy default stream, and the calling thread's
 * per-thread default stream in the current context. Both return NULL when
 * there is no current context. The per-thread stream is created on first
 * use. */
CUstream tf_legacy_default_stream(void);
CUstream tf_per_thread_default_stream(void);

/* ==========================================================================
 * The cuGetProcAddress table
 * ========================================================================== */

/* The number of exported cu* entry points, which is the number of rows in the
 * table that cuGetProcAddress resolves against. */
uint64_t tf_symbol_table_size(void);

/* Row accessors. `i` must be less than tf_symbol_table_size(); out of range,
 * the name accessors return NULL and the integer accessors return -1.
 * The returned strings are static and outlive any tf_reset(). */
const char* tf_symbol_name_at(uint64_t i);      /* e.g. "cuMemAlloc_v2"       */
const char* tf_symbol_base_name_at(uint64_t i); /* e.g. "cuMemAlloc"          */
int tf_symbol_version_at(uint64_t i);           /* e.g. 3020                  */
int tf_symbol_stream_kind_at(uint64_t i);       /* TF_STREAM_KIND_*           */

/* The row index of an exported symbol name, or -1 if there is no such export.
 * Use it to check a name before passing it to tf_symbol_count or
 * tf_set_result. */
int64_t tf_symbol_index(const char* exported_symbol_name);

/* The address of an exported entry point, bypassing cuGetProcAddress's version
 * and stream-flag selection. Returns NULL for an unknown name. A test uses it
 * to call one specific variant, and to check what cuGetProcAddress returned. */
tf_fn_ptr tf_symbol_address(const char* exported_symbol_name);

/* ==========================================================================
 * Device properties
 * ========================================================================== */

/* The value cuDriverGetVersion reports, and the ceiling cuGetProcAddress
 * applies: an entry point introduced after this version cannot be resolved,
 * exactly as it could not be on a real driver of that age. Defaults to the
 * CUDA_VERSION of the cuda.h the library was built against. */
void tf_set_driver_version(int version);

/* The simulated device's total memory in bytes (defaults to 16 GiB) and its
 * streaming-multiprocessor count (defaults to 58). Both feed
 * cuDeviceGetAttribute, cuDeviceTotalMem and cuMemGetInfo, and the memory size
 * is a real budget: an allocation past it fails with CUDA_ERROR_OUT_OF_MEMORY.
 * tf_reset() restores the defaults. */
void tf_set_device_total_mem(uint64_t bytes);
void tf_set_device_sm_count(int sm_count);

#ifdef __cplusplus
} /* extern "C" */
#endif

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif

#endif /* TESSERA_FAKE_CONTROL_H */
