/* libtessera's observability ABI.
 *
 * These are the only non-`cu*` symbols the shim exports, and other Tessera
 * components (the eval harness, the acceptance suite, later tesserad) code
 * against exactly these names. They are C-linkage and safe to call from any
 * thread; none of them throws, and nothing in libtessera lets an exception
 * cross a C ABI boundary.
 *
 * A test normally reaches them one of two ways:
 *
 *   dlsym(handle, "tessera_get_stats_v1")   in-process, on a handle for
 *                                           libtessera.so or the masquerading
 *                                           libcuda.so.1;
 *   $TESSERA_STATS_FILE                     for an application that runs in a
 *                                           child process: the shim writes
 *                                           the same numbers as JSON at exit.
 *
 * Environment variables libtessera reads (all optional):
 *
 *   TESSERA_REAL_LIBCUDA   absolute path of the real driver to forward to.
 *                          Unset falls back to a documented search; see
 *                          shim/README.md.
 *   TESSERA_STATS_FILE     path to write the JSON of tessera_dump_stats() to
 *                          at process exit.
 *   TESSERA_LOG            non-empty and not "0": log lifecycle events to
 *                          stderr. Never logged from the per-launch path.
 */
#ifndef TESSERA_SHIM_STATS_H
#define TESSERA_SHIM_STATS_H

#include <stdint.h>

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The ABI this header describes. tessera_abi_version() returns it. */
#define TESSERA_ABI_VERSION 1

/* Process-wide call counts since the last tessera_reset_stats().
 *
 *   launches             calls to a launch entry point: cuLaunchKernel and its
 *                        _ptsz twin, cuLaunchKernelEx, cuLaunchCooperativeKernel
 *                        and the legacy cuLaunch/cuLaunchGrid/cuLaunchGridAsync.
 *                        Graph launches are NOT included.
 *   graph_launches       calls to cuGraphLaunch(_ptsz). Counted apart because
 *                        one graph launch is one API call but many kernels,
 *                        which would otherwise corrupt the G9 comparison.
 *   hooked_calls         calls to any hooked entry point, launches included.
 *   forwarded_unhooked   calls to an exported symbol the shim does not hook,
 *                        which reached the real driver through a trampoline.
 *   bypassed_lookups     cuGetProcAddress requests for a symbol the shim DOES
 *                        hook where the pointer the driver returned matched
 *                        none of the variants this build knows, so the real
 *                        pointer was handed back unchanged. A non-zero value
 *                        means this build is older than the driver and is
 *                        missing calls: it is the signal H1 uses to tell
 *                        "we captured everything" from "we captured what we
 *                        knew about".
 *
 * The fields are in this order, and fields are only ever appended: a new field
 * gets a new tessera_stats_v2 and a new accessor.
 */
typedef struct tessera_stats_v1 {
  uint64_t launches;
  uint64_t graph_launches;
  uint64_t hooked_calls;
  uint64_t forwarded_unhooked;
  uint64_t bypassed_lookups;
} tessera_stats_v1;

/* TESSERA_ABI_VERSION of the loaded shim. Always safe to call: it neither
 * initialises the shim nor touches the driver. */
int tessera_abi_version(void);

/* Fills *out with the current counts. Returns 0, or -1 if out is NULL.
 * The five fields are read one after another with relaxed atomics, so under
 * concurrent traffic the snapshot can be slightly skewed between fields; it is
 * exact once the application's threads are quiet. */
int tessera_get_stats_v1(tessera_stats_v1* out);

/* Calls to one EXPORTED VARIANT: tessera_get_symbol_count("cuLaunchKernel")
 * and tessera_get_symbol_count("cuLaunchKernel_ptsz") count separately, as the
 * driver's own entry points do. Works for hooked symbols and for the ones that
 * are only forwarded. Returns 0 for a name this build does not export. */
uint64_t tessera_get_symbol_count(const char* exported_symbol_name);

/* Zeroes every counter: the aggregates and all per-symbol counts. Does not
 * touch the loaded driver, the resolved function pointers or the
 * cuGetProcAddress mapping, so a benchmark can reset between windows without
 * paying for re-resolution. */
void tessera_reset_stats(void);

/* Writes the counters to `path` as a JSON object: the five aggregates, plus a
 * "symbols" member mapping exported symbol name to call count for every symbol
 * with a non-zero count. Returns 0 on success, or -1 with errno set.
 *
 * The same content is written at process exit when TESSERA_STATS_FILE names a
 * path, which is how a test that runs its application in a child process gets
 * the child's counters back. A process forked from the one that initialised
 * the shim does not write the file. */
int tessera_dump_stats(const char* path);

#ifdef __cplusplus
} /* extern "C" */
#endif

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif

#endif /* TESSERA_SHIM_STATS_H */
