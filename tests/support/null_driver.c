/* A CUDA driver that does nothing at all, for the I-4 measurement (T7).
 *
 * T7 claims something about the SHIM's per-launch path: in steady state it
 * performs no heap allocation and no syscall. Measuring that against
 * libcuda_fake would measure the fake instead -- it takes a lock, appends to a
 * queue and wakes a worker thread, so a launch through it allocates and issues
 * a FUTEX_WAKE, and none of that is Tessera's. So the driver underneath the
 * window is this: entry points that return CUDA_SUCCESS after one relaxed
 * atomic add, with no allocation, no lock and no syscall of their own.
 *
 * What is left inside the measured window is therefore exactly the shim's
 * wrapper -- its counters and its tail jump -- which is what I-4 is about.
 * The same application run against the fake driver is the positive control
 * that the instrument sees allocations and syscalls when there are any.
 *
 * It is built with SONAME libcuda.so.1 in a directory of its own, so it can be
 * either the library the shim forwards to (TESSERA_REAL_LIBCUDA) or, for the
 * no-shim control, the application's own libcuda.so.1 on LD_LIBRARY_PATH.
 *
 * Handles are small non-null integers cast to pointers: nothing dereferences
 * them, and a null handle would be indistinguishable from a failure.
 */

#include <stdatomic.h>
#include <stddef.h>
#include <string.h>

#include <cuda.h>

#define TNULL_EXPORT __attribute__((visibility("default")))

/* cuda.h declares the _ptsz entry points only for the driver's own sources
 * (__CUDA_API_VERSION_INTERNAL), and this file is neither that nor a consumer
 * of the fake driver's headers, so the one it defines is declared here. */
TNULL_EXPORT CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX,
                                                  unsigned int gridDimY, unsigned int gridDimZ,
                                                  unsigned int blockDimX, unsigned int blockDimY,
                                                  unsigned int blockDimZ,
                                                  unsigned int sharedMemBytes, CUstream hStream,
                                                  void** kernelParams, void** extra);

/* One counter per entry point this library implements, so a test can prove the
 * calls arrived even when there is no shim in the process to count them. */
#define TNULL_SYMBOLS(X) \
  X(cuInit)              \
  X(cuDriverGetVersion)  \
  X(cuDeviceGet)         \
  X(cuDeviceGetCount)    \
  X(cuCtxCreate_v2)      \
  X(cuCtxDestroy_v2)     \
  X(cuModuleLoadData)    \
  X(cuModuleGetFunction) \
  X(cuModuleUnload)      \
  X(cuStreamCreate)      \
  X(cuStreamDestroy_v2)  \
  X(cuStreamSynchronize) \
  X(cuLaunchKernel)      \
  X(cuLaunchKernel_ptsz)

enum {
#define TNULL_ENUM(name) TNULL_IDX_##name,
  TNULL_SYMBOLS(TNULL_ENUM)
#undef TNULL_ENUM
      TNULL_COUNT
};

static atomic_ullong g_counts[TNULL_COUNT];

static const char* const g_names[TNULL_COUNT] = {
#define TNULL_NAME(name) #name,
    TNULL_SYMBOLS(TNULL_NAME)
#undef TNULL_NAME
};

#define TNULL_ENTER(name) \
  atomic_fetch_add_explicit(&g_counts[TNULL_IDX_##name], 1ULL, memory_order_relaxed)

/* Calls to one entry point since the process started, or 0 for a name this
 * library does not implement. Declared before it is defined because the build
 * compiles with -Wmissing-declarations and this one has no header. */
TNULL_EXPORT unsigned long long tnull_call_count(const char* name);

TNULL_EXPORT unsigned long long tnull_call_count(const char* name) {
  if (name == NULL) {
    return 0;
  }
  for (int i = 0; i < TNULL_COUNT; ++i) {
    if (strcmp(g_names[i], name) == 0) {
      return atomic_load_explicit(&g_counts[i], memory_order_relaxed);
    }
  }
  return 0;
}

TNULL_EXPORT CUresult CUDAAPI cuInit(unsigned int flags) {
  (void)flags;
  TNULL_ENTER(cuInit);
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuDriverGetVersion(int* version) {
  TNULL_ENTER(cuDriverGetVersion);
  if (version != NULL) {
    *version = CUDA_VERSION;
  }
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuDeviceGet(CUdevice* device, int ordinal) {
  (void)ordinal;
  TNULL_ENTER(cuDeviceGet);
  if (device != NULL) {
    *device = 0;
  }
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuDeviceGetCount(int* count) {
  TNULL_ENTER(cuDeviceGetCount);
  if (count != NULL) {
    *count = 1;
  }
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  (void)flags;
  (void)dev;
  TNULL_ENTER(cuCtxCreate_v2);
  if (pctx != NULL) {
    *pctx = (CUcontext)0x10;
  }
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuCtxDestroy_v2(CUcontext ctx) {
  (void)ctx;
  TNULL_ENTER(cuCtxDestroy_v2);
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* image) {
  (void)image;
  TNULL_ENTER(cuModuleLoadData);
  if (module != NULL) {
    *module = (CUmodule)0x20;
  }
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuModuleGetFunction(CUfunction* func, CUmodule module,
                                                  const char* name) {
  (void)module;
  (void)name;
  TNULL_ENTER(cuModuleGetFunction);
  if (func != NULL) {
    *func = (CUfunction)0x30;
  }
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuModuleUnload(CUmodule module) {
  (void)module;
  TNULL_ENTER(cuModuleUnload);
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuStreamCreate(CUstream* stream, unsigned int flags) {
  (void)flags;
  TNULL_ENTER(cuStreamCreate);
  if (stream != NULL) {
    *stream = (CUstream)0x40;
  }
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuStreamDestroy_v2(CUstream stream) {
  (void)stream;
  TNULL_ENTER(cuStreamDestroy_v2);
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuStreamSynchronize(CUstream stream) {
  (void)stream;
  TNULL_ENTER(cuStreamSynchronize);
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                             unsigned int gridDimY, unsigned int gridDimZ,
                                             unsigned int blockDimX, unsigned int blockDimY,
                                             unsigned int blockDimZ, unsigned int sharedMemBytes,
                                             CUstream hStream, void** kernelParams, void** extra) {
  (void)f;
  (void)gridDimX;
  (void)gridDimY;
  (void)gridDimZ;
  (void)blockDimX;
  (void)blockDimY;
  (void)blockDimZ;
  (void)sharedMemBytes;
  (void)hStream;
  (void)kernelParams;
  (void)extra;
  TNULL_ENTER(cuLaunchKernel);
  return CUDA_SUCCESS;
}

TNULL_EXPORT CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX,
                                                  unsigned int gridDimY, unsigned int gridDimZ,
                                                  unsigned int blockDimX, unsigned int blockDimY,
                                                  unsigned int blockDimZ,
                                                  unsigned int sharedMemBytes, CUstream hStream,
                                                  void** kernelParams, void** extra) {
  (void)f;
  (void)gridDimX;
  (void)gridDimY;
  (void)gridDimZ;
  (void)blockDimX;
  (void)blockDimY;
  (void)blockDimZ;
  (void)sharedMemBytes;
  (void)hStream;
  (void)kernelParams;
  (void)extra;
  TNULL_ENTER(cuLaunchKernel_ptsz);
  return CUDA_SUCCESS;
}
