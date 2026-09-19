// The "application" half of the shim's strategy smoke test.
//
// It is an ordinary CUDA driver-API program: it includes <cuda.h> with no
// Tessera macros, links against libcuda.so.1, and has no idea the shim exists
// (I-2). The parent runs it twice -- once with the shim masquerading as
// libcuda.so.1 on LD_LIBRARY_PATH, once with libtessera.so on LD_PRELOAD --
// and compares the counters both sides wrote at exit.
//
// It reaches the driver by three of the four access paths ADR-001 lists:
//
//   1. direct linkage (DT_NEEDED libcuda.so.1) for the first launches;
//   2. dlopen("libcuda.so.1") + dlsym for one more;
//   3. cuGetProcAddress for two more, one of them asking for the per-thread
//      default-stream variant.
//
// Expected counts, which smoke_strategies_test.cc asserts on both the shim's
// and the fake's side:
//
//   cuLaunchKernel       7   (5 direct + 1 via dlsym + 1 via cuGetProcAddress)
//   cuLaunchKernel_ptsz  1   (the per-thread-default-stream lookup)
//
// Exit codes: 0 success, 1 a driver call failed, 2 a usage error.

#include <dlfcn.h>

#include <cstdio>
#include <cstring>

#include <cuda.h>

namespace {

constexpr int kDirectLaunches = 5;

bool ok(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "tenant: %s failed with %d\n", what, static_cast<int>(r));
  return false;
}

using LaunchKernelFn = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int, unsigned int,
                                    CUstream, void**, void**);

LaunchKernelFn as_launch(void* p) {
  LaunchKernelFn fn = nullptr;
  std::memcpy(&fn, &p, sizeof(fn));
  return fn;
}

// Every launch in this program is the same 1x1x1 kernel; only the route to the
// entry point differs.
CUresult launch(LaunchKernelFn fn, CUfunction f, CUstream stream) {
  return fn(f, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr);
}

// TESSERA_REAL_LIBCUDA points the shim at itself. The shim must return an
// error instead of recursing (ADR-001's self-load guard).
int run_selfref() {
  const CUresult r = cuInit(0);
  if (r == CUDA_SUCCESS) {
    std::fprintf(stderr, "tenant: cuInit succeeded, but the driver was the shim itself\n");
    return 1;
  }
  std::fprintf(stderr, "tenant: self-reference refused with %d, as it should be\n",
               static_cast<int>(r));
  return 0;
}

int run_workload() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CUstream stream = nullptr;

  if (!ok(cuInit(0), "cuInit") || !ok(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !ok(cuCtxCreate(&context, 0, device), "cuCtxCreate")) {
    return 1;
  }
  // The fake accepts any module image and creates the kernel on demand.
  if (!ok(cuModuleLoadData(&module, "tenant-module"), "cuModuleLoadData") ||
      !ok(cuModuleGetFunction(&function, module, "tenant_kernel"), "cuModuleGetFunction") ||
      !ok(cuStreamCreate(&stream, 0), "cuStreamCreate")) {
    return 1;
  }

  // Path 1: direct linkage.
  for (int i = 0; i < kDirectLaunches; ++i) {
    if (!ok(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr),
            "cuLaunchKernel")) {
      return 1;
    }
  }

  // Path 2: dlopen + dlsym. Under masquerade this finds the shim by search
  // path; under LD_PRELOAD it finds the real library and the dlsym hook is
  // what redirects it.
  void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    std::fprintf(stderr, "tenant: dlopen(libcuda.so.1) failed: %s\n", dlerror());
    return 1;
  }
  const LaunchKernelFn by_dlsym = as_launch(dlsym(handle, "cuLaunchKernel"));
  if (by_dlsym == nullptr) {
    std::fprintf(stderr, "tenant: dlsym(cuLaunchKernel) failed: %s\n", dlerror());
    return 1;
  }
  if (!ok(launch(by_dlsym, function, stream), "cuLaunchKernel via dlsym")) {
    return 1;
  }

  // Path 3: cuGetProcAddress, the way cudart resolves everything. Default
  // flags give the legacy entry point.
  void* resolved = nullptr;
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
  if (!ok(cuGetProcAddress("cuLaunchKernel", &resolved, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT,
                           &status),
          "cuGetProcAddress(cuLaunchKernel)")) {
    return 1;
  }
  if (!ok(launch(as_launch(resolved), function, stream), "cuLaunchKernel via cuGetProcAddress")) {
    return 1;
  }

  // ... and the per-thread default-stream variant, with a NULL stream so that
  // which entry point ran is observable in the fake.
  resolved = nullptr;
  if (!ok(cuGetProcAddress("cuLaunchKernel", &resolved, CUDA_VERSION,
                           CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status),
          "cuGetProcAddress(cuLaunchKernel, per-thread)")) {
    return 1;
  }
  if (!ok(launch(as_launch(resolved), function, nullptr), "cuLaunchKernel_ptsz")) {
    return 1;
  }

  if (!ok(cuCtxSynchronize(), "cuCtxSynchronize") ||
      !ok(cuStreamDestroy(stream), "cuStreamDestroy")) {
    return 1;
  }
  // Returning from main runs both libraries' exit handlers, which is what
  // writes TESSERA_STATS_FILE and TESSERA_FAKE_STATS_FILE.
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s workload|selfref\n", argv[0]);
    return 2;
  }
  if (std::strcmp(argv[1], "workload") == 0) {
    return run_workload();
  }
  if (std::strcmp(argv[1], "selfref") == 0) {
    return run_selfref();
  }
  std::fprintf(stderr, "%s: unknown mode '%s'\n", argv[0], argv[1]);
  return 2;
}
