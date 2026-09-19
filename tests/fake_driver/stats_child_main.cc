// The "application" half of the stats-file round trip.
//
// A shim test runs its tenant in a child process, so the fake's counters have
// to survive into a file at exit. This program makes a fixed, easily checked
// set of driver calls and then exits normally; the parent reads whatever
// TESSERA_FAKE_STATS_FILE named.
//
// Exit codes: 0 on success, 1 if any driver call failed.

#include <cstdio>

#include <cuda.h>

#include "tessera_fake/control.h"
#include "tessera_fake/cuda_variants.h"

namespace {

// Keep this in step with stats_test.cc.
constexpr int kLegacyLaunches = 3;
constexpr int kPerThreadLaunches = 2;

bool ok(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "stats_child: %s failed with %d\n", what, static_cast<int>(r));
  return false;
}

}  // namespace

int main() {
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  if (!ok(cuInit(0), "cuInit")) {
    return 1;
  }
  if (!ok(cuDeviceGet(&dev, 0), "cuDeviceGet")) {
    return 1;
  }
  if (!ok(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate")) {
    return 1;
  }
  tf_set_time_mode(1);

  CUfunction f = tf_register_kernel("child_kernel", nullptr, 10);
  if (f == nullptr) {
    std::fprintf(stderr, "stats_child: tf_register_kernel failed\n");
    return 1;
  }

  for (int i = 0; i < kLegacyLaunches; ++i) {
    if (!ok(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr), "cuLaunchKernel")) {
      return 1;
    }
  }
  for (int i = 0; i < kPerThreadLaunches; ++i) {
    if (!ok(cuLaunchKernel_ptsz(f, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr),
            "cuLaunchKernel_ptsz")) {
      return 1;
    }
  }
  if (!ok(cuCtxSynchronize(), "cuCtxSynchronize")) {
    return 1;
  }

  // Returning from main runs the library's exit handler, which is what writes
  // TESSERA_FAKE_STATS_FILE.
  return 0;
}
