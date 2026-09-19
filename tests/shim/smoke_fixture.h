// Shared scaffolding for the shim smoke tests.
//
// The shim is dlopen'd rather than linked, which is how every real deployment
// gets it, and every call goes through a pointer taken from the shim's own
// handle. The fake is linked into the test binary as libcuda.so.1 AND named by
// TESSERA_REAL_LIBCUDA, so the shim's dlopen of that absolute path lands on
// the same object: one simulation, visible from both sides.

#ifndef TESSERA_TESTS_SHIM_SMOKE_FIXTURE_H
#define TESSERA_TESTS_SHIM_SMOKE_FIXTURE_H

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda.h>
#include <sys/wait.h>

#include <gtest/gtest.h>

#include "tessera/shim/stats.h"

#include "tessera_fake/control.h"

// Generated from the driver stub's export list by shim/tools/gen_symbols.sh.
#include "tessera_symbol_index.h"

namespace tessera_test {
namespace {

#define ASSERT_CU(expr) ASSERT_EQ(CUDA_SUCCESS, (expr))
#define EXPECT_CU(expr) EXPECT_EQ(CUDA_SUCCESS, (expr))

#define TESSERA_SYMBOL(index, name, hooked) name,
const char* const kDriverExports[TESSERA_SYMBOL_COUNT] = {
#include "tessera_symbols.inc"
};
#undef TESSERA_SYMBOL

using FnInit = CUresult (*)(unsigned int);
using FnDeviceGet = CUresult (*)(CUdevice*, int);
using FnDeviceGetCount = CUresult (*)(int*);
using FnCtxCreate = CUresult (*)(CUcontext*, unsigned int, CUdevice);
using FnCtxSynchronize = CUresult (*)();
using FnModuleLoadData = CUresult (*)(CUmodule*, const void*);
using FnModuleGetFunction = CUresult (*)(CUfunction*, CUmodule, const char*);
using FnStreamCreate = CUresult (*)(CUstream*, unsigned int);
using FnLaunchKernel = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int, unsigned int,
                                    CUstream, void**, void**);
using FnMemAllocV1 = CUresult (*)(unsigned int*, unsigned int);
using FnMemAllocV2 = CUresult (*)(CUdeviceptr*, size_t);
using FnMemFreeV2 = CUresult (*)(CUdeviceptr);
using FnStreamBeginCapture = CUresult (*)(CUstream, CUstreamCaptureMode);
using FnStreamEndCapture = CUresult (*)(CUstream, CUgraph*);
using FnGraphInstantiate = CUresult (*)(CUgraphExec*, CUgraph, unsigned long long);
using FnGraphLaunch = CUresult (*)(CUgraphExec, CUstream);
using FnGetProcAddressV1 = CUresult (*)(const char*, void**, int, cuuint64_t);
using FnGetProcAddressV2 = CUresult (*)(const char*, void**, int, cuuint64_t,
                                        CUdriverProcAddressQueryResult*);

using FnAbiVersion = int (*)();
using FnGetStats = int (*)(tessera_stats_v1*);
using FnSymbolCount = uint64_t (*)(const char*);
using FnResetStats = void (*)();
using FnDumpStats = int (*)(const char*);

template <typename T>
T symbol_as(void* handle, const char* name) {
  void* address = dlsym(handle, name);
  T fn = nullptr;
  std::memcpy(&fn, &address, sizeof(fn));
  return fn;
}

// The shim, loaded once for the whole binary. Loading it twice would tell us
// nothing new and the second load would find the first one's state.
struct Loaded {
  void* handle = nullptr;
  uint64_t driver_calls_after_loading_the_shim = 0;

  FnAbiVersion abi_version = nullptr;
  FnGetStats get_stats = nullptr;
  FnSymbolCount symbol_count = nullptr;
  FnResetStats reset_stats = nullptr;
  FnDumpStats dump_stats = nullptr;
};

inline const Loaded& shim() {
  static const Loaded loaded = [] {
    Loaded l;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): single-threaded test setup.
    ::setenv("TESSERA_REAL_LIBCUDA", TESSERA_FAKE_MASQ_LIB, 1);
    l.handle = dlopen(TESSERA_SHIM_PRELOAD_PATH, RTLD_NOW | RTLD_LOCAL);
    if (l.handle == nullptr) {
      std::fprintf(stderr, "dlopen(%s): %s\n", TESSERA_SHIM_PRELOAD_PATH, dlerror());
      return l;
    }
    // T5 in miniature: loading the shim must not call the driver. Whether the
    // driver is MAPPED cannot be the test here, because this binary links the
    // fake itself; what matters, and what a static constructor in the shim
    // would break, is that not one entry point has been entered yet. CTest
    // runs each case in its own process, so this is measured on a fresh one.
    for (uint64_t i = 0; i < tf_symbol_table_size(); ++i) {
      l.driver_calls_after_loading_the_shim += tf_symbol_count(tf_symbol_name_at(i));
    }
    l.abi_version = symbol_as<FnAbiVersion>(l.handle, "tessera_abi_version");
    l.get_stats = symbol_as<FnGetStats>(l.handle, "tessera_get_stats_v1");
    l.symbol_count = symbol_as<FnSymbolCount>(l.handle, "tessera_get_symbol_count");
    l.reset_stats = symbol_as<FnResetStats>(l.handle, "tessera_reset_stats");
    l.dump_stats = symbol_as<FnDumpStats>(l.handle, "tessera_dump_stats");
    return l;
  }();
  return loaded;
}

// open/read rather than stdio: the stats file is a few hundred bytes, and a
// descriptor has one failure mode instead of three.
inline std::string read_file(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return {};
  }
  std::string text;
  char buffer[4096];
  for (;;) {
    const ssize_t n = ::read(fd, buffer, sizeof(buffer));
    if (n <= 0) {
      break;
    }
    text.append(buffer, static_cast<size_t>(n));
  }
  ::close(fd);
  return text;
}

inline tessera_stats_v1 stats() {
  tessera_stats_v1 s;
  std::memset(&s, 0, sizeof(s));
  EXPECT_EQ(0, shim().get_stats(&s));
  return s;
}

// A fresh simulation reached through the shim: the fake is reset, the shim's
// counters are zeroed, and a context exists. Every call below goes through a
// pointer taken from the shim's handle, never through this binary's own
// libcuda.so.1.
class ShimSmokeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NE(nullptr, shim().handle);
    tf_reset();
    launch_ = symbol_as<FnLaunchKernel>(shim().handle, "cuLaunchKernel");
    ASSERT_NE(nullptr, launch_);
    ASSERT_CU(symbol_as<FnInit>(shim().handle, "cuInit")(0));
    ASSERT_CU(symbol_as<FnDeviceGet>(shim().handle, "cuDeviceGet")(&device_, 0));
    ASSERT_CU(symbol_as<FnCtxCreate>(shim().handle, "cuCtxCreate_v2")(&context_, 0, device_));
    tf_set_time_mode(1);
    ASSERT_CU(symbol_as<FnModuleLoadData>(shim().handle, "cuModuleLoadData")(&module_, "image"));
    ASSERT_CU(symbol_as<FnModuleGetFunction>(shim().handle, "cuModuleGetFunction")(
        &function_, module_, "smoke_kernel"));
    // Counting starts here, so the fixture's own setup calls are not in it.
    shim().reset_stats();
  }

  void TearDown() override { tf_reset(); }

  FnLaunchKernel launch_ = nullptr;
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  CUmodule module_ = nullptr;
  CUfunction function_ = nullptr;
};

}  // namespace
}  // namespace tessera_test

#endif  // TESSERA_TESTS_SHIM_SMOKE_FIXTURE_H
