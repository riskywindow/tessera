// Shared scaffolding for the libcuda_fake tests.

#ifndef TESSERA_TESTS_FAKE_DRIVER_FAKE_FIXTURE_H
#define TESSERA_TESTS_FAKE_DRIVER_FAKE_FIXTURE_H

#include <cstdint>
#include <cstring>

#include <cuda.h>

#include <gtest/gtest.h>

#include "tessera_fake/control.h"
#include "tessera_fake/cuda_variants.h"

#define ASSERT_CU(expr) ASSERT_EQ(CUDA_SUCCESS, (expr))
#define EXPECT_CU(expr) EXPECT_EQ(CUDA_SUCCESS, (expr))

namespace tessera_test {

// The reserved stream tokens, spelled without a C-style cast so the test files
// compile under -Wold-style-cast.
inline CUstream legacy_stream_token() {
  return reinterpret_cast<CUstream>(static_cast<uintptr_t>(1));
}
inline CUstream per_thread_stream_token() {
  return reinterpret_cast<CUstream>(static_cast<uintptr_t>(2));
}

// cuGetProcAddress hands back a void*; the table hands back a function pointer.
// Converting between the two with a cast is not portable enough for -Wpedantic,
// so both are compared as raw bytes.
inline tf_fn_ptr as_fn(void* p) {
  tf_fn_ptr f = nullptr;
  std::memcpy(&f, &p, sizeof(f));
  return f;
}

// A fresh simulation with one initialised context, on the manual clock so that
// every test that follows is deterministic.
class FakeDriverTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tf_reset();
    ASSERT_CU(cuInit(0));
    ASSERT_CU(cuDeviceGet(&dev_, 0));
    ASSERT_CU(cuCtxCreate(&ctx_, 0, dev_));
    tf_set_time_mode(1);
  }

  void TearDown() override { tf_reset(); }

  CUdevice dev_ = 0;
  CUcontext ctx_ = nullptr;
};

}  // namespace tessera_test

#endif  // TESSERA_TESTS_FAKE_DRIVER_FAKE_FIXTURE_H
