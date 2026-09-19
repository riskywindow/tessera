// The legacy and _ptsz/_ptds entry points are separate exported symbols with
// separately observable behaviour. Without that, a shim test could not tell
// which of the two an application actually reached.

#include <thread>

#include "fake_fixture.h"

namespace tessera_test {
namespace {

class PerThreadStreamTest : public FakeDriverTest {
 protected:
  CUfunction kernel() { return tf_register_kernel("k", nullptr, 1); }
};

TEST_F(PerThreadStreamTest, DefaultStreamsAreDistinctObjects) {
  CUstream legacy = tf_legacy_default_stream();
  CUstream per_thread = tf_per_thread_default_stream();
  ASSERT_NE(nullptr, legacy);
  ASSERT_NE(nullptr, per_thread);
  EXPECT_NE(legacy, per_thread);
}

TEST_F(PerThreadStreamTest, ANullStreamMeansDifferentThingsToTheTwoEntryPoints) {
  CUfunction k = kernel();
  CUstream legacy = tf_legacy_default_stream();
  CUstream per_thread = tf_per_thread_default_stream();

  ASSERT_CU(cuLaunchKernel(k, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  EXPECT_EQ(0, tf_last_stream_was_per_thread_default());
  EXPECT_EQ(1u, tf_stream_launch_count(legacy));
  EXPECT_EQ(0u, tf_stream_launch_count(per_thread));

  ASSERT_CU(cuLaunchKernel_ptsz(k, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  EXPECT_EQ(1, tf_last_stream_was_per_thread_default());
  EXPECT_EQ(1u, tf_stream_launch_count(legacy));
  EXPECT_EQ(1u, tf_stream_launch_count(per_thread));

  EXPECT_EQ(1u, tf_symbol_count("cuLaunchKernel"));
  EXPECT_EQ(1u, tf_symbol_count("cuLaunchKernel_ptsz"));
  ASSERT_CU(cuCtxSynchronize());
}

TEST_F(PerThreadStreamTest, TheReservedTokensOverrideTheEntryPointsChoice) {
  CUfunction k = kernel();
  CUstream legacy = tf_legacy_default_stream();
  CUstream per_thread = tf_per_thread_default_stream();

  // CU_STREAM_PER_THREAD through a legacy entry point.
  ASSERT_CU(cuLaunchKernel(k, 1, 1, 1, 1, 1, 1, 0, per_thread_stream_token(), nullptr, nullptr));
  EXPECT_EQ(1, tf_last_stream_was_per_thread_default());
  EXPECT_EQ(1u, tf_stream_launch_count(per_thread));

  // CU_STREAM_LEGACY through a per-thread entry point.
  ASSERT_CU(cuLaunchKernel_ptsz(k, 1, 1, 1, 1, 1, 1, 0, legacy_stream_token(), nullptr, nullptr));
  EXPECT_EQ(0, tf_last_stream_was_per_thread_default());
  EXPECT_EQ(1u, tf_stream_launch_count(legacy));
  ASSERT_CU(cuCtxSynchronize());
}

TEST_F(PerThreadStreamTest, AnExplicitStreamIsUsedByBothEntryPoints) {
  CUfunction k = kernel();
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(cuLaunchKernel(k, 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr));
  EXPECT_EQ(0, tf_last_stream_was_per_thread_default());
  ASSERT_CU(cuLaunchKernel_ptsz(k, 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr));
  EXPECT_EQ(0, tf_last_stream_was_per_thread_default());
  EXPECT_EQ(2u, tf_stream_launch_count(s));
  ASSERT_CU(cuStreamSynchronize(s));
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(PerThreadStreamTest, EachThreadGetsItsOwnPerThreadDefaultStream) {
  CUstream mine = tf_per_thread_default_stream();
  CUstream theirs = nullptr;
  std::thread t([&] {
    ASSERT_CU(cuCtxSetCurrent(ctx_));
    theirs = tf_per_thread_default_stream();
  });
  t.join();
  ASSERT_NE(nullptr, theirs);
  EXPECT_NE(mine, theirs);
}

TEST_F(PerThreadStreamTest, PerThreadDefaultStreamsDoNotSerialiseAcrossThreads) {
  CUfunction k = tf_register_kernel("slow", nullptr, 1000);
  const uint64_t t0 = tf_now_us();

  // Both launches happen before anything is synchronized, so both streams see
  // the same start time; only then does the clock move.
  std::thread t([&] {
    ASSERT_CU(cuCtxSetCurrent(ctx_));
    ASSERT_CU(cuLaunchKernel_ptsz(k, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  });
  ASSERT_CU(cuLaunchKernel_ptsz(k, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  t.join();

  // Two 1000 us kernels on two different per-thread streams, not one queue.
  ASSERT_CU(cuStreamSynchronize_ptsz(nullptr));
  ASSERT_CU(cuCtxSynchronize());
  EXPECT_EQ(t0 + 1000u, tf_now_us());
}

TEST_F(PerThreadStreamTest, CopyEntryPointsPickTheirOwnDefaultStream) {
  CUdeviceptr d = 0;
  ASSERT_CU(cuMemAlloc(&d, 64));
  unsigned char host[64] = {};

  ASSERT_CU(cuMemcpyHtoD(d, host, sizeof(host)));
  EXPECT_EQ(0, tf_last_stream_was_per_thread_default());
  EXPECT_EQ(1u, tf_symbol_count("cuMemcpyHtoD_v2"));
  EXPECT_EQ(0u, tf_symbol_count("cuMemcpyHtoD_v2_ptds"));

  ASSERT_CU(cuMemcpyHtoD_v2_ptds(d, host, sizeof(host)));
  EXPECT_EQ(1, tf_last_stream_was_per_thread_default());
  EXPECT_EQ(1u, tf_symbol_count("cuMemcpyHtoD_v2"));
  EXPECT_EQ(1u, tf_symbol_count("cuMemcpyHtoD_v2_ptds"));

  ASSERT_CU(cuMemFree(d));
}

TEST_F(PerThreadStreamTest, SynchronizeEntryPointsPickTheirOwnDefaultStream) {
  CUfunction k = tf_register_kernel("k", nullptr, 500);
  const uint64_t t0 = tf_now_us();

  // Queue on the per-thread default stream, then synchronize the LEGACY one:
  // that must not wait for it.
  ASSERT_CU(cuLaunchKernel_ptsz(k, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  ASSERT_CU(cuStreamSynchronize(nullptr));
  EXPECT_EQ(t0, tf_now_us());
  EXPECT_EQ(CUDA_ERROR_NOT_READY, cuStreamQuery_ptsz(nullptr));

  ASSERT_CU(cuStreamSynchronize_ptsz(nullptr));
  EXPECT_EQ(t0 + 500u, tf_now_us());
  EXPECT_CU(cuStreamQuery_ptsz(nullptr));

  EXPECT_EQ(1u, tf_symbol_count("cuStreamSynchronize"));
  EXPECT_EQ(1u, tf_symbol_count("cuStreamSynchronize_ptsz"));
}

TEST_F(PerThreadStreamTest, EventRecordEntryPointsPickTheirOwnDefaultStream) {
  CUfunction k = tf_register_kernel("k", nullptr, 300);
  CUevent legacy_event = nullptr;
  CUevent pt_event = nullptr;
  ASSERT_CU(cuEventCreate(&legacy_event, CU_EVENT_DEFAULT));
  ASSERT_CU(cuEventCreate(&pt_event, CU_EVENT_DEFAULT));

  ASSERT_CU(cuLaunchKernel_ptsz(k, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  ASSERT_CU(cuEventRecord(legacy_event, nullptr));
  EXPECT_EQ(0, tf_last_stream_was_per_thread_default());
  ASSERT_CU(cuEventRecord_ptsz(pt_event, nullptr));
  EXPECT_EQ(1, tf_last_stream_was_per_thread_default());

  // The legacy default stream is empty, so its event is already due; let the
  // worker catch up with the clock before asking.
  tf_advance_time_us(0);
  EXPECT_CU(cuEventQuery(legacy_event));
  EXPECT_EQ(CUDA_ERROR_NOT_READY, cuEventQuery(pt_event));

  ASSERT_CU(cuCtxSynchronize());
  ASSERT_CU(cuEventDestroy(legacy_event));
  ASSERT_CU(cuEventDestroy(pt_event));
}

TEST_F(PerThreadStreamTest, EveryPerThreadEntryPointIsCountedSeparately) {
  CUfunction k = tf_register_kernel("k", nullptr, 0);
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));

  ASSERT_CU(cuLaunchCooperativeKernel(k, 1, 1, 1, 1, 1, 1, 0, s, nullptr));
  ASSERT_CU(cuLaunchCooperativeKernel_ptsz(k, 1, 1, 1, 1, 1, 1, 0, s, nullptr));
  int priority = 0;
  ASSERT_CU(cuStreamGetPriority(s, &priority));
  ASSERT_CU(cuStreamGetPriority_ptsz(s, &priority));
  unsigned int flags = 0;
  ASSERT_CU(cuStreamGetFlags(s, &flags));
  ASSERT_CU(cuStreamGetFlags_ptsz(s, &flags));

  EXPECT_EQ(1u, tf_symbol_count("cuLaunchCooperativeKernel"));
  EXPECT_EQ(1u, tf_symbol_count("cuLaunchCooperativeKernel_ptsz"));
  EXPECT_EQ(1u, tf_symbol_count("cuStreamGetPriority"));
  EXPECT_EQ(1u, tf_symbol_count("cuStreamGetPriority_ptsz"));
  EXPECT_EQ(1u, tf_symbol_count("cuStreamGetFlags"));
  EXPECT_EQ(1u, tf_symbol_count("cuStreamGetFlags_ptsz"));

  ASSERT_CU(cuStreamSynchronize(s));
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(PerThreadStreamTest, LaunchKernelExUsesTheConfigsStream) {
  CUfunction k = tf_register_kernel("k", nullptr, 0);
  CUlaunchConfig config{};
  config.gridDimX = 1;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 1;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = nullptr;
  config.attrs = nullptr;
  config.numAttrs = 0;

  ASSERT_CU(cuLaunchKernelEx(&config, k, nullptr, nullptr));
  EXPECT_EQ(0, tf_last_stream_was_per_thread_default());
  ASSERT_CU(cuLaunchKernelEx_ptsz(&config, k, nullptr, nullptr));
  EXPECT_EQ(1, tf_last_stream_was_per_thread_default());
  EXPECT_EQ(1u, tf_stream_launch_count(tf_legacy_default_stream()));
  EXPECT_EQ(1u, tf_stream_launch_count(tf_per_thread_default_stream()));
  ASSERT_CU(cuCtxSynchronize());
}

}  // namespace
}  // namespace tessera_test
