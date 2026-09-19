// Streams are FIFOs: work on one stream completes in submission order and each
// kernel occupies its stream for its full simulated duration. Work on separate
// streams overlaps.

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "fake_fixture.h"

namespace tessera_test {
namespace {

// Kernel bodies are C function pointers, so the trace they append to has to be
// a namespace-scope object. Only the simulation's worker thread writes it, and
// it does so under the simulation lock, which a test's cuStreamSynchronize also
// takes: that is the happens-before edge the reads below rely on.
std::vector<int>* g_trace = nullptr;

void trace_push(int v) {
  if (g_trace != nullptr) {
    g_trace->push_back(v);
  }
}
void trace_a(void**) {
  trace_push(1);
}
void trace_b(void**) {
  trace_push(2);
}
void trace_c(void**) {
  trace_push(3);
}

class StreamTest : public FakeDriverTest {
 protected:
  void SetUp() override {
    FakeDriverTest::SetUp();
    trace_.clear();
    g_trace = &trace_;
  }
  void TearDown() override {
    FakeDriverTest::TearDown();
    g_trace = nullptr;
  }

  static CUresult launch(CUfunction f, CUstream s) {
    return cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr);
  }

  std::vector<int> trace_;
};

TEST_F(StreamTest, WorkOnOneStreamCompletesInSubmissionOrder) {
  CUfunction a = tf_register_kernel("a", trace_a, 30);
  CUfunction b = tf_register_kernel("b", trace_b, 10);
  CUfunction c = tf_register_kernel("c", trace_c, 20);
  ASSERT_NE(nullptr, a);

  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(launch(a, s));
  ASSERT_CU(launch(b, s));
  ASSERT_CU(launch(c, s));
  ASSERT_CU(cuStreamSynchronize(s));

  // Submission order, not duration order.
  EXPECT_EQ((std::vector<int>{1, 2, 3}), trace_);
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(StreamTest, KernelDurationsAccumulateOnAStream) {
  CUfunction k = tf_register_kernel("k", nullptr, 25);
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  const uint64_t t0 = tf_now_us();
  for (int i = 0; i < 4; ++i) {
    ASSERT_CU(launch(k, s));
  }
  ASSERT_CU(cuStreamSynchronize(s));
  EXPECT_EQ(t0 + 100u, tf_now_us());
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(StreamTest, SeparateStreamsOverlap) {
  CUfunction k = tf_register_kernel("k", nullptr, 50);
  CUstream s1 = nullptr;
  CUstream s2 = nullptr;
  ASSERT_CU(cuStreamCreate(&s1, 0));
  ASSERT_CU(cuStreamCreate(&s2, 0));
  const uint64_t t0 = tf_now_us();
  ASSERT_CU(launch(k, s1));
  ASSERT_CU(launch(k, s2));
  ASSERT_CU(cuStreamSynchronize(s1));
  ASSERT_CU(cuStreamSynchronize(s2));
  // Two 50 us kernels on two streams finish 50 us from now, not 100 us.
  EXPECT_EQ(t0 + 50u, tf_now_us());
  ASSERT_CU(cuStreamDestroy(s1));
  ASSERT_CU(cuStreamDestroy(s2));
}

TEST_F(StreamTest, ChangingAKernelDurationChangesTheNextLaunch) {
  CUfunction k = tf_register_kernel("k", nullptr, 10);
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  const uint64_t t0 = tf_now_us();
  ASSERT_CU(launch(k, s));
  tf_set_kernel_duration_us(k, 90);
  ASSERT_CU(launch(k, s));
  ASSERT_CU(cuStreamSynchronize(s));
  EXPECT_EQ(t0 + 100u, tf_now_us());
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(StreamTest, QueryReportsPendingWorkUntilTimeAdvances) {
  CUfunction k = tf_register_kernel("k", trace_a, 100);
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(launch(k, s));
  EXPECT_EQ(CUDA_ERROR_NOT_READY, cuStreamQuery(s));
  tf_advance_time_us(99);
  EXPECT_EQ(CUDA_ERROR_NOT_READY, cuStreamQuery(s));
  EXPECT_TRUE(trace_.empty());
  tf_advance_time_us(1);
  EXPECT_CU(cuStreamQuery(s));
  EXPECT_EQ((std::vector<int>{1}), trace_);
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(StreamTest, LaunchCountIsPerStream) {
  CUfunction k = tf_register_kernel("k", nullptr, 1);
  CUstream s1 = nullptr;
  CUstream s2 = nullptr;
  ASSERT_CU(cuStreamCreate(&s1, 0));
  ASSERT_CU(cuStreamCreate(&s2, 0));
  for (int i = 0; i < 3; ++i) {
    ASSERT_CU(launch(k, s1));
  }
  ASSERT_CU(launch(k, s2));
  EXPECT_EQ(3u, tf_stream_launch_count(s1));
  EXPECT_EQ(1u, tf_stream_launch_count(s2));
  ASSERT_CU(cuStreamSynchronize(s1));
  ASSERT_CU(cuStreamSynchronize(s2));
  ASSERT_CU(cuStreamDestroy(s1));
  ASSERT_CU(cuStreamDestroy(s2));
}

TEST_F(StreamTest, WaitEventDelaysTheWaitingStream) {
  CUfunction slow = tf_register_kernel("slow", trace_a, 100);
  CUfunction fast = tf_register_kernel("fast", trace_b, 1);
  CUstream s1 = nullptr;
  CUstream s2 = nullptr;
  CUevent e = nullptr;
  ASSERT_CU(cuStreamCreate(&s1, 0));
  ASSERT_CU(cuStreamCreate(&s2, 0));
  ASSERT_CU(cuEventCreate(&e, CU_EVENT_DEFAULT));

  const uint64_t t0 = tf_now_us();
  ASSERT_CU(launch(slow, s1));
  ASSERT_CU(cuEventRecord(e, s1));
  ASSERT_CU(cuStreamWaitEvent(s2, e, 0));
  ASSERT_CU(launch(fast, s2));
  ASSERT_CU(cuStreamSynchronize(s2));

  EXPECT_EQ(t0 + 101u, tf_now_us());
  EXPECT_EQ((std::vector<int>{1, 2}), trace_);

  ASSERT_CU(cuStreamSynchronize(s1));
  ASSERT_CU(cuEventDestroy(e));
  ASSERT_CU(cuStreamDestroy(s1));
  ASSERT_CU(cuStreamDestroy(s2));
}

TEST_F(StreamTest, StreamPriorityAndFlagsAreReported) {
  int least = 0;
  int greatest = 0;
  ASSERT_CU(cuCtxGetStreamPriorityRange(&least, &greatest));
  EXPECT_LT(greatest, least);

  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreateWithPriority(&s, CU_STREAM_NON_BLOCKING, -3));
  int priority = 0;
  unsigned int flags = 0;
  ASSERT_CU(cuStreamGetPriority(s, &priority));
  ASSERT_CU(cuStreamGetFlags(s, &flags));
  EXPECT_EQ(-3, priority);
  EXPECT_EQ(static_cast<unsigned int>(CU_STREAM_NON_BLOCKING), flags);
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(StreamTest, DestroyedStreamHandleIsRejected) {
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(cuStreamDestroy(s));
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, cuStreamSynchronize(s));
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, cuStreamDestroy(s));
  EXPECT_EQ(0u, tf_stream_launch_count(s));
}

TEST_F(StreamTest, DefaultStreamsCannotBeDestroyed) {
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, cuStreamDestroy(tf_legacy_default_stream()));
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, cuStreamDestroy(tf_per_thread_default_stream()));
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, cuStreamDestroy(legacy_stream_token()));
}

TEST_F(StreamTest, ContextSynchronizeDrainsEveryStream) {
  CUfunction k = tf_register_kernel("k", trace_a, 40);
  CUstream s1 = nullptr;
  CUstream s2 = nullptr;
  ASSERT_CU(cuStreamCreate(&s1, 0));
  ASSERT_CU(cuStreamCreate(&s2, 0));
  const uint64_t t0 = tf_now_us();
  ASSERT_CU(launch(k, s1));
  ASSERT_CU(launch(k, s2));
  ASSERT_CU(launch(k, nullptr));
  ASSERT_CU(cuCtxSynchronize());
  EXPECT_EQ(3u, trace_.size());
  EXPECT_EQ(t0 + 40u, tf_now_us());
  ASSERT_CU(cuStreamDestroy(s1));
  ASSERT_CU(cuStreamDestroy(s2));
}

// --- thread safety ---------------------------------------------------------

std::atomic<uint64_t> g_body_calls{0};
void counting_body(void**) {
  g_body_calls.fetch_add(1, std::memory_order_relaxed);
}

TEST_F(StreamTest, ConcurrentLaunchesOnSeveralStreams) {
  constexpr int kThreads = 4;
  constexpr int kLaunches = 200;
  g_body_calls.store(0);
  CUfunction k = tf_register_kernel("concurrent", counting_body, 1);

  std::vector<CUstream> streams(kThreads, nullptr);
  for (int i = 0; i < kThreads; ++i) {
    ASSERT_CU(cuStreamCreate(&streams[i], 0));
  }

  std::vector<std::thread> workers;
  std::atomic<int> failures{0};
  workers.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    workers.emplace_back([&, i] {
      ASSERT_CU(cuCtxSetCurrent(ctx_));
      for (int n = 0; n < kLaunches; ++n) {
        if (launch(k, streams[i]) != CUDA_SUCCESS) {
          failures.fetch_add(1);
        }
      }
      if (cuStreamSynchronize(streams[i]) != CUDA_SUCCESS) {
        failures.fetch_add(1);
      }
    });
  }
  for (std::thread& t : workers) {
    t.join();
  }

  EXPECT_EQ(0, failures.load());
  EXPECT_EQ(static_cast<uint64_t>(kThreads) * kLaunches, g_body_calls.load());
  for (int i = 0; i < kThreads; ++i) {
    EXPECT_EQ(static_cast<uint64_t>(kLaunches), tf_stream_launch_count(streams[i]));
    ASSERT_CU(cuStreamDestroy(streams[i]));
  }
}

TEST_F(StreamTest, ConcurrentLaunchesOnOneStreamAllRun) {
  constexpr int kThreads = 3;
  constexpr int kLaunches = 100;
  g_body_calls.store(0);
  CUfunction k = tf_register_kernel("shared", counting_body, 2);
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    workers.emplace_back([&] {
      ASSERT_CU(cuCtxSetCurrent(ctx_));
      for (int n = 0; n < kLaunches; ++n) {
        ASSERT_CU(launch(k, s));
      }
    });
  }
  for (std::thread& t : workers) {
    t.join();
  }
  ASSERT_CU(cuStreamSynchronize(s));

  const uint64_t total = static_cast<uint64_t>(kThreads) * kLaunches;
  EXPECT_EQ(total, g_body_calls.load());
  EXPECT_EQ(total, tf_stream_launch_count(s));
  ASSERT_CU(cuStreamDestroy(s));
}

}  // namespace
}  // namespace tessera_test
