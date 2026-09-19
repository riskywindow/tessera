// Stream capture and graph replay. The fake's graph is deliberately minimal:
// a capture records the list of operations submitted to the stream, and a
// launch replays that list. That is enough for the shim, which must not gate
// or inject events while a stream is capturing.

#include <vector>

#include "fake_fixture.h"

namespace tessera_test {
namespace {

std::vector<int>* g_trace = nullptr;
void trace_push(int v) {
  if (g_trace != nullptr) {
    g_trace->push_back(v);
  }
}
void node_a(void**) {
  trace_push(1);
}
void node_b(void**) {
  trace_push(2);
}

class GraphTest : public FakeDriverTest {
 protected:
  void SetUp() override {
    FakeDriverTest::SetUp();
    trace_.clear();
    g_trace = &trace_;
    ASSERT_CU(cuStreamCreate(&stream_, 0));
  }
  void TearDown() override {
    FakeDriverTest::TearDown();
    g_trace = nullptr;
  }

  static CUresult launch(CUfunction f, CUstream s) {
    return cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr);
  }

  std::vector<int> trace_;
  CUstream stream_ = nullptr;
};

TEST_F(GraphTest, CaptureRecordsWithoutExecuting) {
  CUfunction a = tf_register_kernel("a", node_a, 10);
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_INVALIDATED;

  ASSERT_CU(cuStreamIsCapturing(stream_, &status));
  EXPECT_EQ(CU_STREAM_CAPTURE_STATUS_NONE, status);

  ASSERT_CU(cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_GLOBAL));
  ASSERT_CU(cuStreamIsCapturing(stream_, &status));
  EXPECT_EQ(CU_STREAM_CAPTURE_STATUS_ACTIVE, status);

  ASSERT_CU(launch(a, stream_));
  ASSERT_CU(launch(a, stream_));
  tf_advance_time_us(1000);
  EXPECT_TRUE(trace_.empty()) << "captured launches must not run";

  CUgraph graph = nullptr;
  ASSERT_CU(cuStreamEndCapture(stream_, &graph));
  ASSERT_NE(nullptr, graph);
  ASSERT_CU(cuStreamIsCapturing(stream_, &status));
  EXPECT_EQ(CU_STREAM_CAPTURE_STATUS_NONE, status);
  ASSERT_CU(cuGraphDestroy(graph));
}

TEST_F(GraphTest, ReplayRunsTheCapturedListInOrderAndCostsItsTime) {
  CUfunction a = tf_register_kernel("a", node_a, 10);
  CUfunction b = tf_register_kernel("b", node_b, 30);

  ASSERT_CU(cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_GLOBAL));
  ASSERT_CU(launch(a, stream_));
  ASSERT_CU(launch(b, stream_));
  ASSERT_CU(launch(a, stream_));
  CUgraph graph = nullptr;
  ASSERT_CU(cuStreamEndCapture(stream_, &graph));

  CUgraphExec exec = nullptr;
  ASSERT_CU(cuGraphInstantiate(&exec, graph, 0));
  ASSERT_NE(nullptr, exec);

  const uint64_t t0 = tf_now_us();
  ASSERT_CU(cuGraphLaunch(exec, stream_));
  ASSERT_CU(cuStreamSynchronize(stream_));
  EXPECT_EQ((std::vector<int>{1, 2, 1}), trace_);
  EXPECT_EQ(t0 + 50u, tf_now_us());

  // Replaying it again runs the same list again.
  ASSERT_CU(cuGraphLaunch(exec, stream_));
  ASSERT_CU(cuStreamSynchronize(stream_));
  EXPECT_EQ((std::vector<int>{1, 2, 1, 1, 2, 1}), trace_);
  EXPECT_EQ(t0 + 100u, tf_now_us());
  EXPECT_EQ(6u, tf_stream_launch_count(stream_));

  ASSERT_CU(cuGraphExecDestroy(exec));
  ASSERT_CU(cuGraphDestroy(graph));
}

TEST_F(GraphTest, CapturedCopiesAndFillsReplayToo) {
  CUdeviceptr d = 0;
  ASSERT_CU(cuMemAlloc(&d, 16));
  unsigned int words[4] = {};

  ASSERT_CU(cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_GLOBAL));
  ASSERT_CU(cuMemsetD32Async(d, 0xFEEDFACE, 4, stream_));
  ASSERT_CU(cuMemcpyDtoHAsync(words, d, sizeof(words), stream_));
  CUgraph graph = nullptr;
  ASSERT_CU(cuStreamEndCapture(stream_, &graph));

  CUgraphExec exec = nullptr;
  ASSERT_CU(cuGraphInstantiate(&exec, graph, 0));
  EXPECT_EQ(0u, words[0]);
  ASSERT_CU(cuGraphLaunch(exec, stream_));
  ASSERT_CU(cuStreamSynchronize(stream_));
  for (unsigned int w : words) {
    EXPECT_EQ(0xFEEDFACEu, w);
  }

  ASSERT_CU(cuGraphExecDestroy(exec));
  ASSERT_CU(cuGraphDestroy(graph));
  ASSERT_CU(cuMemFree(d));
}

TEST_F(GraphTest, SynchronizingACapturingStreamIsRefused) {
  ASSERT_CU(cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_GLOBAL));
  EXPECT_EQ(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, cuStreamSynchronize(stream_));
  CUgraph graph = nullptr;
  ASSERT_CU(cuStreamEndCapture(stream_, &graph));
  ASSERT_CU(cuGraphDestroy(graph));
}

TEST_F(GraphTest, TheLegacyDefaultStreamCannotBeCaptured) {
  EXPECT_EQ(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
            cuStreamBeginCapture(nullptr, CU_STREAM_CAPTURE_MODE_GLOBAL));
  EXPECT_EQ(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
            cuStreamBeginCapture(legacy_stream_token(), CU_STREAM_CAPTURE_MODE_GLOBAL));
}

TEST_F(GraphTest, CaptureCannotBeStartedTwiceOrEndedWithoutStarting) {
  CUgraph graph = nullptr;
  EXPECT_EQ(CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, cuStreamEndCapture(stream_, &graph));
  ASSERT_CU(cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_GLOBAL));
  EXPECT_EQ(CUDA_ERROR_ILLEGAL_STATE, cuStreamBeginCapture(stream_, CU_STREAM_CAPTURE_MODE_GLOBAL));
  ASSERT_CU(cuStreamEndCapture(stream_, &graph));
  ASSERT_CU(cuGraphDestroy(graph));
}

TEST_F(GraphTest, TheVersionOneCaptureEntryPointWorks) {
  auto begin_v1 =
      reinterpret_cast<TF_PFN_cuStreamBeginCapture_v1>(tf_symbol_address("cuStreamBeginCapture"));
  ASSERT_NE(nullptr, begin_v1);
  CUfunction a = tf_register_kernel("a", node_a, 5);

  ASSERT_CU(begin_v1(stream_));
  ASSERT_CU(launch(a, stream_));
  CUgraph graph = nullptr;
  ASSERT_CU(cuStreamEndCapture(stream_, &graph));
  EXPECT_EQ(1u, tf_symbol_count("cuStreamBeginCapture"));
  EXPECT_EQ(0u, tf_symbol_count("cuStreamBeginCapture_v2"));

  CUgraphExec exec = nullptr;
  ASSERT_CU(cuGraphInstantiate(&exec, graph, 0));
  ASSERT_CU(cuGraphLaunch(exec, stream_));
  ASSERT_CU(cuStreamSynchronize(stream_));
  EXPECT_EQ((std::vector<int>{1}), trace_);

  ASSERT_CU(cuGraphExecDestroy(exec));
  ASSERT_CU(cuGraphDestroy(graph));
}

TEST_F(GraphTest, GraphLaunchHasBothStreamVariants) {
  CUfunction a = tf_register_kernel("a", node_a, 5);
  ASSERT_CU(cuStreamBeginCapture_v2_ptsz(stream_, CU_STREAM_CAPTURE_MODE_GLOBAL));
  ASSERT_CU(launch(a, stream_));
  CUgraph graph = nullptr;
  ASSERT_CU(cuStreamEndCapture_ptsz(stream_, &graph));
  CUgraphExec exec = nullptr;
  ASSERT_CU(cuGraphInstantiate(&exec, graph, 0));

  ASSERT_CU(cuGraphLaunch(exec, nullptr));
  EXPECT_EQ(0, tf_last_stream_was_per_thread_default());
  ASSERT_CU(cuGraphLaunch_ptsz(exec, nullptr));
  EXPECT_EQ(1, tf_last_stream_was_per_thread_default());
  EXPECT_EQ(1u, tf_stream_launch_count(tf_legacy_default_stream()));
  EXPECT_EQ(1u, tf_stream_launch_count(tf_per_thread_default_stream()));
  EXPECT_EQ(1u, tf_symbol_count("cuStreamBeginCapture_v2_ptsz"));
  EXPECT_EQ(1u, tf_symbol_count("cuStreamEndCapture_ptsz"));

  ASSERT_CU(cuCtxSynchronize());
  EXPECT_EQ(2u, trace_.size());
  ASSERT_CU(cuGraphExecDestroy(exec));
  ASSERT_CU(cuGraphDestroy(graph));
}

TEST_F(GraphTest, AnEmptyGraphCanBeCreatedAndLaunched) {
  CUgraph graph = nullptr;
  ASSERT_CU(cuGraphCreate(&graph, 0));
  CUgraphExec exec = nullptr;
  ASSERT_CU(cuGraphInstantiate(&exec, graph, 0));
  ASSERT_CU(cuGraphLaunch(exec, stream_));
  ASSERT_CU(cuStreamSynchronize(stream_));
  EXPECT_TRUE(trace_.empty());
  ASSERT_CU(cuGraphExecDestroy(exec));
  ASSERT_CU(cuGraphDestroy(graph));
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, cuGraphDestroy(graph));
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, cuGraphExecDestroy(exec));
}

}  // namespace
}  // namespace tessera_test
