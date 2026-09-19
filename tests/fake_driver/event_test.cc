// Events: record, query, synchronize and elapsed time, all against the
// simulated clock.

#include "fake_fixture.h"

namespace tessera_test {
namespace {

class EventTest : public FakeDriverTest {
 protected:
  static CUresult launch(CUfunction f, CUstream s) {
    return cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr);
  }
};

TEST_F(EventTest, AnUnrecordedEventIsComplete) {
  CUevent e = nullptr;
  ASSERT_CU(cuEventCreate(&e, CU_EVENT_DEFAULT));
  EXPECT_CU(cuEventQuery(e));
  EXPECT_CU(cuEventSynchronize(e));
  ASSERT_CU(cuEventDestroy(e));
}

TEST_F(EventTest, QueryIsNotReadyUntilTheStreamReachesTheRecord) {
  CUfunction k = tf_register_kernel("k", nullptr, 200);
  CUstream s = nullptr;
  CUevent e = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(cuEventCreate(&e, CU_EVENT_DEFAULT));

  ASSERT_CU(launch(k, s));
  ASSERT_CU(cuEventRecord(e, s));
  EXPECT_EQ(CUDA_ERROR_NOT_READY, cuEventQuery(e));
  tf_advance_time_us(199);
  EXPECT_EQ(CUDA_ERROR_NOT_READY, cuEventQuery(e));
  tf_advance_time_us(1);
  EXPECT_CU(cuEventQuery(e));

  ASSERT_CU(cuEventDestroy(e));
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(EventTest, ElapsedTimeMeasuresTheWorkBetweenTwoRecords) {
  CUfunction k = tf_register_kernel("k", nullptr, 7500);
  CUstream s = nullptr;
  CUevent start = nullptr;
  CUevent stop = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(cuEventCreate(&start, CU_EVENT_DEFAULT));
  ASSERT_CU(cuEventCreate(&stop, CU_EVENT_DEFAULT));

  ASSERT_CU(cuEventRecord(start, s));
  ASSERT_CU(launch(k, s));
  ASSERT_CU(cuEventRecord(stop, s));
  ASSERT_CU(cuStreamSynchronize(s));

  float ms = 0.0F;
  ASSERT_CU(cuEventElapsedTime(&ms, start, stop));
  EXPECT_FLOAT_EQ(7.5F, ms);

  ASSERT_CU(cuEventDestroy(start));
  ASSERT_CU(cuEventDestroy(stop));
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(EventTest, ElapsedTimeAccumulatesAcrossLaunches) {
  CUfunction k = tf_register_kernel("k", nullptr, 1000);
  CUstream s = nullptr;
  CUevent start = nullptr;
  CUevent stop = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(cuEventCreate(&start, CU_EVENT_DEFAULT));
  ASSERT_CU(cuEventCreate(&stop, CU_EVENT_DEFAULT));

  ASSERT_CU(cuEventRecord(start, s));
  for (int i = 0; i < 3; ++i) {
    ASSERT_CU(launch(k, s));
  }
  ASSERT_CU(cuEventRecordWithFlags(stop, s, CU_EVENT_RECORD_DEFAULT));
  ASSERT_CU(cuStreamSynchronize(s));

  float ms = 0.0F;
  ASSERT_CU(cuEventElapsedTime(&ms, start, stop));
  EXPECT_FLOAT_EQ(3.0F, ms);

  ASSERT_CU(cuEventDestroy(start));
  ASSERT_CU(cuEventDestroy(stop));
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(EventTest, ElapsedTimeIsNotReadyWhileWorkIsPending) {
  CUfunction k = tf_register_kernel("k", nullptr, 500);
  CUstream s = nullptr;
  CUevent start = nullptr;
  CUevent stop = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(cuEventCreate(&start, CU_EVENT_DEFAULT));
  ASSERT_CU(cuEventCreate(&stop, CU_EVENT_DEFAULT));

  ASSERT_CU(cuEventRecord(start, s));
  ASSERT_CU(launch(k, s));
  ASSERT_CU(cuEventRecord(stop, s));

  float ms = -1.0F;
  EXPECT_EQ(CUDA_ERROR_NOT_READY, cuEventElapsedTime(&ms, start, stop));

  ASSERT_CU(cuEventSynchronize(stop));
  EXPECT_CU(cuEventElapsedTime(&ms, start, stop));
  EXPECT_FLOAT_EQ(0.5F, ms);

  ASSERT_CU(cuEventDestroy(start));
  ASSERT_CU(cuEventDestroy(stop));
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(EventTest, ElapsedTimeNeedsBothEventsRecorded) {
  CUevent a = nullptr;
  CUevent b = nullptr;
  ASSERT_CU(cuEventCreate(&a, CU_EVENT_DEFAULT));
  ASSERT_CU(cuEventCreate(&b, CU_EVENT_DEFAULT));
  float ms = 0.0F;
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, cuEventElapsedTime(&ms, a, b));
  ASSERT_CU(cuEventDestroy(a));
  ASSERT_CU(cuEventDestroy(b));
}

TEST_F(EventTest, EventSynchronizeJumpsTheManualClock) {
  CUfunction k = tf_register_kernel("k", nullptr, 3000);
  CUstream s = nullptr;
  CUevent e = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(cuEventCreate(&e, CU_EVENT_DEFAULT));

  const uint64_t t0 = tf_now_us();
  ASSERT_CU(launch(k, s));
  ASSERT_CU(cuEventRecord(e, s));
  ASSERT_CU(cuEventSynchronize(e));
  EXPECT_EQ(t0 + 3000u, tf_now_us());
  EXPECT_CU(cuEventQuery(e));

  ASSERT_CU(cuEventDestroy(e));
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(EventTest, RerecordingAnEventMovesItsTimestamp) {
  CUfunction k = tf_register_kernel("k", nullptr, 100);
  CUstream s = nullptr;
  CUevent start = nullptr;
  CUevent stop = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(cuEventCreate(&start, CU_EVENT_DEFAULT));
  ASSERT_CU(cuEventCreate(&stop, CU_EVENT_DEFAULT));

  ASSERT_CU(cuEventRecord(start, s));
  ASSERT_CU(launch(k, s));
  ASSERT_CU(cuEventRecord(stop, s));
  ASSERT_CU(cuStreamSynchronize(s));

  float ms = -1.0F;
  ASSERT_CU(cuEventElapsedTime(&ms, start, stop));
  EXPECT_FLOAT_EQ(0.1F, ms);

  // Re-record start on the now-idle stream: it moves to the current time, so
  // the interval from stop to the new start is empty.
  ASSERT_CU(cuEventRecord(start, s));
  ASSERT_CU(cuStreamSynchronize(s));
  ASSERT_CU(cuEventElapsedTime(&ms, stop, start));
  EXPECT_FLOAT_EQ(0.0F, ms);

  ASSERT_CU(cuEventDestroy(start));
  ASSERT_CU(cuEventDestroy(stop));
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(EventTest, DestroyedEventHandleIsRejected) {
  CUevent e = nullptr;
  ASSERT_CU(cuEventCreate(&e, CU_EVENT_DEFAULT));
  ASSERT_CU(cuEventDestroy(e));
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, cuEventQuery(e));
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, cuEventDestroy(e));
  EXPECT_EQ(CUDA_ERROR_INVALID_HANDLE, cuEventRecord(e, nullptr));
}

}  // namespace
}  // namespace tessera_test
