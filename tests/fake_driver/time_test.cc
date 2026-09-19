// The two time modes. Manual is what makes the rest of the suite
// deterministic; scaled-real is what a shim benchmark runs against.

#include <chrono>
#include <thread>

#include "fake_fixture.h"

namespace tessera_test {
namespace {

using Clock = std::chrono::steady_clock;

class TimeTest : public FakeDriverTest {
 protected:
  static CUresult launch(CUfunction f, CUstream s) {
    return cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr);
  }
};

TEST_F(TimeTest, ManualClockStandsStillOnItsOwn) {
  const uint64_t t0 = tf_now_us();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(t0, tf_now_us());
}

TEST_F(TimeTest, AdvanceMovesTheManualClockExactly) {
  const uint64_t t0 = tf_now_us();
  tf_advance_time_us(1234);
  EXPECT_EQ(t0 + 1234u, tf_now_us());
  tf_advance_time_us(1);
  EXPECT_EQ(t0 + 1235u, tf_now_us());
}

TEST_F(TimeTest, ScaledRealClockFollowsTheWallClock) {
  tf_set_time_mode(0);
  const uint64_t t0 = tf_now_us();
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  const uint64_t t1 = tf_now_us();
  // A 30 ms sleep must show up as at least 10 ms of simulated time even on a
  // heavily loaded or emulated machine. No upper bound: this is real time.
  EXPECT_GE(t1 - t0, 10000u);
}

TEST_F(TimeTest, SwitchingModesPreservesTheCurrentTime) {
  tf_advance_time_us(5000);
  const uint64_t before = tf_now_us();
  tf_set_time_mode(0);
  const uint64_t after = tf_now_us();
  EXPECT_GE(after, before);
  EXPECT_LT(after - before, 1000000u);
  tf_set_time_mode(1);
  const uint64_t frozen = tf_now_us();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(frozen, tf_now_us());
  EXPECT_GE(frozen, before);
}

TEST_F(TimeTest, ManualModeSynchronizeJumpsTheClockInsteadOfWaiting) {
  CUfunction k = tf_register_kernel("slow", nullptr, 5'000'000);  // 5 simulated seconds
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  const uint64_t t0 = tf_now_us();

  const Clock::time_point wall0 = Clock::now();
  ASSERT_CU(launch(k, s));
  ASSERT_CU(cuStreamSynchronize(s));
  const auto wall_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - wall0).count();

  EXPECT_EQ(t0 + 5'000'000u, tf_now_us());
  // Five simulated seconds, but the call must not have waited for them.
  EXPECT_LT(wall_ms, 2000);
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(TimeTest, ScaledModeSynchronizeReallyWaits) {
  tf_set_time_mode(0);
  CUfunction k = tf_register_kernel("twenty_ms", nullptr, 20000);
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));

  const Clock::time_point wall0 = Clock::now();
  ASSERT_CU(launch(k, s));
  ASSERT_CU(cuStreamSynchronize(s));
  const auto wall_us =
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - wall0).count();

  EXPECT_GE(wall_us, 15000);
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(TimeTest, WorkIsNotCompletedBeforeItsDeadline) {
  CUfunction k = tf_register_kernel("k", nullptr, 100);
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(launch(k, s));
  ASSERT_CU(launch(k, s));
  tf_advance_time_us(150);
  // The first launch is done at +100, the second only at +200.
  EXPECT_EQ(CUDA_ERROR_NOT_READY, cuStreamQuery(s));
  tf_advance_time_us(50);
  EXPECT_CU(cuStreamQuery(s));
  ASSERT_CU(cuStreamDestroy(s));
}

}  // namespace
}  // namespace tessera_test
