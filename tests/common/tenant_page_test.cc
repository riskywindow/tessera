// The TenantPage contract is exercised the way it is actually used: mapped
// MAP_SHARED into two processes, updated with atomics, and waited on with a
// futex. The layout rules themselves are static_asserts in the header, so
// this covers the runtime behaviour the shim's hot path depends on.
#include "tessera/common/tenant_page.h"

#include <gtest/gtest.h>

#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>

namespace tessera {
namespace {

int FutexWait(std::atomic<std::uint32_t>* addr, std::uint32_t expected,
              std::chrono::nanoseconds timeout) {
  struct timespec ts {};
  ts.tv_sec = static_cast<time_t>(timeout.count() / 1'000'000'000);
  ts.tv_nsec = static_cast<long>(timeout.count() % 1'000'000'000);
  return static_cast<int>(syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(addr),
                                  FUTEX_WAIT_PRIVATE, expected, &ts, nullptr, 0));
}

int FutexWake(std::atomic<std::uint32_t>* addr, int count) {
  return static_cast<int>(syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(addr),
                                  FUTEX_WAKE_PRIVATE, count, nullptr, nullptr, 0));
}

class TenantPageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    void* mem = mmap(nullptr, kTenantPageSize, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(mem, MAP_FAILED) << std::strerror(errno);
    std::memset(mem, 0, kTenantPageSize);
    page_ = new (mem) TenantPage{};
    page_->magic.store(kTenantPageMagic, std::memory_order_release);
    page_->layout_version.store(kTenantPageLayoutVersion, std::memory_order_release);
  }

  void TearDown() override {
    ASSERT_EQ(munmap(page_, kTenantPageSize), 0) << std::strerror(errno);
  }

  TenantPage* page_ = nullptr;
};

// A page must be describable by its first bytes alone: a shim that maps a
// stale or foreign file has to be able to tell.
TEST_F(TenantPageTest, IdentifiesItself) {
  EXPECT_EQ(page_->magic.load(std::memory_order_acquire), kTenantPageMagic);
  EXPECT_EQ(page_->layout_version.load(std::memory_order_acquire),
            kTenantPageLayoutVersion);
}

// The debit is the whole fast path: one fetch_sub whose previous value decides
// whether the launch proceeds. Credits are allowed to go negative.
TEST_F(TenantPageTest, CreditDebitReportsPriorBalance) {
  page_->credits_us.store(100, std::memory_order_relaxed);

  const std::int64_t before = page_->credits_us.fetch_sub(30, std::memory_order_acq_rel);
  EXPECT_EQ(before, 100);
  EXPECT_GE(before, 30) << "launch should have been admitted";
  EXPECT_EQ(page_->credits_us.load(std::memory_order_relaxed), 70);

  const std::int64_t before2 = page_->credits_us.fetch_sub(200, std::memory_order_acq_rel);
  EXPECT_LT(before2, 200) << "launch should have been gated";
  EXPECT_EQ(page_->credits_us.load(std::memory_order_relaxed), -130)
      << "an over-debit must be visible so the refill can repay it";
}

// Two processes share one page: the child debits, the parent observes.
TEST_F(TenantPageTest, SharedAcrossProcesses) {
  page_->credits_us.store(1000, std::memory_order_relaxed);

  const pid_t pid = fork();
  ASSERT_NE(pid, -1) << std::strerror(errno);
  if (pid == 0) {
    for (int i = 0; i < 10; ++i) {
      page_->credits_us.fetch_sub(10, std::memory_order_acq_rel);
      page_->launched.fetch_add(1, std::memory_order_relaxed);
    }
    _exit(0);
  }

  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid) << std::strerror(errno);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);

  EXPECT_EQ(page_->credits_us.load(std::memory_order_acquire), 900);
  EXPECT_EQ(page_->launched.load(std::memory_order_acquire), 10u);
}

// The gated path blocks on futex_word and the daemon's refill wakes it. The
// test drives the daemon side from a child process, as production does.
TEST_F(TenantPageTest, FutexWordWakesAWaiter) {
  page_->credits_us.store(0, std::memory_order_relaxed);
  const std::uint32_t observed = page_->futex_word.load(std::memory_order_acquire);

  const pid_t pid = fork();
  ASSERT_NE(pid, -1) << std::strerror(errno);
  if (pid == 0) {
    // Stand in for the daemon's refill: add credits, bump the word, wake.
    usleep(20'000);
    page_->credits_us.store(500, std::memory_order_release);
    page_->futex_word.fetch_add(1, std::memory_order_acq_rel);
    page_->daemon_epoch.fetch_add(1, std::memory_order_release);
    FutexWake(&page_->futex_word, INT32_MAX);
    _exit(0);
  }

  // The shim's wait loop: bounded waits, re-checking credits each time.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  bool refilled = false;
  while (std::chrono::steady_clock::now() < deadline) {
    if (page_->credits_us.load(std::memory_order_acquire) > 0) {
      refilled = true;
      break;
    }
    FutexWait(&page_->futex_word, observed, std::chrono::milliseconds(1));
  }

  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid) << std::strerror(errno);
  EXPECT_TRUE(refilled) << "waiter was never woken by the refill";
  EXPECT_EQ(page_->credits_us.load(std::memory_order_acquire), 500);
  EXPECT_NE(page_->futex_word.load(std::memory_order_acquire), observed);
}

// I-5: staleness is judged from the daemon's own clock stamp, so a shim that
// maps a page mid-run can decide to fail open without watching two ticks.
TEST_F(TenantPageTest, StalenessIsJudgedFromTheEpochStamp) {
  struct timespec now {};
  ASSERT_EQ(clock_gettime(CLOCK_MONOTONIC, &now), 0);
  const std::int64_t now_ns = now.tv_sec * 1'000'000'000LL + now.tv_nsec;

  page_->epoch_mono_ns.store(now_ns, std::memory_order_release);
  EXPECT_LT(now_ns - page_->epoch_mono_ns.load(std::memory_order_acquire),
            100'000'000LL)
      << "a just-written stamp must not look stale";

  page_->epoch_mono_ns.store(now_ns - 150'000'000LL, std::memory_order_release);
  EXPECT_GT(now_ns - page_->epoch_mono_ns.load(std::memory_order_acquire),
            100'000'000LL)
      << "a 150 ms old stamp must trip the 100 ms fail-open threshold";
}

}  // namespace
}  // namespace tessera
