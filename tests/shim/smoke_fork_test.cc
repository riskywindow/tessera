// The shim across fork(). A separate binary because ThreadSanitizer kills a
// child that starts threads after a multi-threaded fork unless die_after_fork
// is off -- and recreating its worker thread in the child is exactly what the
// fake driver does on purpose (fake_driver/README.md).

#include "smoke_fixture.h"

namespace tessera_test {
namespace {

// The shim must survive fork(): the child's counters start at zero, the
// parent's are untouched, and a hooked call in the child completes rather than
// deadlocking on a lock the parent held.
TEST_F(ShimSmokeTest, ForkChildKeepsWorkingWithItsOwnCounters) {
  ASSERT_CU(launch_(function_, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  ASSERT_EQ(1u, stats().launches);

  const pid_t pid = fork();
  ASSERT_NE(-1, pid);
  if (pid == 0) {
    // The child: counters reset, and the shim still forwards.
    int status = 0;
    if (stats().launches != 0) {
      status = 2;
    }
    if (launch_(function_, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr) != CUDA_SUCCESS) {
      status = 3;
    }
    if (status == 0 && stats().launches != 1) {
      status = 4;
    }
    // _exit, so the child runs neither gtest's atexit handlers nor the
    // libraries' exit dumps.
    _exit(status);
  }
  int status = 0;
  ASSERT_EQ(pid, waitpid(pid, &status, 0));
  ASSERT_TRUE(WIFEXITED(status)) << "the child did not exit normally";
  EXPECT_EQ(0, WEXITSTATUS(status));
  EXPECT_EQ(1u, stats().launches) << "the child's calls leaked into the parent's counters";
}

}  // namespace
}  // namespace tessera_test
