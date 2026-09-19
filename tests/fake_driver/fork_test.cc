// Fork safety.
//
// The fake has a worker thread, and a worker thread does not survive fork. If
// the child's first cuStreamSynchronize waited for work that no thread is left
// to complete, every shim fork test would hang instead of failing, and I-5
// ("fail open, never deadlock") would be untestable. These tests fork with the
// simulation busy and require the child to finish a driver call and exit.
//
// This is a separate binary from the rest of the suite because it runs with
// TSAN_OPTIONS=die_after_fork=0: ThreadSanitizer's default is to kill a child
// that creates threads after a multi-threaded fork, which is exactly what the
// fake does on purpose.

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include <sys/wait.h>

#include "fake_fixture.h"

namespace tessera_test {
namespace {

constexpr int kChildTimeoutSeconds = 10;

std::string temp_path(const char* name) {
  const char* dir = std::getenv("TMPDIR");
  if (dir == nullptr || dir[0] == '\0') {
    dir = "/tmp";
  }
  std::string p(dir);
  p += "/tessera_fake_";
  p += name;
  p += "_";
  p += std::to_string(getpid());
  p += ".json";
  return p;
}

// Waits for `pid`, failing rather than hanging if the child never exits.
// Returns the exit status, or -1 on timeout.
int wait_for_child(pid_t pid) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(kChildTimeoutSeconds);
  while (std::chrono::steady_clock::now() < deadline) {
    int status = 0;
    const pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) {
      if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
      }
      return -2;
    }
    if (r < 0) {
      return -3;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  kill(pid, SIGKILL);
  int status = 0;
  waitpid(pid, &status, 0);
  return -1;
}

class ForkTest : public FakeDriverTest {
 protected:
  static CUresult launch(CUfunction f, CUstream s) {
    return cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr);
  }
};

TEST_F(ForkTest, AChildCompletesADriverCallWithQueuedWorkInherited) {
  CUfunction k = tf_register_kernel("k", nullptr, 1000);
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  // Queue work the child inherits but no thread in the child is running yet.
  ASSERT_CU(launch(k, s));
  ASSERT_CU(launch(k, s));

  const pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    alarm(kChildTimeoutSeconds);
    const CUresult r = cuStreamSynchronize(s);
    _exit(r == CUDA_SUCCESS ? 0 : 20);
  }
  EXPECT_EQ(0, wait_for_child(pid));

  // The parent's own queue is untouched and still works.
  ASSERT_CU(cuStreamSynchronize(s));
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(ForkTest, AChildCompletesADriverCallWhileAnotherThreadIsBusy) {
  CUfunction k = tf_register_kernel("k", nullptr, 1);
  CUstream busy_stream = nullptr;
  CUstream child_stream = nullptr;
  ASSERT_CU(cuStreamCreate(&busy_stream, 0));
  ASSERT_CU(cuStreamCreate(&child_stream, 0));
  ASSERT_CU(launch(k, child_stream));

  // A second thread hammers the simulation lock across the fork. The fake's
  // pthread_atfork handlers must hold that lock over fork() so the child never
  // inherits it locked by a thread that no longer exists.
  std::atomic<bool> stop{false};
  std::thread hammer([&] {
    ASSERT_CU(cuCtxSetCurrent(ctx_));
    while (!stop.load(std::memory_order_relaxed)) {
      if (launch(k, busy_stream) != CUDA_SUCCESS) {
        break;
      }
      if (cuStreamSynchronize(busy_stream) != CUDA_SUCCESS) {
        break;
      }
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  const pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    alarm(kChildTimeoutSeconds);
    const CUresult r = cuStreamSynchronize(child_stream);
    _exit(r == CUDA_SUCCESS ? 0 : 20);
  }
  const int status = wait_for_child(pid);

  stop.store(true, std::memory_order_relaxed);
  hammer.join();
  EXPECT_EQ(0, status);

  ASSERT_CU(cuStreamSynchronize(child_stream));
  ASSERT_CU(cuStreamDestroy(busy_stream));
  ASSERT_CU(cuStreamDestroy(child_stream));
}

TEST_F(ForkTest, ChildCountersStartAtZeroAndParentCountersAreUnchanged) {
  CUfunction k = tf_register_kernel("k", nullptr, 1);
  for (int i = 0; i < 5; ++i) {
    ASSERT_CU(launch(k, nullptr));
  }
  ASSERT_CU(cuCtxSynchronize());
  ASSERT_EQ(5u, tf_symbol_count("cuLaunchKernel"));

  const pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    alarm(kChildTimeoutSeconds);
    int code = 0;
    if (tf_symbol_count("cuLaunchKernel") != 0) {
      code = 21;
    }
    if (code == 0 && launch(k, nullptr) != CUDA_SUCCESS) {
      code = 22;
    }
    if (code == 0 && cuCtxSynchronize() != CUDA_SUCCESS) {
      code = 23;
    }
    if (code == 0 && tf_symbol_count("cuLaunchKernel") != 1) {
      code = 24;
    }
    _exit(code);
  }
  EXPECT_EQ(0, wait_for_child(pid));
  EXPECT_EQ(5u, tf_symbol_count("cuLaunchKernel"));
}

TEST_F(ForkTest, EmulatingRealCudaMakesEveryCallInTheChildFail) {
  CUfunction k = tf_register_kernel("k", nullptr, 1);
  tf_set_fork_mode(1);

  const pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    alarm(kChildTimeoutSeconds);
    int code = 0;
    if (launch(k, nullptr) != CUDA_ERROR_NOT_INITIALIZED) {
      code = 31;
    }
    int count = 0;
    if (code == 0 && cuDeviceGetCount(&count) != CUDA_ERROR_NOT_INITIALIZED) {
      code = 32;
    }
    if (code == 0 && cuInit(0) != CUDA_ERROR_NOT_INITIALIZED) {
      code = 33;
    }
    // Calls are still counted, so a test can see what the child attempted.
    if (code == 0 && tf_symbol_count("cuLaunchKernel") != 1) {
      code = 34;
    }
    _exit(code);
  }
  EXPECT_EQ(0, wait_for_child(pid));

  // The parent is unaffected: fork emulation only applies to forked children.
  EXPECT_CU(launch(k, nullptr));
  ASSERT_CU(cuCtxSynchronize());
}

TEST_F(ForkTest, AChildDumpsOnDemandButNotAtExit) {
  // The exit dump belongs to the process that initialised the library: a
  // forked child must not clobber it. Dumping on demand still works.
  const std::string on_demand = temp_path("fork_ondemand");
  const std::string at_exit = temp_path("fork_atexit");
  std::remove(on_demand.c_str());
  std::remove(at_exit.c_str());

  const pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    alarm(kChildTimeoutSeconds);
    int code = 0;
    if (setenv("TESSERA_FAKE_STATS_FILE", at_exit.c_str(), 1) != 0) {
      code = 40;
    }
    int count = 0;
    if (code == 0 && cuDeviceGetCount(&count) != CUDA_SUCCESS) {
      code = 41;
    }
    if (code == 0 && tf_dump_stats(on_demand.c_str()) != 0) {
      code = 42;
    }
    std::exit(code);  // runs the library's exit handler
  }
  EXPECT_EQ(0, wait_for_child(pid));

  EXPECT_EQ(0, access(on_demand.c_str(), F_OK)) << "the child's explicit dump is missing";
  EXPECT_NE(0, access(at_exit.c_str(), F_OK)) << "a forked child wrote the exit stats file";
  std::remove(on_demand.c_str());
  std::remove(at_exit.c_str());
}

}  // namespace
}  // namespace tessera_test
