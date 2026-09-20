// T7 of docs/gates/M0.md: I-4, the hot path.
//
// "When credits are available, the per-launch path in the shim performs no
// syscalls, no locks, no heap allocation, and no driver calls beyond the
// forwarded launch." v0 has no credits yet, but it already owns the path that
// M2 will add gating to, so the discipline is established now, while the path
// is three instructions long and the measurement is unambiguous.
//
// Two instruments, each with its own positive control, because a measurement
// that can only ever report zero is not a measurement:
//
//   allocations  the application interposes malloc/calloc/realloc/free and
//                operator new, counts them across the window, and then makes
//                one deliberate allocation in a control region. A run that
//                reports zero for the window has just demonstrated, in the
//                same process, that it can report one.
//
//   syscalls     the application brackets the window with write(-1, ...),
//                which strace records as EBADF and which costs the shim
//                nothing. Counting the lines strace emits strictly between
//                the markers gives the syscalls the window performed. The
//                control region contains a deliberate one.
//
// What sits underneath the window is tests/support/null_driver.c, not
// libcuda_fake: the fake takes a lock, queues work and wakes a worker, so
// measuring through it would measure the simulation. With the null driver the
// only code inside the window is the shim's wrapper. Running the same
// application against the fake is the second positive control: the instrument
// must see the fake's allocations and syscalls.
#include <unistd.h>

#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "acceptance_fixture.h"

namespace tessera_test {
namespace {

constexpr int kLaunches = 2000;

// Under a sanitizer neither instrument measures the shim. AddressSanitizer
// replaces the allocator, so the application's malloc interposition never sees
// the calls it is counting, and the sanitizer runtime issues syscalls of its
// own inside the window (shadow-memory bookkeeping, error-report plumbing).
// The I-4 claim is therefore measured in the unsanitized configurations, where
// the window contains only the shim's wrapper. The sanitized builds still run
// every other acceptance test, including the whole interception matrix.
bool SanitizersAreActive() {
#if defined(TESSERA_SANITIZED)
  return true;
#else
  return false;
#endif
}

RunOutcome RunHotPath(Strategy strategy, const std::string& driver,
                      std::vector<std::string> prefix = {}) {
  RunRequest request;
  request.strategy = strategy;
  request.args = {"hotpath", std::to_string(kLaunches)};
  request.real_libcuda = driver;
  request.prefix = std::move(prefix);
  request.timeout_ms = 120000;
  return run_acceptance_app(request);
}

// run_child spawns with posix_spawn, which does not search PATH, so the
// tracer has to be named by absolute path.
std::string StracePath() {
  // strace is native-only tooling; under qemu the child is the emulator.
  if (std::string(TESSERA_TEST_EMULATOR) != "") {
    return {};
  }
  for (const char* const candidate : {"/usr/bin/strace", "/bin/strace", "/usr/sbin/strace"}) {
    if (::access(candidate, X_OK) == 0) {
      return candidate;
    }
  }
  return {};
}

// Counts the syscall lines strace emitted strictly between two markers. The
// markers are themselves write() calls and are not counted.
struct WindowSyscalls {
  bool found_begin = false;
  bool found_end = false;
  int count = 0;
};

WindowSyscalls CountBetween(const std::string& trace, const std::string& begin_marker,
                            const std::string& end_marker) {
  WindowSyscalls result;
  size_t pos = 0;
  bool inside = false;
  while (pos < trace.size()) {
    size_t eol = trace.find('\n', pos);
    if (eol == std::string::npos) {
      eol = trace.size();
    }
    const std::string line = trace.substr(pos, eol - pos);
    pos = eol + 1;

    if (line.find(begin_marker) != std::string::npos) {
      inside = true;
      result.found_begin = true;
      continue;
    }
    if (line.find(end_marker) != std::string::npos) {
      if (inside) {
        result.found_end = true;
      }
      inside = false;
      continue;
    }
    if (!inside || line.empty()) {
      continue;
    }
    // Skip strace's own bookkeeping and signal lines; count only syscalls,
    // which strace prints as "name(args) = result".
    if (line.rfind("---", 0) == 0 || line.rfind("+++", 0) == 0) {
      continue;
    }
    if (line.find('(') == std::string::npos) {
      continue;
    }
    ++result.count;
  }
  return result;
}

void ExpectHotPathRan(const RunOutcome& run) {
  ASSERT_TRUE(run.spawn.spawned) << run.describe();
  ASSERT_TRUE(run.app_completed()) << run.describe();
  ASSERT_EQ(run.app_number("launch_failures").value_or(1), 0u) << run.describe();
  ASSERT_EQ(run.app_number("launches").value_or(0), static_cast<uint64_t>(kLaunches))
      << run.describe();

  // The launches really did go through the shim: without this, a window with
  // no allocations and no syscalls could simply be a window with no shim.
  const uint64_t before = run.app_number("shim_launches_before_window").value_or(0);
  const uint64_t after = run.app_number("shim_launches_after_window").value_or(0);
  ASSERT_EQ(run.app_number("shim_present").value_or(0), 1u) << run.describe();
  EXPECT_EQ(after - before, static_cast<uint64_t>(kLaunches))
      << "the shim did not count the launches in the measured window\n"
      << run.describe();
}

TEST(T7HotPath, NoHeapAllocationPerLaunch) {
  if (SanitizersAreActive()) {
    GTEST_SKIP() << "the allocation instrument measures the sanitizer, not the shim";
  }
  const RunOutcome run = RunHotPath(Strategy::kMasquerade, TESSERA_NULL_DRIVER_LIB);
  ASSERT_NO_FATAL_FAILURE(ExpectHotPathRan(run));

  ASSERT_EQ(run.app_number("alloc_counting_available").value_or(0), 1u)
      << "the allocation hooks are not in effect, so zero would mean nothing\n"
      << run.describe();

  EXPECT_EQ(run.app_number("window_alloc_events").value_or(~0ull), 0ull)
      << kLaunches << " launches through the shim touched the heap (I-4)\n"
      << run.describe();

  // The instrument's positive control, in the same process and the same run.
  EXPECT_GT(run.app_number("control_allocations").value_or(0), 0u)
      << "the allocation counter reported zero for a region that deliberately "
         "allocates, so its zero for the window is not evidence\n"
      << run.describe();
}

TEST(T7HotPath, NoSyscallPerLaunch) {
  if (SanitizersAreActive()) {
    GTEST_SKIP() << "the sanitizer runtime issues syscalls inside the measured window";
  }
  const std::string strace = StracePath();
  if (strace.empty()) {
    GTEST_SKIP() << "strace is unavailable (or this is a cross build run under qemu); "
                    "the syscall half of T7 is native-x86-64 only";
  }

  const RunOutcome run =
      RunHotPath(Strategy::kMasquerade, TESSERA_NULL_DRIVER_LIB,
                 {strace, "-f", "-qq", "-e", "signal=none", "-o", "/dev/stderr"});
  ASSERT_NO_FATAL_FAILURE(ExpectHotPathRan(run));

  const WindowSyscalls window =
      CountBetween(run.spawn.output, "tessera-window-begin", "tessera-window-end");
  ASSERT_TRUE(window.found_begin && window.found_end)
      << "could not find the window markers in strace output; the measurement did not happen\n"
      << run.describe();

  EXPECT_EQ(window.count, 0) << kLaunches << " launches through the shim issued syscalls (I-4)\n"
                             << run.describe();

  // Positive control: the instrument must see the deliberate syscall in the
  // control region. Without this, "0 syscalls" could just mean "strace output
  // was not parsed".
  const WindowSyscalls control =
      CountBetween(run.spawn.output, "tessera-control-begin", "tessera-control-end");
  ASSERT_TRUE(control.found_begin && control.found_end) << run.describe();
  EXPECT_GT(control.count, 0)
      << "the strace parser found no syscalls in a region that makes one, so its zero "
         "for the window is not evidence\n"
      << run.describe();
}

// The second positive control, and the reason the null driver exists: run the
// identical application against libcuda_fake, whose launch takes a lock,
// queues work and wakes a worker thread. The instrument must light up. If this
// ever reports zero, the measurement above is blind and T7 means nothing.
TEST(T7HotPathControl, TheInstrumentSeesASimulatedDriversWork) {
  if (SanitizersAreActive()) {
    GTEST_SKIP() << "the allocation instrument measures the sanitizer, not the shim";
  }
  const RunOutcome run = RunHotPath(Strategy::kMasquerade, TESSERA_FAKE_MASQ_LIB);
  ASSERT_NO_FATAL_FAILURE(ExpectHotPathRan(run));
  ASSERT_EQ(run.app_number("alloc_counting_available").value_or(0), 1u) << run.describe();

  EXPECT_GT(run.app_number("window_alloc_events").value_or(0), 0u)
      << "a driver that queues work and wakes a thread appeared to allocate nothing, so "
         "the allocation instrument is blind\n"
      << run.describe();
}

}  // namespace
}  // namespace tessera_test
