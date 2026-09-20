// T2 of docs/gates/M0.md (gate G6): per-thread default stream, and symbol
// version selection.
//
// cuGetProcAddress does not return "the function called X". It returns the
// variant of X selected by the cudaVersion the caller was built against and by
// the CU_GET_PROC_ADDRESS_* stream flag. Those variants are distinct exported
// symbols with distinct behaviour: cuLaunchKernel_ptsz resolves a NULL stream
// argument to the calling thread's per-thread default stream, cuLaunchKernel
// to the context's legacy one; cuMemAlloc takes a 32-bit size and cuMemAlloc_v2
// a size_t. A shim that returns the wrong one either miscounts or corrupts the
// ABI, and the application in the middle of this test is the one that would
// crash.
//
// The application drives every case and reports failures; the fake driver
// records which exported entry point each call actually reached, so "the right
// variant" is measured at the far end rather than assumed from the pointer.
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "acceptance_fixture.h"

namespace tessera_test {
namespace {

const std::set<std::string>& hooked_symbols() {
  static const std::set<std::string> names = [] {
    std::set<std::string> set;
#define TESSERA_SYMBOL(index, name, hooked) \
  if ((hooked) != 0) {                      \
    set.insert(name);                       \
  }
#include "tessera_symbols.inc"
#undef TESSERA_SYMBOL
    return set;
  }();
  return names;
}

RunOutcome RunPtsz(Strategy strategy) {
  RunRequest request;
  request.strategy = strategy;
  request.args = {"ptsz"};
  return run_acceptance_app(request);
}

void ExpectVariantsWereExercised(const RunOutcome& run) {
  // Per-thread versus legacy: both must have been reached, or the test is
  // asserting about a distinction the workload never made.
  EXPECT_GT(count_of(run.issued, "cuLaunchKernel_ptsz"), 0u)
      << "the per-thread entry point was never reached\n"
      << run.describe();
  EXPECT_GT(count_of(run.issued, "cuLaunchKernel"), 0u)
      << "the legacy entry point was never reached\n"
      << run.describe();

  // Version selection: the v1 and v2 forms of the same base name, and three
  // versions of cuCtxCreate.
  EXPECT_GT(count_of(run.issued, "cuMemAlloc"), 0u) << run.describe();
  EXPECT_GT(count_of(run.issued, "cuMemAlloc_v2"), 0u) << run.describe();
  for (const char* const name : {"cuCtxCreate_v2", "cuCtxCreate_v3", "cuCtxCreate_v4"}) {
    EXPECT_GT(count_of(run.issued, name), 0u)
        << name << " was not exercised, so version selection is untested\n"
        << run.describe();
  }
}

class PerThreadDefaultStream : public ::testing::TestWithParam<Strategy> {};

TEST_P(PerThreadDefaultStream, EveryVariantResolvesToItsOwnEntryPoint) {
  const RunOutcome run = RunPtsz(GetParam());
  ASSERT_TRUE(run.spawn.spawned) << run.describe();
  ASSERT_TRUE(run.app_completed()) << run.describe();

  // The application compares, for each case, the entry point the fake says it
  // reached against the one the flag or version asked for.
  EXPECT_EQ(run.app_number("stream_failures").value_or(1), 0u)
      << "a stream-flag lookup reached the wrong entry point\n"
      << run.describe();
  EXPECT_EQ(run.app_number("version_failures").value_or(1), 0u)
      << "a versioned lookup reached the wrong entry point\n"
      << run.describe();
  ASSERT_NO_FATAL_FAILURE(ExpectVariantsWereExercised(run));

  ASSERT_TRUE(run.shim_file_written) << run.describe();

  // Each variant is counted under its own name: a shim that collapsed
  // cuLaunchKernel_ptsz into cuLaunchKernel would still add up to the right
  // total, and would be wrong.
  for (const auto& [name, issued] : run.issued) {
    EXPECT_EQ(count_of(run.fake_symbols, name), issued)
        << "driver received a different number of " << name << " calls\n"
        << run.describe();
    if (hooked_symbols().count(name) != 0) {
      EXPECT_EQ(count_of(run.shim_symbols, name), issued)
          << "shim did not count hooked variant " << name << " separately\n"
          << run.describe();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(T2, PerThreadDefaultStream,
                         ::testing::Values(Strategy::kMasquerade, Strategy::kPreload),
                         [](const ::testing::TestParamInfo<Strategy>& entry) {
                           return entry.param == Strategy::kMasquerade ? "masquerade" : "preload";
                         });

// Negative control. Without a shim the application resolves the same variants
// straight from the driver, so the per-variant counts below are what correct
// resolution looks like. If a shim arm ever matches the control on totals but
// not per variant, the difference is the shim's doing.
TEST(T2NegativeControl, TheSameVariantsAreReachedWithNoShimAtAll) {
  const RunOutcome control = RunPtsz(Strategy::kNoShim);
  ASSERT_TRUE(control.app_completed()) << control.describe();
  EXPECT_EQ(control.app_number("stream_failures").value_or(1), 0u) << control.describe();
  EXPECT_EQ(control.app_number("version_failures").value_or(1), 0u) << control.describe();
  EXPECT_FALSE(control.shim_file_written) << control.describe();
  ASSERT_NO_FATAL_FAILURE(ExpectVariantsWereExercised(control));

  // And the shim arms reach exactly the same entry points, variant by variant.
  for (const Strategy strategy : {Strategy::kMasquerade, Strategy::kPreload}) {
    const RunOutcome run = RunPtsz(strategy);
    ASSERT_TRUE(run.app_completed()) << run.describe();
    EXPECT_EQ(run.fake_symbols, control.fake_symbols)
        << "under " << strategy_name(strategy)
        << " the driver was reached through a different set of entry points than with no shim\n"
        << run.describe();
  }
}

}  // namespace
}  // namespace tessera_test
