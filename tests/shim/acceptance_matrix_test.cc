// T1, T4, T5 and T6 of docs/gates/M0.md: the interception matrix, forwarding,
// lazy initialisation, and the self-reference guard.
//
// T1 is the criterion the rest of the project rests on (gate G5). For every
// combination of interposition strategy and access path, three independently
// produced counts must agree symbol by symbol: what the application issued,
// what the shim counted, and what the driver underneath received. A shim that
// misses a path shows up as a shortfall in the middle column; a shim that
// changes what reaches the driver shows up in the right one.
#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "acceptance_fixture.h"

namespace tessera_test {
namespace {

// Which symbols the shim hooks, taken from the shim's own build rather than
// from a list maintained here, so the two cannot drift apart.
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

bool is_hooked(const std::string& name) {
  return hooked_symbols().count(name) != 0;
}

// Symbols every workload run must have issued. If the application stopped
// exercising these, an equality assertion over "whatever it happened to call"
// could pass while proving nothing.
const std::vector<std::string>& required_symbols() {
  static const std::vector<std::string> names = {
      "cuInit",       "cuCtxCreate_v2", "cuMemAlloc_v2",   "cuLaunchKernel",
      "cuMemFree_v2", "cuStreamCreate", "cuMemGetInfo_v2",
  };
  return names;
}

void ExpectWorkloadRan(const RunOutcome& run) {
  ASSERT_TRUE(run.spawn.spawned) << run.describe();
  EXPECT_FALSE(run.spawn.timed_out) << run.describe();
  ASSERT_TRUE(run.app_completed()) << run.describe();
  EXPECT_EQ(run.app_number("workload_ok").value_or(0), 1u) << run.describe();
  // Only the cuGetProcAddress paths resolve symbols by name, so only they
  // report lookup failures.
  if (const std::optional<uint64_t> failures = run.app_number("lookup_failures")) {
    EXPECT_EQ(*failures, 0u) << run.describe();
  }
  // The application's recorder is a fixed-size table; if it overflowed, its
  // counts are truncated and every comparison below is meaningless.
  EXPECT_EQ(run.app_number("recorder_overflowed").value_or(1), 0u) << run.describe();

  // Vacuity guards: the comparison must be over real, plural work.
  EXPECT_GE(run.issued.size(), 15u) << "too few distinct symbols to be a real workload";
  for (const std::string& name : required_symbols()) {
    EXPECT_GT(count_of(run.issued, name), 0u)
        << name << " was not issued; the workload no longer covers it";
  }
}

// ---------------------------------------------------------------------------
// T1: interception matrix, 2 strategies x 4 access paths
// ---------------------------------------------------------------------------

class InterceptionMatrix : public ::testing::TestWithParam<std::tuple<Strategy, std::string>> {};

TEST_P(InterceptionMatrix, ShimSeesEveryCallAndForwardsItUnchanged) {
  const Strategy strategy = std::get<0>(GetParam());
  const std::string path = std::get<1>(GetParam());

  const RunOutcome run = run_workload(strategy, path);
  ASSERT_NO_FATAL_FAILURE(ExpectWorkloadRan(run));

  ASSERT_TRUE(run.shim_file_written)
      << "the shim wrote no counter file, so it was never in the process\n"
      << run.describe();

  // The whole of T1, and the part that must hold in every cell: every call to
  // a symbol the shim hooks is seen by the shim, and arrives at the driver.
  // These are the symbols gating and quotas will act on in M2.
  size_t hooked_checked = 0;
  for (const auto& [name, issued] : run.issued) {
    EXPECT_EQ(count_of(run.fake_symbols, name), issued)
        << "driver received a different number of " << name << " calls\n"
        << run.describe();
    if (!is_hooked(name)) {
      continue;
    }
    ++hooked_checked;
    EXPECT_EQ(count_of(run.shim_symbols, name), issued)
        << "shim missed calls to hooked symbol " << name << " via " << path << " under "
        << strategy_name(strategy) << "\n"
        << run.describe();
  }
  EXPECT_GE(hooked_checked, 8u) << "too few hooked symbols exercised for this to mean anything";

  // Masquerade is strictly stronger and the matrix says so: because the shim
  // IS libcuda.so.1, even the symbols it does not hook pass through it, so it
  // can account for everything the application asked the driver to do. Under
  // LD_PRELOAD the dlsym hook substitutes only symbols it has a typed wrapper
  // for, so an unhooked symbol fetched by handle reaches the driver directly
  // and the shim never sees it. That difference is a measured property of the
  // two strategies (ADR-001), not a defect, and H1 is where it decides the
  // default.
  if (strategy == Strategy::kMasquerade) {
    for (const auto& [name, issued] : run.issued) {
      EXPECT_EQ(count_of(run.shim_symbols, name), issued)
          << "masquerade should see every call, including unhooked " << name << "\n"
          << run.describe();
    }
  }

  // And nothing the application did not do: a shim that injects calls of its
  // own would show up here rather than in the loop above.
  for (const auto& [name, seen] : run.shim_symbols) {
    EXPECT_EQ(count_of(run.issued, name), seen)
        << "shim counted " << name << " which the application never issued\n"
        << run.describe();
  }

  EXPECT_GT(run.shim_number("hooked_calls").value_or(0), 0u) << run.describe();
  EXPECT_EQ(count_of(run.shim_symbols, "cuLaunchKernel"), count_of(run.issued, "cuLaunchKernel"));
  // v0 counts graph launches separately; none of these workloads launch graphs.
  EXPECT_EQ(run.shim_number("graph_launches").value_or(1), 0u) << run.describe();
}

INSTANTIATE_TEST_SUITE_P(T1, InterceptionMatrix,
                         ::testing::Combine(::testing::Values(Strategy::kMasquerade,
                                                              Strategy::kPreload),
                                            ::testing::ValuesIn(access_paths())),
                         [](const ::testing::TestParamInfo<InterceptionMatrix::ParamType>& entry) {
                           const std::string strategy =
                               std::get<0>(entry.param) == Strategy::kMasquerade ? "masquerade"
                                                                                 : "preload";
                           return strategy + "_" + std::get<1>(entry.param);
                         });

// ---------------------------------------------------------------------------
// T1 negative control
// ---------------------------------------------------------------------------

// Without the shim the same application does the same work and the driver
// still records it -- so the equality above is not something the workload
// satisfies on its own. What disappears is the middle column.
class NegativeControl : public ::testing::TestWithParam<std::string> {};

TEST_P(NegativeControl, WithoutTheShimThereIsNothingToMatchAgainst) {
  const std::string path = GetParam();
  const RunOutcome control = run_workload(Strategy::kNoShim, path);
  ASSERT_NO_FATAL_FAILURE(ExpectWorkloadRan(control));

  EXPECT_FALSE(control.shim_file_written)
      << "a shim counter file appeared in a run with no shim: the arms are not distinct\n"
      << control.describe();
  EXPECT_TRUE(control.shim_symbols.empty());

  // The driver still saw the work, so the workload itself is not the reason
  // the shim arm passes.
  for (const std::string& name : required_symbols()) {
    EXPECT_EQ(count_of(control.fake_symbols, name), count_of(control.issued, name))
        << control.describe();
  }
  EXPECT_GT(count_of(control.issued, "cuLaunchKernel"), 0u);
}

INSTANTIATE_TEST_SUITE_P(T1, NegativeControl, ::testing::ValuesIn(access_paths()),
                         [](const ::testing::TestParamInfo<std::string>& entry) {
                           return entry.param;
                         });

// The same workload must reach the driver identically whether or not the shim
// is in the way. This is I-3 on the CPU: the shim may change timing, never
// what the driver is asked to do.
TEST(T4Forwarding, TheDriverSeesTheSameCallsWithAndWithoutTheShim) {
  for (const std::string& path : access_paths()) {
    const RunOutcome control = run_workload(Strategy::kNoShim, path);
    const RunOutcome masq = run_workload(Strategy::kMasquerade, path);
    const RunOutcome preload = run_workload(Strategy::kPreload, path);
    ASSERT_NO_FATAL_FAILURE(ExpectWorkloadRan(control));
    ASSERT_NO_FATAL_FAILURE(ExpectWorkloadRan(masq));
    ASSERT_NO_FATAL_FAILURE(ExpectWorkloadRan(preload));

    EXPECT_EQ(masq.fake_symbols, control.fake_symbols)
        << "masquerade changed what the driver was asked to do, via " << path;
    EXPECT_EQ(preload.fake_symbols, control.fake_symbols)
        << "preload changed what the driver was asked to do, via " << path;
  }
}

// Unhooked symbols are forwarded by generated trampolines rather than typed
// wrappers. The workload calls plenty of them (cuDeviceGetName, cuModuleLoadData,
// cuEventCreate and so on), and they must arrive at the driver exactly once each.
TEST(T4Forwarding, UnhookedSymbolsReachTheDriverThroughTrampolines) {
  const RunOutcome run = run_workload(Strategy::kMasquerade, "direct");
  ASSERT_NO_FATAL_FAILURE(ExpectWorkloadRan(run));
  ASSERT_GT(run.shim_number("forwarded_unhooked").value_or(0), 0u)
      << "no unhooked symbol was exercised, so this test proves nothing\n"
      << run.describe();

  for (const char* const name : {"cuDeviceGetName", "cuModuleLoadData", "cuEventCreate"}) {
    const uint64_t issued = count_of(run.issued, name);
    ASSERT_GT(issued, 0u) << name << " is no longer exercised by the workload";
    EXPECT_EQ(count_of(run.fake_symbols, name), issued) << name << " did not reach the driver";
    EXPECT_EQ(count_of(run.shim_symbols, name), issued) << name << " was not counted";
  }
}

// Error pass-through for hooked wrappers is covered where the error can be
// injected: the fake driver's own suite (tf_set_result) proves each entry
// point returns what it was told to, and T6 below proves a CUDA error raised
// underneath the shim arrives at the application unchanged through three
// different call shapes. The workload itself has no failing call to compare,
// so there is nothing further to assert here without inventing one.

// ---------------------------------------------------------------------------
// T5: lazy initialisation
// ---------------------------------------------------------------------------

// A library that masquerades as libcuda.so.1 is constructed at a moment it
// does not choose. Loading it must therefore touch the driver exactly zero
// times; the first driver call is the application's.
TEST(T5LazyInit, LoadingTheShimCallsNothing) {
  for (const Strategy strategy : {Strategy::kMasquerade, Strategy::kPreload}) {
    RunRequest request;
    request.strategy = strategy;
    request.args = {"lazyinit"};  // inspect state BEFORE making any driver call
    const RunOutcome run = run_acceptance_app(request);
    ASSERT_TRUE(run.app_completed()) << run.describe();

    EXPECT_EQ(run.app_number("shim_calls_before").value_or(1), 0u)
        << "the shim counted calls before the application made one, under "
        << strategy_name(strategy) << "\n"
        << run.describe();
    // Only masquerade can answer "did the shim load the driver too early?".
    // Under LD_PRELOAD the application's own DT_NEEDED libcuda.so.1 is mapped
    // by the loader before main() runs, so the driver is present no matter
    // what the shim does and the question is not the shim's to answer.
    if (strategy == Strategy::kMasquerade) {
      EXPECT_EQ(run.app_number("driver_mapped_before_first_call").value_or(1), 0u)
          << "the shim loaded the real driver before the application asked for anything\n"
          << run.describe();
      EXPECT_EQ(run.app_number("driver_mapped_after_first_call").value_or(0), 1u)
          << "the driver was never loaded at all, so the check above is vacuous\n"
          << run.describe();
    }

    // Whether the driver's own counters are reachable is itself a property
    // worth pinning down rather than working around. The shim loads the real
    // driver RTLD_LOCAL (ADR-001), so under masquerade the driver's symbols
    // must NOT appear in the global scope; under LD_PRELOAD the application
    // linked the driver itself, so they must.
    const bool fake_visible = run.app_number("fake_counts_available").value_or(0) == 1u;
    if (strategy == Strategy::kMasquerade) {
      EXPECT_FALSE(fake_visible)
          << "the real driver's symbols reached the global scope: the shim did not load it "
             "RTLD_LOCAL\n"
          << run.describe();
    } else {
      EXPECT_TRUE(fake_visible)
          << "the application's own driver should be visible under LD_PRELOAD\n"
          << run.describe();
    }

    if (fake_visible) {
      EXPECT_EQ(run.app_number("fake_calls_before_first_call").value_or(1), 0u)
          << "the driver was called before the application's first call, under "
          << strategy_name(strategy) << "\n"
          << run.describe();
      EXPECT_GT(run.app_number("fake_calls_after_first_call").value_or(0), 0u) << run.describe();
    }

    // ...and the run did go on to do real work, so the zeroes above are not
    // just an application that never started.
    EXPECT_GT(run.app_number("shim_calls_after").value_or(0), 0u) << run.describe();
  }
}

// ---------------------------------------------------------------------------
// T6: the self-reference guard
// ---------------------------------------------------------------------------

// TESSERA_REAL_LIBCUDA pointing at the shim itself is a misconfiguration that
// must fail fast. The bound here is the point: a regression that recurses or
// deadlocks fails this test by timing out rather than hanging the suite.
TEST(T6SelfReference, PointingTheShimAtItselfFailsInsteadOfRecursing) {
  RunRequest request;
  request.strategy = Strategy::kMasquerade;
  request.args = {"selfref"};
  request.real_libcuda = TESSERA_SHIM_MASQ_LIB;
  request.timeout_ms = 20000;

  const RunOutcome run = run_acceptance_app(request);
  ASSERT_TRUE(run.spawn.spawned) << run.describe();
  EXPECT_FALSE(run.spawn.timed_out)
      << "the shim hung when pointed at itself (I-5: never deadlock)\n"
      << run.describe();
  EXPECT_TRUE(run.spawn.exited) << "the shim crashed rather than returning an error\n"
                                << run.describe();
  // Fast, not just eventually: recursion would blow the stack slowly.
  EXPECT_LT(run.spawn.wall_us, 10'000'000) << run.describe();

  // An error, and the same error through three different call shapes: an
  // init, an unhooked symbol reached by trampoline, and a hooked wrapper.
  // This is also the one place the matrix sees a CUDA error travel from
  // underneath the shim to the application unchanged.
  EXPECT_EQ(run.app_number("init_failed").value_or(0), 1u) << run.describe();
  EXPECT_EQ(run.app_number("unhooked_failed").value_or(0), 1u) << run.describe();
  EXPECT_EQ(run.app_number("hooked_failed").value_or(0), 1u) << run.describe();
  EXPECT_EQ(run.spawn.exit_code, 0) << "the application did not see all three calls fail\n"
                                    << run.describe();
}

}  // namespace
}  // namespace tessera_test
