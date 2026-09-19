// In-process smoke tests for libtessera: load it, drive the fake driver
// through it, and check that both sides counted the same calls.
//
// The fork test lives in smoke_fork_test.cc, which needs TSAN_OPTIONS of its
// own. The full interception matrix (both strategies x four access paths) is a
// separate suite; this file is the shim's own smoke test.

#include "smoke_fixture.h"

namespace tessera_test {
namespace {

TEST_F(ShimSmokeTest, ReportsItsAbiVersion) {
  EXPECT_EQ(TESSERA_ABI_VERSION, shim().abi_version());
}

TEST_F(ShimSmokeTest, LoadingTheShimCallsNothing) {
  EXPECT_EQ(0u, shim().driver_calls_after_loading_the_shim)
      << "loading the shim entered a driver entry point; lazy init is not lazy";
}

TEST_F(ShimSmokeTest, LaunchesAreCountedOnceOnEachSide) {
  constexpr int kLaunches = 4;
  for (int i = 0; i < kLaunches; ++i) {
    ASSERT_CU(launch_(function_, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  }
  const tessera_stats_v1 s = stats();
  EXPECT_EQ(kLaunches, s.launches);
  EXPECT_EQ(0u, s.graph_launches);
  EXPECT_EQ(kLaunches, s.hooked_calls);
  EXPECT_EQ(0u, s.forwarded_unhooked);
  EXPECT_EQ(0u, s.bypassed_lookups);
  // The three counts T1 requires to agree: the calls the application made,
  // the shim's per-symbol counter, and the fake's.
  EXPECT_EQ(kLaunches, shim().symbol_count("cuLaunchKernel"));
  EXPECT_EQ(kLaunches, tf_symbol_count("cuLaunchKernel"));
  EXPECT_EQ(0u, tf_symbol_count("cuLaunchKernel_ptsz"));
}

TEST_F(ShimSmokeTest, UnhookedSymbolsReachTheDriverThroughATrampoline) {
  int count = 0;
  ASSERT_CU(symbol_as<FnDeviceGetCount>(shim().handle, "cuDeviceGetCount")(&count));
  EXPECT_EQ(1, count) << "the out-parameter did not come back from the fake";

  const tessera_stats_v1 s = stats();
  EXPECT_EQ(1u, s.forwarded_unhooked);
  EXPECT_EQ(0u, s.hooked_calls);
  EXPECT_EQ(1u, shim().symbol_count("cuDeviceGetCount"));
  EXPECT_EQ(1u, tf_symbol_count("cuDeviceGetCount"));
}

TEST_F(ShimSmokeTest, BothMemoryVariantsAreHookedSeparately) {
  // A quota that only covers _v2 is a quota with a hole; the v1 entry point
  // is a different symbol with different argument types (M2 depends on this).
  unsigned int v1 = 0;
  CUdeviceptr v2 = 0;
  ASSERT_CU(symbol_as<FnMemAllocV1>(shim().handle, "cuMemAlloc")(&v1, 256));
  ASSERT_CU(symbol_as<FnMemAllocV2>(shim().handle, "cuMemAlloc_v2")(&v2, 256));
  EXPECT_NE(0u, v1);
  EXPECT_NE(0u, v2);
  EXPECT_EQ(1u, shim().symbol_count("cuMemAlloc"));
  EXPECT_EQ(1u, shim().symbol_count("cuMemAlloc_v2"));
  EXPECT_EQ(1u, tf_symbol_count("cuMemAlloc"));
  EXPECT_EQ(1u, tf_symbol_count("cuMemAlloc_v2"));
  EXPECT_CU(symbol_as<FnMemFreeV2>(shim().handle, "cuMemFree_v2")(v2));
}

TEST_F(ShimSmokeTest, ProcAddressReturnsTheShimsWrapperForTheVariantTheDriverChose) {
  const auto get_proc = symbol_as<FnGetProcAddressV2>(shim().handle, "cuGetProcAddress_v2");
  ASSERT_NE(nullptr, get_proc);

  void* resolved = nullptr;
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  ASSERT_CU(
      get_proc("cuLaunchKernel", &resolved, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &status));
  EXPECT_EQ(CU_GET_PROC_ADDRESS_SUCCESS, status);
  EXPECT_EQ(dlsym(shim().handle, "cuLaunchKernel"), resolved)
      << "cuGetProcAddress handed back something that is not the shim's wrapper";
  EXPECT_NE(tf_symbol_address("cuLaunchKernel"), nullptr);

  // And the wrapper it returned really is a wrapper: calling it moves both
  // sides' counters.
  FnLaunchKernel fn = nullptr;
  std::memcpy(&fn, &resolved, sizeof(fn));
  ASSERT_CU(fn(function_, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  EXPECT_EQ(1u, shim().symbol_count("cuLaunchKernel"));
  EXPECT_EQ(1u, tf_symbol_count("cuLaunchKernel"));
  EXPECT_EQ(0u, stats().bypassed_lookups);
}

TEST_F(ShimSmokeTest, PerThreadDefaultStreamFlagResolvesToThePtszWrapper) {
  const auto get_proc = symbol_as<FnGetProcAddressV2>(shim().handle, "cuGetProcAddress_v2");
  void* resolved = nullptr;
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  ASSERT_CU(get_proc("cuLaunchKernel", &resolved, CUDA_VERSION,
                     CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status));
  EXPECT_EQ(dlsym(shim().handle, "cuLaunchKernel_ptsz"), resolved);

  FnLaunchKernel fn = nullptr;
  std::memcpy(&fn, &resolved, sizeof(fn));
  ASSERT_CU(fn(function_, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  EXPECT_EQ(1, tf_last_stream_was_per_thread_default())
      << "the NULL stream did not resolve to the per-thread default stream, so the _ptsz entry "
         "point is not the one that ran";
  EXPECT_EQ(1u, shim().symbol_count("cuLaunchKernel_ptsz"));
  EXPECT_EQ(1u, tf_symbol_count("cuLaunchKernel_ptsz"));
  EXPECT_EQ(0u, shim().symbol_count("cuLaunchKernel"));
}

TEST_F(ShimSmokeTest, VersionedLookupsFollowTheDriversChoice) {
  const auto get_proc_v1 = symbol_as<FnGetProcAddressV1>(shim().handle, "cuGetProcAddress");
  ASSERT_NE(nullptr, get_proc_v1);

  void* old_alloc = nullptr;
  ASSERT_CU(get_proc_v1("cuMemAlloc", &old_alloc, 3000, CU_GET_PROC_ADDRESS_DEFAULT));
  EXPECT_EQ(dlsym(shim().handle, "cuMemAlloc"), old_alloc) << "3000 should resolve to v1";

  void* new_alloc = nullptr;
  ASSERT_CU(get_proc_v1("cuMemAlloc", &new_alloc, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT));
  EXPECT_EQ(dlsym(shim().handle, "cuMemAlloc_v2"), new_alloc) << "12060 should resolve to _v2";
  EXPECT_NE(old_alloc, new_alloc);
  EXPECT_EQ(0u, stats().bypassed_lookups);
}

TEST_F(ShimSmokeTest, LookingUpTheLookupItselfReturnsTheShim) {
  // This is what cudart does first, and a shim that misses it sees nothing.
  const auto get_proc = symbol_as<FnGetProcAddressV2>(shim().handle, "cuGetProcAddress_v2");
  void* resolved = nullptr;
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  ASSERT_CU(
      get_proc("cuGetProcAddress", &resolved, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &status));
  EXPECT_EQ(dlsym(shim().handle, "cuGetProcAddress_v2"), resolved);

  void* older = nullptr;
  ASSERT_CU(get_proc("cuGetProcAddress", &older, 11030, CU_GET_PROC_ADDRESS_DEFAULT, &status));
  EXPECT_EQ(dlsym(shim().handle, "cuGetProcAddress"), older);
}

TEST_F(ShimSmokeTest, UnknownSymbolsAreForwardedUntouched) {
  const auto get_proc = symbol_as<FnGetProcAddressV2>(shim().handle, "cuGetProcAddress_v2");
  void* resolved = reinterpret_cast<void*>(0x1);
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
  EXPECT_EQ(CUDA_ERROR_NOT_FOUND, get_proc("cuNoSuchEntryPoint", &resolved, CUDA_VERSION,
                                           CU_GET_PROC_ADDRESS_DEFAULT, &status));
  EXPECT_EQ(CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND, status);
  EXPECT_EQ(0u, stats().bypassed_lookups) << "a symbol we do not hook is not a bypassed lookup";

  // A symbol the shim does not hook resolves to the real driver's own entry
  // point, unchanged.
  void* unhooked = nullptr;
  ASSERT_CU(
      get_proc("cuDeviceGetCount", &unhooked, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &status));
  tf_fn_ptr as_fn = nullptr;
  std::memcpy(&as_fn, &unhooked, sizeof(as_fn));
  EXPECT_EQ(tf_symbol_address("cuDeviceGetCount"), as_fn);
  EXPECT_EQ(0u, stats().bypassed_lookups);
}

TEST_F(ShimSmokeTest, EveryDriverExportIsReachableThroughTheShim) {
  // Interception completeness: when the shim IS libcuda.so.1, a symbol it does
  // not define does not exist as far as the application is concerned.
  size_t missing = 0;
  for (const char* name : kDriverExports) {
    if (dlsym(shim().handle, name) == nullptr) {
      ++missing;
      EXPECT_NE(nullptr, dlsym(shim().handle, name)) << name << " is not exported by the shim";
    }
  }
  EXPECT_EQ(0u, missing);
  // A floor, not a fixed number: the driver stub exports 640 symbols on
  // x86-64 and 641 on sbsa, and the list is generated per architecture.
  EXPECT_GT(TESSERA_SYMBOL_COUNT, 500u);
}

TEST_F(ShimSmokeTest, TheSymbolTableIsSortedForBinarySearch) {
  // tessera_get_symbol_count() binary-searches this table, so the generator's
  // byte-order sort is a precondition, not a formatting choice.
  for (size_t i = 1; i < TESSERA_SYMBOL_COUNT; ++i) {
    ASSERT_LT(std::strcmp(kDriverExports[i - 1], kDriverExports[i]), 0)
        << kDriverExports[i - 1] << " and " << kDriverExports[i] << " are out of order";
  }
  // And a name it does not export is not mistaken for one it does.
  EXPECT_EQ(0u, shim().symbol_count("cuNoSuchEntryPoint"));
  EXPECT_EQ(0u, shim().symbol_count(""));
}

// G9 depends on this separation: a graph launch is one API call but many
// kernels, so counting it as a launch would make the shim's number and CUPTI's
// number incomparable.
TEST_F(ShimSmokeTest, GraphLaunchesAreCountedApartFromLaunches) {
  CUstream stream = nullptr;
  ASSERT_CU(symbol_as<FnStreamCreate>(shim().handle, "cuStreamCreate")(&stream, 0));
  ASSERT_CU(symbol_as<FnStreamBeginCapture>(shim().handle, "cuStreamBeginCapture_v2")(
      stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  ASSERT_CU(launch_(function_, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
  CUgraph graph = nullptr;
  ASSERT_CU(symbol_as<FnStreamEndCapture>(shim().handle, "cuStreamEndCapture")(stream, &graph));
  ASSERT_NE(nullptr, graph);
  CUgraphExec exec = nullptr;
  ASSERT_CU(
      symbol_as<FnGraphInstantiate>(shim().handle, "cuGraphInstantiateWithFlags")(&exec, graph, 0));

  const uint64_t launches_before = stats().launches;
  ASSERT_CU(symbol_as<FnGraphLaunch>(shim().handle, "cuGraphLaunch")(exec, stream));
  const tessera_stats_v1 s = stats();
  EXPECT_EQ(1u, s.graph_launches);
  EXPECT_EQ(launches_before, s.launches) << "a graph launch was counted as a kernel launch";
  EXPECT_EQ(1u, shim().symbol_count("cuGraphLaunch"));
  EXPECT_EQ(1u, tf_symbol_count("cuGraphLaunch"));
  // The capture hooks are wired up too, which is where M2's "never gate during
  // capture" rule will live.
  EXPECT_EQ(1u, shim().symbol_count("cuStreamBeginCapture_v2"));
  EXPECT_EQ(1u, shim().symbol_count("cuStreamEndCapture"));
}

TEST_F(ShimSmokeTest, ResetAndDump) {
  ASSERT_CU(launch_(function_, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  ASSERT_EQ(1u, stats().launches);

  const std::string path = std::string(::testing::TempDir()) + "/tessera_shim_smoke_stats.json";
  ASSERT_EQ(0, shim().dump_stats(path.c_str()));
  const std::string json = read_file(path);
  std::remove(path.c_str());
  ASSERT_FALSE(json.empty());
  EXPECT_NE(std::string::npos, json.find("\"launches\":1"));
  EXPECT_NE(std::string::npos, json.find("\"cuLaunchKernel\":1"));

  shim().reset_stats();
  EXPECT_EQ(0u, stats().launches);
  EXPECT_EQ(0u, shim().symbol_count("cuLaunchKernel"));
  // Resetting the counters must not have cost us the resolved pointers.
  EXPECT_CU(launch_(function_, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  EXPECT_EQ(1u, stats().launches);
}

}  // namespace
}  // namespace tessera_test
