// cuGetProcAddress and cuGetProcAddress_v2.
//
// cudart resolves every driver entry point through these, including
// cuGetProcAddress itself, and it picks between versioned and
// per-thread-default-stream variants by (symbol, cudaVersion, flags). If the
// fake got that wrong, every shim interception test would be testing the wrong
// entry point.

#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "fake_fixture.h"

namespace tessera_test {
namespace {

constexpr int kCudaVersion = 12060;

cuuint64_t flags_for(int stream_kind) {
  switch (stream_kind) {
    case TF_STREAM_KIND_PER_THREAD:
      return CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
    case TF_STREAM_KIND_LEGACY:
      return CU_GET_PROC_ADDRESS_LEGACY_STREAM;
    default:
      return CU_GET_PROC_ADDRESS_DEFAULT;
  }
}

class ProcAddressTest : public FakeDriverTest {
 protected:
  static tf_fn_ptr resolve(const char* symbol, int version, cuuint64_t flags,
                           CUdriverProcAddressQueryResult* status, CUresult* result) {
    void* pfn = nullptr;
    *result = cuGetProcAddress(symbol, &pfn, version, flags, status);
    return as_fn(pfn);
  }
};

// Every row of the table must be reachable: asking for the row's base name at
// exactly the version that row was introduced, with the flag that matches its
// stream kind, must hand back that row's entry point and nothing else.
TEST_F(ProcAddressTest, EveryTableRowResolvesToItself) {
  const uint64_t n = tf_symbol_table_size();
  ASSERT_GT(n, 0u);
  for (uint64_t i = 0; i < n; ++i) {
    const char* exported = tf_symbol_name_at(i);
    const char* base = tf_symbol_base_name_at(i);
    const int version = tf_symbol_version_at(i);
    const int kind = tf_symbol_stream_kind_at(i);
    ASSERT_NE(nullptr, exported);
    ASSERT_NE(nullptr, base);

    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    CUresult r = CUDA_ERROR_UNKNOWN;
    const tf_fn_ptr got = resolve(base, version, flags_for(kind), &status, &r);
    EXPECT_EQ(CUDA_SUCCESS, r) << exported;
    EXPECT_EQ(CU_GET_PROC_ADDRESS_SUCCESS, status) << exported;
    EXPECT_EQ(tf_symbol_address(exported), got)
        << "resolving " << base << " at version " << version << " should give " << exported;
  }
}

// One version below a row's introduction must select the previous row for that
// base and stream kind, or report VERSION_NOT_SUFFICIENT if there is none.
TEST_F(ProcAddressTest, OneVersionEarlierSelectsThePreviousRow) {
  const uint64_t n = tf_symbol_table_size();
  for (uint64_t i = 0; i < n; ++i) {
    const char* base = tf_symbol_base_name_at(i);
    const int version = tf_symbol_version_at(i);
    const int kind = tf_symbol_stream_kind_at(i);

    // The best row strictly older than this one, same base and stream kind.
    const char* expected = nullptr;
    int expected_version = -1;
    for (uint64_t j = 0; j < n; ++j) {
      if (std::strcmp(tf_symbol_base_name_at(j), base) != 0) {
        continue;
      }
      if (tf_symbol_stream_kind_at(j) != kind) {
        continue;
      }
      const int vj = tf_symbol_version_at(j);
      if (vj >= version) {
        continue;
      }
      if (vj > expected_version) {
        expected_version = vj;
        expected = tf_symbol_name_at(j);
      }
    }

    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    CUresult r = CUDA_ERROR_UNKNOWN;
    const tf_fn_ptr got = resolve(base, version - 1, flags_for(kind), &status, &r);
    if (expected == nullptr) {
      EXPECT_EQ(CUDA_ERROR_NOT_FOUND, r) << base << " at " << (version - 1);
      EXPECT_EQ(CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, status) << base;
      EXPECT_EQ(nullptr, got) << base;
    } else {
      EXPECT_EQ(CUDA_SUCCESS, r) << base << " at " << (version - 1);
      EXPECT_EQ(CU_GET_PROC_ADDRESS_SUCCESS, status) << base;
      EXPECT_EQ(tf_symbol_address(expected), got) << base << " at " << (version - 1);
    }
  }
}

// A base name either has no stream-dependent variants at all, or has both a
// legacy and a per-thread-default-stream one. Anything else would leave a flag
// combination with no entry point behind it.
TEST_F(ProcAddressTest, StreamVariantsComeInPairs) {
  const uint64_t n = tf_symbol_table_size();
  std::map<std::string, std::set<int>> kinds;
  for (uint64_t i = 0; i < n; ++i) {
    kinds[tf_symbol_base_name_at(i)].insert(tf_symbol_stream_kind_at(i));
  }
  for (const auto& entry : kinds) {
    const std::set<int>& k = entry.second;
    const bool has_legacy = k.count(TF_STREAM_KIND_LEGACY) != 0;
    const bool has_pt = k.count(TF_STREAM_KIND_PER_THREAD) != 0;
    EXPECT_EQ(has_legacy, has_pt) << entry.first;
    if (has_legacy) {
      EXPECT_EQ(0u, k.count(TF_STREAM_KIND_NONE)) << entry.first;
    }
  }
}

TEST_F(ProcAddressTest, KnownVersionedSelections) {
  struct Row {
    const char* base;
    int version;
    cuuint64_t flags;
    const char* expected;
  };
  const Row rows[] = {
      {"cuMemAlloc", 3000, CU_GET_PROC_ADDRESS_DEFAULT, "cuMemAlloc"},
      {"cuMemAlloc", 3019, CU_GET_PROC_ADDRESS_DEFAULT, "cuMemAlloc"},
      {"cuMemAlloc", 3020, CU_GET_PROC_ADDRESS_DEFAULT, "cuMemAlloc_v2"},
      {"cuMemAlloc", 12060, CU_GET_PROC_ADDRESS_DEFAULT, "cuMemAlloc_v2"},
      {"cuCtxCreate", 2000, CU_GET_PROC_ADDRESS_DEFAULT, "cuCtxCreate"},
      {"cuCtxCreate", 3020, CU_GET_PROC_ADDRESS_DEFAULT, "cuCtxCreate_v2"},
      {"cuCtxCreate", 11040, CU_GET_PROC_ADDRESS_DEFAULT, "cuCtxCreate_v3"},
      {"cuCtxCreate", 12040, CU_GET_PROC_ADDRESS_DEFAULT, "cuCtxCreate_v3"},
      {"cuCtxCreate", 12050, CU_GET_PROC_ADDRESS_DEFAULT, "cuCtxCreate_v4"},
      {"cuStreamBeginCapture", 10000, CU_GET_PROC_ADDRESS_LEGACY_STREAM, "cuStreamBeginCapture"},
      {"cuStreamBeginCapture", 10010, CU_GET_PROC_ADDRESS_LEGACY_STREAM, "cuStreamBeginCapture_v2"},
      {"cuStreamBeginCapture", 10000, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
       "cuStreamBeginCapture_ptsz"},
      {"cuStreamBeginCapture", 12060, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
       "cuStreamBeginCapture_v2_ptsz"},
      {"cuMemcpyHtoD", 2000, CU_GET_PROC_ADDRESS_LEGACY_STREAM, "cuMemcpyHtoD"},
      {"cuMemcpyHtoD", 3020, CU_GET_PROC_ADDRESS_LEGACY_STREAM, "cuMemcpyHtoD_v2"},
      {"cuMemcpyHtoD", 12060, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
       "cuMemcpyHtoD_v2_ptds"},
      {"cuLaunchKernel", 4000, CU_GET_PROC_ADDRESS_DEFAULT, "cuLaunchKernel"},
      {"cuLaunchKernel", 12060, CU_GET_PROC_ADDRESS_LEGACY_STREAM, "cuLaunchKernel"},
      {"cuLaunchKernel", 12060, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
       "cuLaunchKernel_ptsz"},
      {"cuDeviceTotalMem", 3000, CU_GET_PROC_ADDRESS_DEFAULT, "cuDeviceTotalMem"},
      {"cuDeviceTotalMem", 3020, CU_GET_PROC_ADDRESS_DEFAULT, "cuDeviceTotalMem_v2"},
      {"cuGetProcAddress", 11030, CU_GET_PROC_ADDRESS_DEFAULT, "cuGetProcAddress"},
      {"cuGetProcAddress", 11080, CU_GET_PROC_ADDRESS_DEFAULT, "cuGetProcAddress"},
      {"cuGetProcAddress", 12000, CU_GET_PROC_ADDRESS_DEFAULT, "cuGetProcAddress_v2"},
      {"cuGetProcAddress", 12060, CU_GET_PROC_ADDRESS_DEFAULT, "cuGetProcAddress_v2"},
  };
  for (const Row& row : rows) {
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    CUresult r = CUDA_ERROR_UNKNOWN;
    const tf_fn_ptr got = resolve(row.base, row.version, row.flags, &status, &r);
    EXPECT_EQ(CUDA_SUCCESS, r) << row.base << " @" << row.version;
    EXPECT_EQ(CU_GET_PROC_ADDRESS_SUCCESS, status) << row.base << " @" << row.version;
    EXPECT_EQ(tf_symbol_address(row.expected), got)
        << row.base << " @" << row.version << " should be " << row.expected;
  }
}

TEST_F(ProcAddressTest, DefaultFlagMeansLegacyStream) {
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  CUresult r = CUDA_ERROR_UNKNOWN;
  const tf_fn_ptr d =
      resolve("cuStreamSynchronize", kCudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &status, &r);
  ASSERT_CU(r);
  EXPECT_EQ(tf_symbol_address("cuStreamSynchronize"), d);
  const tf_fn_ptr l =
      resolve("cuStreamSynchronize", kCudaVersion, CU_GET_PROC_ADDRESS_LEGACY_STREAM, &status, &r);
  ASSERT_CU(r);
  EXPECT_EQ(d, l);
}

TEST_F(ProcAddressTest, UnknownSymbolIsSymbolNotFound) {
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
  CUresult r = CUDA_SUCCESS;
  const tf_fn_ptr got =
      resolve("cuNoSuchDriverFunction", kCudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &status, &r);
  EXPECT_EQ(CUDA_ERROR_NOT_FOUND, r);
  EXPECT_EQ(CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND, status);
  EXPECT_EQ(nullptr, got);
}

// The exported variant names are not what an application asks for: the table is
// keyed by base name, exactly as the real driver's is.
TEST_F(ProcAddressTest, ExportedVariantNamesAreNotLookupKeys) {
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
  CUresult r = CUDA_SUCCESS;
  resolve("cuMemAlloc_v2", kCudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &status, &r);
  EXPECT_EQ(CUDA_ERROR_NOT_FOUND, r);
  EXPECT_EQ(CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND, status);
  resolve("cuLaunchKernel_ptsz", kCudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &status, &r);
  EXPECT_EQ(CUDA_ERROR_NOT_FOUND, r);
}

TEST_F(ProcAddressTest, TooOldAVersionIsVersionNotSufficient) {
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
  CUresult r = CUDA_SUCCESS;
  const tf_fn_ptr got = resolve("cuMemAlloc", 1000, CU_GET_PROC_ADDRESS_DEFAULT, &status, &r);
  EXPECT_EQ(CUDA_ERROR_NOT_FOUND, r);
  EXPECT_EQ(CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, status);
  EXPECT_EQ(nullptr, got);

  // A symbol that exists, asked for before the version that introduced it.
  resolve("cuLaunchKernelEx", 11050, CU_GET_PROC_ADDRESS_DEFAULT, &status, &r);
  EXPECT_EQ(CUDA_ERROR_NOT_FOUND, r);
  EXPECT_EQ(CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, status);
}

TEST_F(ProcAddressTest, AskingForBothStreamFlagsIsInvalid) {
  void* pfn = nullptr;
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
  const CUresult r = cuGetProcAddress(
      "cuLaunchKernel", &pfn, kCudaVersion,
      CU_GET_PROC_ADDRESS_LEGACY_STREAM | CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status);
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, r);
  EXPECT_EQ(nullptr, pfn);
}

TEST_F(ProcAddressTest, NullArgumentsAreRejected) {
  void* pfn = nullptr;
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE,
            cuGetProcAddress(nullptr, &pfn, kCudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, nullptr));
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuGetProcAddress("cuInit", nullptr, kCudaVersion,
                                                       CU_GET_PROC_ADDRESS_DEFAULT, nullptr));
}

// This is the one that matters for the shim: cudart asks the driver for
// "cuGetProcAddress" and then uses what it gets back for everything else.
TEST_F(ProcAddressTest, TheLookupFunctionCanFindItself) {
  void* pfn = nullptr;
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  ASSERT_CU(cuGetProcAddress("cuGetProcAddress", &pfn, kCudaVersion, CU_GET_PROC_ADDRESS_DEFAULT,
                             &status));
  EXPECT_EQ(CU_GET_PROC_ADDRESS_SUCCESS, status);
  ASSERT_EQ(tf_symbol_address("cuGetProcAddress_v2"), as_fn(pfn));

  // Use the pointer we were handed to resolve something else, as cudart does.
  auto v2 = reinterpret_cast<TF_PFN_cuGetProcAddress_v2>(as_fn(pfn));
  void* inner = nullptr;
  ASSERT_CU(v2("cuMemAlloc", &inner, kCudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &status));
  EXPECT_EQ(tf_symbol_address("cuMemAlloc_v2"), as_fn(inner));
}

TEST_F(ProcAddressTest, TheVersionOneLookupFunctionWorks) {
  auto v1 = reinterpret_cast<TF_PFN_cuGetProcAddress_v1>(tf_symbol_address("cuGetProcAddress"));
  ASSERT_NE(nullptr, v1);
  void* pfn = nullptr;
  ASSERT_CU(
      v1("cuStreamSynchronize", &pfn, kCudaVersion, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM));
  EXPECT_EQ(tf_symbol_address("cuStreamSynchronize_ptsz"), as_fn(pfn));
  EXPECT_EQ(1u, tf_symbol_count("cuGetProcAddress"));
}

TEST_F(ProcAddressTest, TheReportedDriverVersionIsACeiling) {
  tf_set_driver_version(11000);
  int reported = 0;
  ASSERT_CU(cuDriverGetVersion(&reported));
  EXPECT_EQ(11000, reported);

  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  CUresult r = CUDA_ERROR_UNKNOWN;
  // Asking for a newer CUDA than the driver claims cannot conjure newer entry
  // points: cuCtxCreate_v3 arrived in 11040.
  const tf_fn_ptr got = resolve("cuCtxCreate", 12060, CU_GET_PROC_ADDRESS_DEFAULT, &status, &r);
  ASSERT_CU(r);
  EXPECT_EQ(tf_symbol_address("cuCtxCreate_v2"), got);

  tf_set_driver_version(12060);
  const tf_fn_ptr newer = resolve("cuCtxCreate", 12060, CU_GET_PROC_ADDRESS_DEFAULT, &status, &r);
  ASSERT_CU(r);
  EXPECT_EQ(tf_symbol_address("cuCtxCreate_v4"), newer);
}

TEST_F(ProcAddressTest, AResolvedPointerReallyCallsThatEntryPoint) {
  void* pfn = nullptr;
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  ASSERT_CU(cuGetProcAddress("cuLaunchKernel", &pfn, kCudaVersion,
                             CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM, &status));
  auto launch = reinterpret_cast<TF_PFN_cuLaunchKernel>(as_fn(pfn));
  ASSERT_NE(nullptr, launch);

  CUfunction k = tf_register_kernel("k", nullptr, 1);
  ASSERT_CU(launch(k, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));

  EXPECT_EQ(1u, tf_symbol_count("cuLaunchKernel_ptsz"));
  EXPECT_EQ(0u, tf_symbol_count("cuLaunchKernel"));
  EXPECT_EQ(1, tf_last_stream_was_per_thread_default());
  ASSERT_CU(cuCtxSynchronize());
}

TEST_F(ProcAddressTest, LookupWorksBeforeInitialisation) {
  tf_reset();  // back to an uninitialised driver, with no context
  void* pfn = nullptr;
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  EXPECT_EQ(CUDA_SUCCESS,
            cuGetProcAddress("cuInit", &pfn, kCudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &status));
  EXPECT_EQ(tf_symbol_address("cuInit"), as_fn(pfn));
  // ... but anything else still needs cuInit.
  int count = 0;
  EXPECT_EQ(CUDA_ERROR_NOT_INITIALIZED, cuDeviceGetCount(&count));
}

TEST_F(ProcAddressTest, TableAccessorsRejectOutOfRangeRows) {
  const uint64_t n = tf_symbol_table_size();
  EXPECT_EQ(nullptr, tf_symbol_name_at(n));
  EXPECT_EQ(nullptr, tf_symbol_base_name_at(n));
  EXPECT_EQ(-1, tf_symbol_version_at(n));
  EXPECT_EQ(-1, tf_symbol_stream_kind_at(n));
  EXPECT_EQ(-1, tf_symbol_index("cuNotASymbol"));
  EXPECT_EQ(nullptr, tf_symbol_address("cuNotASymbol"));
  ASSERT_GE(tf_symbol_index("cuInit"), 0);
  EXPECT_STREQ("cuInit", tf_symbol_name_at(static_cast<uint64_t>(tf_symbol_index("cuInit"))));
}

}  // namespace
}  // namespace tessera_test
