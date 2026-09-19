// The tf_* control API: counters, forced results, the stats file, and reset.

#include <spawn.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <sys/wait.h>

#include "fake_fixture.h"

extern char** environ;

namespace tessera_test {
namespace {

// Parses the flat {"name":count,...} object the fake writes. Deliberately not a
// general JSON parser: if the format ever drifts, this should fail loudly.
std::map<std::string, uint64_t> parse_stats(const std::string& path) {
  std::map<std::string, uint64_t> out;
  std::ifstream in(path);
  if (!in) {
    return out;
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  size_t i = text.find('{');
  if (i == std::string::npos) {
    return out;
  }
  ++i;
  while (true) {
    const size_t key_start = text.find('"', i);
    if (key_start == std::string::npos) {
      break;
    }
    const size_t key_end = text.find('"', key_start + 1);
    if (key_end == std::string::npos) {
      break;
    }
    const size_t colon = text.find(':', key_end);
    if (colon == std::string::npos) {
      break;
    }
    size_t value_end = colon + 1;
    while (value_end < text.size() && text[value_end] >= '0' && text[value_end] <= '9') {
      ++value_end;
    }
    out[text.substr(key_start + 1, key_end - key_start - 1)] =
        std::strtoull(text.substr(colon + 1, value_end - colon - 1).c_str(), nullptr, 10);
    i = value_end;
  }
  return out;
}

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

class ControlTest : public FakeDriverTest {};

TEST_F(ControlTest, CountersAreKeyedByExportedSymbol) {
  EXPECT_EQ(1u, tf_symbol_count("cuInit"));
  EXPECT_EQ(1u, tf_symbol_count("cuDeviceGet"));
  EXPECT_EQ(1u, tf_symbol_count("cuCtxCreate_v2"));
  EXPECT_EQ(0u, tf_symbol_count("cuCtxCreate"));
  EXPECT_EQ(0u, tf_symbol_count("cuCtxCreate_v3"));
  EXPECT_EQ(0u, tf_symbol_count("not_a_symbol_at_all"));

  int count = 0;
  for (int i = 0; i < 5; ++i) {
    ASSERT_CU(cuDeviceGetCount(&count));
  }
  EXPECT_EQ(5u, tf_symbol_count("cuDeviceGetCount"));
}

TEST_F(ControlTest, ResetZeroesEverything) {
  int count = 0;
  ASSERT_CU(cuDeviceGetCount(&count));
  ASSERT_GT(tf_symbol_count("cuDeviceGetCount"), 0u);
  tf_advance_time_us(1000);
  ASSERT_GT(tf_now_us(), 0u);

  tf_reset();
  EXPECT_EQ(0u, tf_symbol_count("cuDeviceGetCount"));
  EXPECT_EQ(0u, tf_symbol_count("cuInit"));
  // The driver is uninitialised again.
  EXPECT_EQ(CUDA_ERROR_NOT_INITIALIZED, cuDeviceGetCount(&count));
}

TEST_F(ControlTest, AForcedResultIsReturnedAndStillCounted) {
  tf_set_result("cuStreamCreate", CUDA_ERROR_OUT_OF_MEMORY);
  CUstream s = nullptr;
  EXPECT_EQ(CUDA_ERROR_OUT_OF_MEMORY, cuStreamCreate(&s, 0));
  EXPECT_EQ(nullptr, s);
  EXPECT_EQ(1u, tf_symbol_count("cuStreamCreate"));

  // CUDA_SUCCESS clears the injection.
  tf_set_result("cuStreamCreate", CUDA_SUCCESS);
  ASSERT_CU(cuStreamCreate(&s, 0));
  EXPECT_NE(nullptr, s);
  EXPECT_EQ(2u, tf_symbol_count("cuStreamCreate"));
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(ControlTest, ForcedResultsArePerExportedVariant) {
  CUfunction k = tf_register_kernel("k", nullptr, 1);
  tf_set_result("cuLaunchKernel_ptsz", CUDA_ERROR_LAUNCH_FAILED);

  EXPECT_CU(cuLaunchKernel(k, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  EXPECT_EQ(CUDA_ERROR_LAUNCH_FAILED,
            cuLaunchKernel_ptsz(k, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  EXPECT_EQ(1u, tf_symbol_count("cuLaunchKernel"));
  EXPECT_EQ(1u, tf_symbol_count("cuLaunchKernel_ptsz"));
  // The forced launch never reached a stream.
  EXPECT_EQ(1u, tf_stream_launch_count(tf_legacy_default_stream()));
  EXPECT_EQ(0u, tf_stream_launch_count(tf_per_thread_default_stream()));

  tf_clear_results();
  EXPECT_CU(cuLaunchKernel_ptsz(k, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  EXPECT_EQ(1u, tf_stream_launch_count(tf_per_thread_default_stream()));
  ASSERT_CU(cuCtxSynchronize());
}

TEST_F(ControlTest, InjectingIntoAnUnknownSymbolIsIgnored) {
  tf_set_result("cuNotARealSymbol", CUDA_ERROR_UNKNOWN);
  int count = 0;
  EXPECT_CU(cuDeviceGetCount(&count));
}

TEST_F(ControlTest, ResetClearsForcedResults) {
  tf_set_result("cuDeviceGetCount", CUDA_ERROR_NOT_SUPPORTED);
  int count = 0;
  EXPECT_EQ(CUDA_ERROR_NOT_SUPPORTED, cuDeviceGetCount(&count));
  tf_reset();
  ASSERT_CU(cuInit(0));
  EXPECT_CU(cuDeviceGetCount(&count));
}

TEST_F(ControlTest, StatsFileRoundTrip) {
  CUfunction k = tf_register_kernel("k", nullptr, 1);
  for (int i = 0; i < 4; ++i) {
    ASSERT_CU(cuLaunchKernel(k, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  }
  ASSERT_CU(cuLaunchKernel_ptsz(k, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
  ASSERT_CU(cuCtxSynchronize());

  const std::string path = temp_path("inproc");
  ASSERT_EQ(0, tf_dump_stats(path.c_str()));
  const std::map<std::string, uint64_t> stats = parse_stats(path);
  std::remove(path.c_str());

  ASSERT_FALSE(stats.empty());
  EXPECT_EQ(4u, stats.at("cuLaunchKernel"));
  EXPECT_EQ(1u, stats.at("cuLaunchKernel_ptsz"));
  EXPECT_EQ(1u, stats.at("cuInit"));
  EXPECT_EQ(0u, stats.count("cuGraphLaunch")) << "uncalled symbols must be omitted";

  // Every entry in the file agrees with tf_symbol_count.
  for (const auto& entry : stats) {
    EXPECT_GE(tf_symbol_index(entry.first.c_str()), 0) << entry.first;
    EXPECT_EQ(tf_symbol_count(entry.first.c_str()), entry.second) << entry.first;
  }
}

TEST_F(ControlTest, DumpToAnUnwritablePathFails) {
  EXPECT_EQ(-1, tf_dump_stats("/this/directory/does/not/exist/stats.json"));
  EXPECT_EQ(-1, tf_dump_stats(nullptr));
  EXPECT_EQ(-1, tf_dump_stats(""));
}

// The shim runs its "application" in a child process, so the counters have to
// reach a file at exit without anyone calling tf_dump_stats.
TEST_F(ControlTest, AChildProcessWritesItsCountersAtExit) {
  const std::string path = temp_path("child");
  std::remove(path.c_str());

  std::vector<std::string> argv_storage;
  const char* emulator = TESSERA_TEST_EMULATOR;
  if (emulator != nullptr && emulator[0] != '\0') {
    std::istringstream words(emulator);
    std::string word;
    while (words >> word) {
      argv_storage.push_back(word);
    }
  }
  argv_storage.push_back(TESSERA_STATS_CHILD_PATH);

  std::vector<char*> argv;
  argv.reserve(argv_storage.size() + 1);
  for (std::string& a : argv_storage) {
    argv.push_back(a.data());
  }
  argv.push_back(nullptr);

  const std::string env_entry = "TESSERA_FAKE_STATS_FILE=" + path;
  std::vector<char*> envp;
  for (char** e = environ; *e != nullptr; ++e) {
    if (std::strncmp(*e, "TESSERA_FAKE_STATS_FILE=", 24) != 0) {
      envp.push_back(*e);
    }
  }
  std::string env_copy = env_entry;
  envp.push_back(env_copy.data());
  envp.push_back(nullptr);

  pid_t pid = 0;
  ASSERT_EQ(0, posix_spawn(&pid, argv[0], nullptr, nullptr, argv.data(), envp.data()))
      << "spawning " << argv[0];
  int status = 0;
  ASSERT_EQ(pid, waitpid(pid, &status, 0));
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(0, WEXITSTATUS(status));

  const std::map<std::string, uint64_t> stats = parse_stats(path);
  std::remove(path.c_str());
  ASSERT_FALSE(stats.empty()) << "no stats written to " << path;
  EXPECT_EQ(1u, stats.at("cuInit"));
  EXPECT_EQ(3u, stats.at("cuLaunchKernel"));
  EXPECT_EQ(2u, stats.at("cuLaunchKernel_ptsz"));
  EXPECT_EQ(1u, stats.at("cuCtxSynchronize"));

  // The parent's own counters are untouched by the child.
  EXPECT_EQ(0u, tf_symbol_count("cuLaunchKernel"));
}

TEST_F(ControlTest, KernelRegistrationIsIdempotentByName) {
  CUfunction a = tf_register_kernel("same", nullptr, 10);
  CUfunction b = tf_register_kernel("same", nullptr, 20);
  EXPECT_EQ(a, b);

  CUmodule m = nullptr;
  const char image[] = "not a real cubin";
  ASSERT_CU(cuModuleLoadData(&m, image));
  CUfunction from_module = nullptr;
  ASSERT_CU(cuModuleGetFunction(&from_module, m, "same"));
  EXPECT_EQ(a, from_module);

  // An unregistered name is created on demand with the default duration.
  tf_set_default_kernel_duration_us(250);
  CUfunction fresh = nullptr;
  ASSERT_CU(cuModuleGetFunction(&fresh, m, "never_registered"));
  ASSERT_NE(nullptr, fresh);
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  const uint64_t t0 = tf_now_us();
  ASSERT_CU(cuLaunchKernel(fresh, 1, 1, 1, 1, 1, 1, 0, s, nullptr, nullptr));
  ASSERT_CU(cuStreamSynchronize(s));
  EXPECT_EQ(t0 + 250u, tf_now_us());

  ASSERT_CU(cuStreamDestroy(s));
  ASSERT_CU(cuModuleUnload(m));
}

TEST_F(ControlTest, NoControlCallHasFailed) {
  // The tf_* functions that return void swallow nothing: a failed allocation
  // would show up here rather than disappearing.
  EXPECT_EQ(0u, tf_control_failure_count());
  tf_set_time_mode(1);
  tf_advance_time_us(10);
  tf_set_result("cuInit", CUDA_ERROR_UNKNOWN);
  tf_clear_results();
  EXPECT_NE(nullptr, tf_register_kernel("probe", nullptr, 1));
  EXPECT_EQ(0u, tf_control_failure_count());
}

TEST_F(ControlTest, ErrorNamesAndStringsAreAvailable) {
  const char* name = nullptr;
  const char* text = nullptr;
  ASSERT_CU(cuGetErrorName(CUDA_ERROR_OUT_OF_MEMORY, &name));
  ASSERT_CU(cuGetErrorString(CUDA_ERROR_OUT_OF_MEMORY, &text));
  EXPECT_STREQ("CUDA_ERROR_OUT_OF_MEMORY", name);
  EXPECT_NE(nullptr, text);
  EXPECT_GT(std::strlen(text), 0u);
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuGetErrorName(CUDA_ERROR_MAP_FAILED, &name));
}

TEST_F(ControlTest, ContextStackBehavesLikeCudas) {
  CUcontext current = nullptr;
  ASSERT_CU(cuCtxGetCurrent(&current));
  EXPECT_EQ(ctx_, current);

  CUcontext second = nullptr;
  ASSERT_CU(cuCtxCreate(&second, 0, dev_));
  ASSERT_CU(cuCtxGetCurrent(&current));
  EXPECT_EQ(second, current);

  CUcontext popped = nullptr;
  ASSERT_CU(cuCtxPopCurrent(&popped));
  EXPECT_EQ(second, popped);
  ASSERT_CU(cuCtxGetCurrent(&current));
  EXPECT_EQ(ctx_, current);

  ASSERT_CU(cuCtxPushCurrent(second));
  ASSERT_CU(cuCtxGetCurrent(&current));
  EXPECT_EQ(second, current);

  ASSERT_CU(cuCtxSetCurrent(ctx_));
  ASSERT_CU(cuCtxGetCurrent(&current));
  EXPECT_EQ(ctx_, current);

  ASSERT_CU(cuCtxDestroy(second));
  EXPECT_EQ(CUDA_ERROR_INVALID_CONTEXT, cuCtxSetCurrent(second));
}

TEST_F(ControlTest, ThePrimaryContextIsRefCounted) {
  CUcontext primary = nullptr;
  CUcontext again = nullptr;
  ASSERT_CU(cuDevicePrimaryCtxRetain(&primary, dev_));
  ASSERT_CU(cuDevicePrimaryCtxRetain(&again, dev_));
  EXPECT_EQ(primary, again);
  ASSERT_CU(cuDevicePrimaryCtxRelease(dev_));
  // Still alive after one release.
  ASSERT_CU(cuCtxPushCurrent(primary));
  ASSERT_CU(cuCtxPopCurrent(nullptr));
  ASSERT_CU(cuDevicePrimaryCtxRelease_v2(dev_));
  EXPECT_EQ(CUDA_ERROR_INVALID_CONTEXT, cuCtxPushCurrent(primary));
}

TEST_F(ControlTest, CallsBeforeInitialisationAreRejected) {
  tf_reset();
  int count = 0;
  CUdevice d = 0;
  CUcontext c = nullptr;
  EXPECT_EQ(CUDA_ERROR_NOT_INITIALIZED, cuDeviceGetCount(&count));
  EXPECT_EQ(CUDA_ERROR_NOT_INITIALIZED, cuDeviceGet(&d, 0));
  EXPECT_EQ(CUDA_ERROR_NOT_INITIALIZED, cuCtxCreate(&c, 0, 0));
  // These do not need cuInit, exactly as on a real driver.
  int version = 0;
  EXPECT_CU(cuDriverGetVersion(&version));
  EXPECT_GE(version, 12040);
}

TEST_F(ControlTest, CallsWithoutACurrentContextAreRejected) {
  ASSERT_CU(cuCtxPopCurrent(nullptr));
  CUstream s = nullptr;
  CUdeviceptr d = 0;
  EXPECT_EQ(CUDA_ERROR_INVALID_CONTEXT, cuStreamCreate(&s, 0));
  EXPECT_EQ(CUDA_ERROR_INVALID_CONTEXT, cuMemAlloc(&d, 16));
  EXPECT_EQ(CUDA_ERROR_INVALID_CONTEXT, cuCtxSynchronize());
  ASSERT_CU(cuCtxPushCurrent(ctx_));
  EXPECT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(cuStreamDestroy(s));
}

}  // namespace
}  // namespace tessera_test
