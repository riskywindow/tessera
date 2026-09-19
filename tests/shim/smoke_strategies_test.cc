// Both interposition strategies, end to end, with the tenant in a child
// process.
//
//   masquerade   the shim IS libcuda.so.1, first on LD_LIBRARY_PATH, and the
//                real driver comes from TESSERA_REAL_LIBCUDA;
//   preload      libtessera.so on LD_PRELOAD, with the fake found normally as
//                libcuda.so.1, so the dlsym hook is what catches the
//                dlopen+dlsym path.
//
// The tenant is an ordinary CUDA program (smoke_tenant_main.cc) that reaches
// the driver by three different access paths. Counters come back through the
// two stats files both libraries write at exit, because the application runs
// in a child process.

#include <fcntl.h>
#include <spawn.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <sys/wait.h>

#include <gtest/gtest.h>

extern "C" char** environ;

namespace tessera_test {
namespace {

std::string temp_path(const char* name) {
  std::ostringstream path;
  path << ::testing::TempDir() << "/tessera_" << name << "_" << ::getpid() << ".json";
  return path.str();
}

// open/read rather than stdio: these files are a few hundred bytes, and a
// descriptor has one failure mode instead of three.
std::string read_file(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return {};
  }
  std::string text;
  char buffer[4096];
  for (;;) {
    const ssize_t n = ::read(fd, buffer, sizeof(buffer));
    if (n <= 0) {
      break;
    }
    text.append(buffer, static_cast<size_t>(n));
  }
  ::close(fd);
  return text;
}

// Both stats files are flat JSON written by the libraries themselves, so a
// full parser would be more machinery than the format deserves: every value we
// want is an unsigned integer after "key":.
bool json_number(const std::string& json, const std::string& key, uint64_t* out) {
  const std::string needle = "\"" + key + "\":";
  const size_t at = json.find(needle);
  if (at == std::string::npos) {
    return false;
  }
  size_t i = at + needle.size();
  if (i >= json.size() || json[i] < '0' || json[i] > '9') {
    return false;
  }
  uint64_t value = 0;
  for (; i < json.size() && json[i] >= '0' && json[i] <= '9'; ++i) {
    value = (value * 10) + static_cast<uint64_t>(json[i] - '0');
  }
  *out = value;
  return true;
}

uint64_t count_of(const std::string& json, const std::string& key) {
  uint64_t value = 0;
  EXPECT_TRUE(json_number(json, key, &value)) << "no \"" << key << "\" in " << json;
  return value;
}

// Every "name": number pair inside the shim's "symbols" object, or every pair
// in the fake's flat object.
std::map<std::string, uint64_t> symbol_counts(const std::string& json, bool nested) {
  std::map<std::string, uint64_t> counts;
  size_t i = 0;
  if (nested) {
    const size_t at = json.find("\"symbols\":{");
    if (at == std::string::npos) {
      return counts;
    }
    i = at + std::strlen("\"symbols\":{");
  }
  while (i < json.size()) {
    const size_t key_start = json.find('"', i);
    if (key_start == std::string::npos) {
      break;
    }
    const size_t key_end = json.find('"', key_start + 1);
    if (key_end == std::string::npos) {
      break;
    }
    const std::string key = json.substr(key_start + 1, key_end - key_start - 1);
    size_t value_start = key_end + 1;
    if (value_start >= json.size() || json[value_start] != ':') {
      break;
    }
    ++value_start;
    uint64_t value = 0;
    size_t j = value_start;
    for (; j < json.size() && json[j] >= '0' && json[j] <= '9'; ++j) {
      value = (value * 10) + static_cast<uint64_t>(json[j] - '0');
    }
    if (j == value_start) {
      break;  // not a number: the object ended
    }
    if (key.compare(0, 2, "cu") == 0) {
      counts[key] = value;
    }
    i = j;
  }
  return counts;
}

struct TenantRun {
  int exit_code = -1;
  std::string shim_json;
  std::string fake_json;
};

// Runs the tenant with `overrides` layered over this process's environment.
TenantRun run_tenant(const char* mode, const std::vector<std::string>& overrides) {
  TenantRun run;
  const std::string shim_stats = temp_path("shim_stats");
  const std::string fake_stats = temp_path("fake_stats");
  std::remove(shim_stats.c_str());
  std::remove(fake_stats.c_str());

  std::vector<std::string> env = overrides;
  env.push_back("TESSERA_STATS_FILE=" + shim_stats);
  env.push_back("TESSERA_FAKE_STATS_FILE=" + fake_stats);

  std::vector<char*> envp;
  for (char** e = environ; *e != nullptr; ++e) {
    const char* eq = std::strchr(*e, '=');
    const std::string key(*e, eq != nullptr ? static_cast<size_t>(eq - *e) : std::strlen(*e));
    bool overridden = false;
    for (const std::string& entry : env) {
      if (entry.compare(0, key.size() + 1, key + "=") == 0) {
        overridden = true;
        break;
      }
    }
    if (!overridden) {
      envp.push_back(*e);
    }
  }
  for (std::string& entry : env) {
    envp.push_back(entry.data());
  }
  envp.push_back(nullptr);

  // Cross builds run the tenant under the emulator, exactly as CTest runs this
  // binary; there is no binfmt registration to fall back on.
  std::vector<std::string> argv_storage;
  const char* emulator = TESSERA_TEST_EMULATOR;
  if (emulator != nullptr && emulator[0] != '\0') {
    std::istringstream words(emulator);
    std::string word;
    while (words >> word) {
      argv_storage.push_back(word);
    }
  }
  argv_storage.emplace_back(TESSERA_SHIM_TENANT_PATH);
  argv_storage.emplace_back(mode);

  std::vector<char*> argv;
  argv.reserve(argv_storage.size() + 1);
  for (std::string& a : argv_storage) {
    argv.push_back(a.data());
  }
  argv.push_back(nullptr);

  pid_t pid = 0;
  const int spawned = posix_spawn(&pid, argv[0], nullptr, nullptr, argv.data(), envp.data());
  EXPECT_EQ(0, spawned) << "spawning " << argv[0];
  if (spawned != 0) {
    return run;
  }
  int status = 0;
  EXPECT_EQ(pid, waitpid(pid, &status, 0));
  if (WIFEXITED(status)) {
    run.exit_code = WEXITSTATUS(status);
  } else {
    ADD_FAILURE() << "tenant did not exit normally (status " << status << ")";
  }
  run.shim_json = read_file(shim_stats);
  run.fake_json = read_file(fake_stats);
  std::remove(shim_stats.c_str());
  std::remove(fake_stats.c_str());
  return run;
}

std::string preload_value() {
  // Under gcc's sanitizers the runtime is a shared object and must precede any
  // other preloaded library.
  const std::string sanitizer = TESSERA_SANITIZER_PRELOAD;
  if (!sanitizer.empty()) {
    return sanitizer + " " + TESSERA_SHIM_PRELOAD_PATH;
  }
  return TESSERA_SHIM_PRELOAD_PATH;
}

// The counts smoke_tenant_main.cc is written to produce.
constexpr uint64_t kLegacyLaunches = 7;
constexpr uint64_t kPerThreadLaunches = 1;

void check_workload(const TenantRun& run) {
  ASSERT_EQ(0, run.exit_code);
  ASSERT_FALSE(run.shim_json.empty()) << "the shim wrote no TESSERA_STATS_FILE";
  ASSERT_FALSE(run.fake_json.empty()) << "the fake wrote no TESSERA_FAKE_STATS_FILE";

  EXPECT_EQ(kLegacyLaunches + kPerThreadLaunches, count_of(run.shim_json, "launches"));
  EXPECT_EQ(0u, count_of(run.shim_json, "graph_launches"));
  EXPECT_EQ(0u, count_of(run.shim_json, "bypassed_lookups"));
  EXPECT_GT(count_of(run.shim_json, "forwarded_unhooked"), 0u)
      << "no unhooked symbol reached the driver through a trampoline";

  const std::map<std::string, uint64_t> shim = symbol_counts(run.shim_json, true);
  const std::map<std::string, uint64_t> fake = symbol_counts(run.fake_json, false);

  ASSERT_EQ(kLegacyLaunches, shim.at("cuLaunchKernel"));
  ASSERT_EQ(kPerThreadLaunches, shim.at("cuLaunchKernel_ptsz"));
  EXPECT_EQ(1u, shim.at("cuInit"));
  EXPECT_EQ(1u, shim.at("cuCtxCreate_v2"));
  EXPECT_EQ(1u, shim.at("cuStreamCreate"));
  EXPECT_EQ(2u, shim.at("cuGetProcAddress_v2"));

  // T1's central claim, in miniature: for every symbol the shim saw, the fake
  // saw the same number of calls. The shim never calls the driver on its own
  // behalf, so this has to hold symbol by symbol, in both directions.
  for (const auto& [name, count] : shim) {
    const auto found = fake.find(name);
    ASSERT_NE(fake.end(), found) << name << " reached the shim but never the driver";
    EXPECT_EQ(count, found->second) << name;
  }
  for (const auto& [name, count] : fake) {
    const auto found = shim.find(name);
    ASSERT_NE(shim.end(), found) << name << " reached the driver without passing the shim";
    EXPECT_EQ(count, found->second) << name;
  }
}

TEST(ShimStrategiesTest, MasqueradeSeesEveryCall) {
  const TenantRun run =
      run_tenant("workload", {
                                 std::string("LD_LIBRARY_PATH=") + TESSERA_SHIM_MASQ_DIR,
                                 std::string("LD_PRELOAD="),
                                 std::string("TESSERA_REAL_LIBCUDA=") + TESSERA_FAKE_MASQ_LIB,
                             });
  check_workload(run);
}

TEST(ShimStrategiesTest, PreloadSeesEveryCall) {
  const TenantRun run =
      run_tenant("workload", {
                                 std::string("LD_LIBRARY_PATH=") + TESSERA_FAKE_MASQ_DIR,
                                 "LD_PRELOAD=" + preload_value(),
                                 std::string("TESSERA_REAL_LIBCUDA=") + TESSERA_FAKE_MASQ_LIB,
                             });
  check_workload(run);
}

// T6: TESSERA_REAL_LIBCUDA pointing at the shim itself must produce an error,
// not infinite recursion. The tenant exits 0 only if cuInit failed; if the
// guard were missing, the child would recurse until it died instead, and this
// test's timeout would catch it.
TEST(ShimStrategiesTest, PointingTheShimAtItselfIsRefused) {
  const std::string self = std::string(TESSERA_SHIM_MASQ_DIR) + "/libcuda.so.1";
  const TenantRun run =
      run_tenant("selfref", {
                                std::string("LD_LIBRARY_PATH=") + TESSERA_SHIM_MASQ_DIR,
                                std::string("LD_PRELOAD="),
                                "TESSERA_REAL_LIBCUDA=" + self,
                            });
  EXPECT_EQ(0, run.exit_code) << "the shim did not refuse to forward to itself";
}

}  // namespace
}  // namespace tessera_test
