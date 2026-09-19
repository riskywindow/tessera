// The shape of the shared object other agents depend on: the exported symbol
// set, and the masquerading copy that a test drops on LD_LIBRARY_PATH so that
// dlopen("libcuda.so.1") finds the fake.

#include <dlfcn.h>

#include <string>
#include <vector>

#include "fake_fixture.h"

namespace tessera_test {
namespace {

class ExportsTest : public ::testing::Test {};

TEST_F(ExportsTest, EveryTableEntryPointIsAnExportedDynamicSymbol) {
  const uint64_t n = tf_symbol_table_size();
  ASSERT_GT(n, 0u);
  for (uint64_t i = 0; i < n; ++i) {
    const char* name = tf_symbol_name_at(i);
    ASSERT_NE(nullptr, name);
    void* p = dlsym(RTLD_DEFAULT, name);
    EXPECT_NE(nullptr, p) << name << " is in the table but not exported";
    if (p != nullptr) {
      EXPECT_EQ(tf_symbol_address(name), as_fn(p)) << name;
    }
  }
}

TEST_F(ExportsTest, TheControlApiIsExported) {
  const char* names[] = {"tf_reset",
                         "tf_symbol_count",
                         "tf_dump_stats",
                         "tf_register_kernel",
                         "tf_set_kernel_duration_us",
                         "tf_set_time_mode",
                         "tf_advance_time_us",
                         "tf_now_us",
                         "tf_set_result",
                         "tf_set_fork_mode",
                         "tf_stream_launch_count",
                         "tf_last_stream_was_per_thread_default",
                         "tf_symbol_table_size",
                         "tf_symbol_address"};
  for (const char* name : names) {
    EXPECT_NE(nullptr, dlsym(RTLD_DEFAULT, name)) << name;
  }
}

TEST_F(ExportsTest, NothingElseLeaksOutOfTheLibrary) {
  // The version script is generated from the entry-point table, so these
  // internal names must not be reachable.
  const char* hidden[] = {"_ZN2tf3Sim3getEv", "tessera_fake_internal", "cuNotAnEntryPoint"};
  for (const char* name : hidden) {
    EXPECT_EQ(nullptr, dlsym(RTLD_DEFAULT, name)) << name;
  }
}

// The directory <build>/fake_driver/fake_masq holds libcuda.so.1 and the
// libcuda.so symlink and nothing else, so a shim test can put exactly that on
// LD_LIBRARY_PATH.
TEST_F(ExportsTest, TheMasqueradingCopyCanBeDlopenedAsLibcudaSoOne) {
  const std::string dir = TESSERA_FAKE_MASQ_DIR;
  for (const char* soname : {"libcuda.so.1", "libcuda.so"}) {
    const std::string path = dir + "/" + soname;
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(nullptr, handle) << path << ": " << dlerror();
    EXPECT_NE(nullptr, dlsym(handle, "cuInit")) << soname;
    EXPECT_NE(nullptr, dlsym(handle, "cuGetProcAddress_v2")) << soname;
    EXPECT_NE(nullptr, dlsym(handle, "cuLaunchKernel_ptsz")) << soname;
    EXPECT_NE(nullptr, dlsym(handle, "tf_reset")) << soname;
    EXPECT_EQ(0, dlclose(handle));
  }
}

}  // namespace
}  // namespace tessera_test
