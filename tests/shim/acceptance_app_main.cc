// The application the acceptance matrix runs.
//
// It is an ordinary CUDA driver-API program: it includes <cuda.h> with no
// Tessera macros, links against libcuda.so.1, and never names libtessera at
// build time (I-2). Which library its calls land in is decided entirely by the
// environment its parent gives it, which is what makes one binary enough for
// every cell of the matrix -- including the control runs with no shim at all.
//
// It reaches the driver by the four access paths ADR-001 lists, one per run:
//
//   direct   DT_NEEDED libcuda.so.1, called through the PLT
//   dlopen   dlopen("libcuda.so.1") + dlsym(handle, name)
//   gpa1     cuGetProcAddress (the four-argument v1 entry point), which is
//            itself first resolved through cuGetProcAddress, as cudart does
//   gpa2     cuGetProcAddress_v2, likewise bootstrapped through itself
//
// EVERY driver call it makes is recorded under the EXPORTED symbol name that
// the call actually reaches -- `cuCtxCreate` is a cuda.h macro for
// `cuCtxCreate_v2`, and the recorded name is the expanded one, so the
// application's own bookkeeping cannot drift from the symbol the linker or
// cuGetProcAddress picked. Those counts are written to $TESSERA_APP_REPORT as
// JSON; libtessera writes $TESSERA_STATS_FILE and libcuda_fake writes
// $TESSERA_FAKE_STATS_FILE at exit. Comparing the three files is T1.
//
// Exit codes: 0 success, 1 a driver call failed unexpectedly, 2 usage,
// 3 a symbol could not be resolved, 4 the environment is not what the mode
// needs (for example a mode that needs the fake's control API cannot reach
// it).

#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#include <cuda.h>
#include <sys/wait.h>

// The v1 cuGetProcAddress lives behind a cuda.h macro that points at the
// five-argument _v2 entry point. An application that wants the old one -- and
// an old application is exactly what path gpa1 imitates -- has to say so.
#undef cuGetProcAddress
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion,
                                             cuuint64_t flags);

// Included for their declarations only. Neither library is linked: every
// tf_* and tessera_* function is reached through a pointer from dlsym, because
// in most cells of the matrix one or the other is not in the process at all.
#include "tessera/shim/stats.h"

#include "tessera_fake/control.h"

namespace {

// ---------------------------------------------------------------------------
// Heap-allocation counting (T7)
//
// Interposing malloc from the executable takes precedence over libc for the
// whole process, which is what makes the count a count of EVERY allocation in
// the measured window rather than only the ones this file makes. Forwarding
// goes to glibc's own __libc_* entry points rather than dlsym(RTLD_NEXT),
// which would allocate while we are answering an allocation.
//
// Not compiled under a sanitizer: ASan and TSan own malloc in an instrumented
// process, and a definition here would take the allocator away from the
// runtime that is also intercepting the frees. The T7 allocation check is
// skipped on those configurations, with the reason recorded, rather than
// measured wrongly.
// ---------------------------------------------------------------------------

struct AllocCounts {
  uint64_t mallocs = 0;
  uint64_t callocs = 0;
  uint64_t reallocs = 0;
  uint64_t frees = 0;
  uint64_t news = 0;
  uint64_t deletes = 0;

  uint64_t allocations() const { return mallocs + callocs + reallocs + news; }
  uint64_t total() const { return allocations() + frees + deletes; }
};

}  // namespace

#if !defined(TESSERA_SANITIZED)
namespace {
// Plain (non-atomic) counters: the T7 window is single-threaded by
// construction, and an atomic add here would be a lock-free write that the
// window is trying to prove it does not need.
volatile bool g_alloc_counting = false;
AllocCounts g_allocs;
}  // namespace

extern "C" {
void* __libc_malloc(size_t size);
void* __libc_calloc(size_t count, size_t size);
void* __libc_realloc(void* ptr, size_t size);
void __libc_free(void* ptr);

void* malloc(size_t size) {
  if (g_alloc_counting) {
    ++g_allocs.mallocs;
  }
  return __libc_malloc(size);
}

void* calloc(size_t count, size_t size) {
  if (g_alloc_counting) {
    ++g_allocs.callocs;
  }
  return __libc_calloc(count, size);
}

void* realloc(void* ptr, size_t size) {
  if (g_alloc_counting) {
    ++g_allocs.reallocs;
  }
  return __libc_realloc(ptr, size);
}

void free(void* ptr) {
  if (g_alloc_counting) {
    ++g_allocs.frees;
  }
  __libc_free(ptr);
}
}  // extern "C"

void* operator new(size_t size) {
  if (g_alloc_counting) {
    ++g_allocs.news;
  }
  void* p = __libc_malloc(size == 0 ? 1 : size);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}

void operator delete(void* p) noexcept {
  if (g_alloc_counting) {
    ++g_allocs.deletes;
  }
  __libc_free(p);
}

void operator delete(void* p, size_t) noexcept {
  operator delete(p);
}
#endif  // !TESSERA_SANITIZED

namespace {

bool alloc_counting_available() {
#if defined(TESSERA_SANITIZED)
  return false;
#else
  return true;
#endif
}

AllocCounts alloc_snapshot() {
#if defined(TESSERA_SANITIZED)
  return AllocCounts{};
#else
  return g_allocs;
#endif
}

void alloc_counting(bool on) {
#if !defined(TESSERA_SANITIZED)
  g_alloc_counting = on;
#else
  static_cast<void>(on);
#endif
}

AllocCounts alloc_delta(const AllocCounts& before, const AllocCounts& after) {
  AllocCounts d;
  d.mallocs = after.mallocs - before.mallocs;
  d.callocs = after.callocs - before.callocs;
  d.reallocs = after.reallocs - before.reallocs;
  d.frees = after.frees - before.frees;
  d.news = after.news - before.news;
  d.deletes = after.deletes - before.deletes;
  return d;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

int64_t now_us() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (static_cast<int64_t>(ts.tv_sec) * 1000000) + (ts.tv_nsec / 1000);
}

void fail(const char* what, CUresult r) {
  std::fprintf(stderr, "app: %s failed with %d\n", what, static_cast<int>(r));
}

const char* env_or_empty(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? value : "";
}

// A syscall that is impossible to confuse with anything else in an strace log
// and that changes nothing: a write to a closed descriptor. The measured
// window for T7 is the span between two of these.
void syscall_marker(const char* text) {
  const ssize_t ignored = ::write(-1, text, std::strlen(text));
  static_cast<void>(ignored);
}

// /proc/self/maps, read without allocating a container: T5 asks whether an
// object is mapped at all, which is a substring question.
//
// The path is resolved first. The kernel prints the real path of a mapped
// file, while the path we are given may reach it through a symlink -- the
// repository's build/ is a symlink onto tmpfs, so the two spellings differ and
// a substring test would answer "not mapped" for an object that is.
bool object_is_mapped(const char* path) {
  if (path == nullptr || path[0] == '\0') {
    return false;
  }
  char resolved[PATH_MAX];
  if (::realpath(path, resolved) != nullptr) {
    path = resolved;
  }
  const int fd = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return false;
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
  return text.find(path) != std::string::npos;
}

// ---------------------------------------------------------------------------
// What the application did, as it saw it
// ---------------------------------------------------------------------------

// Call counts by exported symbol name. A fixed array with a linear scan: the
// T7 window must not allocate, and this is called from inside it.
class Recorder {
 public:
  static constexpr int kMax = 96;

  void record(const char* name) {
    for (int i = 0; i < count_; ++i) {
      if (names_[i] == name || std::strcmp(names_[i], name) == 0) {
        ++counts_[i];
        return;
      }
    }
    if (count_ < kMax) {
      names_[count_] = name;
      counts_[count_] = 1;
      ++count_;
      return;
    }
    overflowed_ = true;
  }

  int size() const { return count_; }
  const char* name(int i) const { return names_[i]; }
  uint64_t count(int i) const { return counts_[i]; }
  bool overflowed() const { return overflowed_; }

 private:
  const char* names_[kMax] = {};
  uint64_t counts_[kMax] = {};
  int count_ = 0;
  bool overflowed_ = false;
};

// Every return code and out-parameter the application saw, in call order, as
// bytes. T4 compares this blob across all eight cells of the matrix and
// against the no-shim control: equal blobs mean the shim changed nothing that
// the application can observe (I-3 in miniature).
class Blob {
 public:
  void add(CUresult r) { add_u32(static_cast<uint32_t>(r)); }
  void add_u32(uint32_t v) { add_bytes(&v, sizeof(v)); }
  void add_u64(uint64_t v) { add_bytes(&v, sizeof(v)); }
  void add_bytes(const void* p, size_t n) {
    const auto* bytes = static_cast<const unsigned char*>(p);
    text_.append(reinterpret_cast<const char*>(bytes), n);
  }
  std::string hex() const {
    static const char* kDigits = "0123456789abcdef";
    std::string out;
    out.reserve(text_.size() * 2);
    for (const char c : text_) {
      const auto byte = static_cast<unsigned char>(c);
      out.push_back(kDigits[byte >> 4]);
      out.push_back(kDigits[byte & 0x0f]);
    }
    return out;
  }

 private:
  std::string text_;
};

// The report file, built as JSON in call order. Numeric objects come first so
// that a reader can stop at the first non-numeric member of each one.
class Report {
 public:
  void number(const std::string& key, uint64_t value) {
    line("\"" + key + "\":" + std::to_string(value));
  }
  void text(const std::string& key, const std::string& value) {
    line("\"" + key + "\":\"" + value + "\"");
  }
  void counts(const std::string& key, const Recorder& recorder) {
    std::string object = "\"" + key + "\":{";
    for (int i = 0; i < recorder.size(); ++i) {
      if (i != 0) {
        object += ",";
      }
      object += "\"" + std::string(recorder.name(i)) + "\":" + std::to_string(recorder.count(i));
    }
    line(object + "}");
  }
  void begin_object(const std::string& key) { line("\"" + key + "\":{"); }
  void object_number(const std::string& key, uint64_t value) {
    if (object_first_) {
      object_first_ = false;
    } else {
      body_ += ",";
    }
    body_ += "\"" + key + "\":" + std::to_string(value);
  }
  void end_object() {
    body_ += "}";
    object_first_ = true;
  }

  int write(const char* path) const {
    if (path == nullptr || path[0] == '\0') {
      return 0;
    }
    std::FILE* f = std::fopen(path, "we");
    if (f == nullptr) {
      return -1;
    }
    const int n = std::fprintf(f, "{%s}\n", body_.c_str());
    std::fclose(f);
    return n < 0 ? -1 : 0;
  }

 private:
  void line(const std::string& entry) {
    if (!body_.empty()) {
      body_ += ",";
    }
    body_ += entry;
  }

  std::string body_;
  bool object_first_ = true;
};

// ---------------------------------------------------------------------------
// The driver entry points this application uses
//
// exported: the name to call. It is a cuda.h macro wherever cuda.h has one, so
//           XSTR() below records the EXPANDED name -- the exported symbol the
//           call really reaches.
// base:     the name passed to cuGetProcAddress, which is never a macro.
// version:  the cudaVersion passed with it. CUDA_VERSION everywhere except
//           cuCtxCreate, where 10000 selects cuCtxCreate_v2 -- the same entry
//           point cuda.h's macro selects for the other three paths, so all
//           four paths exercise one identical set of exported symbols.
// ---------------------------------------------------------------------------

#define TESSERA_APP_STR(x) #x
#define TESSERA_APP_XSTR(x) TESSERA_APP_STR(x)

// clang-format off
#define TESSERA_APP_APIS(X)                                                           \
  X(kInit,              cuInit,              cuInit,              CUDA_VERSION)       \
  X(kDriverGetVersion,  cuDriverGetVersion,  cuDriverGetVersion,  CUDA_VERSION)       \
  X(kDeviceGetCount,    cuDeviceGetCount,    cuDeviceGetCount,    CUDA_VERSION)       \
  X(kDeviceGet,         cuDeviceGet,         cuDeviceGet,         CUDA_VERSION)       \
  X(kDeviceGetName,     cuDeviceGetName,     cuDeviceGetName,     CUDA_VERSION)       \
  X(kDeviceTotalMem,    cuDeviceTotalMem,    cuDeviceTotalMem,    CUDA_VERSION)       \
  X(kCtxCreate,         cuCtxCreate,         cuCtxCreate,         10000)              \
  X(kCtxDestroy,        cuCtxDestroy,        cuCtxDestroy,        CUDA_VERSION)       \
  X(kMemAlloc,          cuMemAlloc,          cuMemAlloc,          CUDA_VERSION)       \
  X(kMemFree,           cuMemFree,           cuMemFree,           CUDA_VERSION)       \
  X(kMemsetD8,          cuMemsetD8,          cuMemsetD8,          CUDA_VERSION)       \
  X(kMemcpyHtoD,        cuMemcpyHtoD,        cuMemcpyHtoD,        CUDA_VERSION)       \
  X(kMemcpyDtoHAsync,   cuMemcpyDtoHAsync,   cuMemcpyDtoHAsync,   CUDA_VERSION)       \
  X(kMemGetInfo,        cuMemGetInfo,        cuMemGetInfo,        CUDA_VERSION)       \
  X(kStreamCreate,      cuStreamCreate,      cuStreamCreate,      CUDA_VERSION)       \
  X(kStreamSynchronize, cuStreamSynchronize, cuStreamSynchronize, CUDA_VERSION)       \
  X(kStreamDestroy,     cuStreamDestroy,     cuStreamDestroy,     CUDA_VERSION)       \
  X(kModuleLoadData,    cuModuleLoadData,    cuModuleLoadData,    CUDA_VERSION)       \
  X(kModuleGetFunction, cuModuleGetFunction, cuModuleGetFunction, CUDA_VERSION)       \
  X(kModuleUnload,      cuModuleUnload,      cuModuleUnload,      CUDA_VERSION)       \
  X(kLaunchKernel,      cuLaunchKernel,      cuLaunchKernel,      CUDA_VERSION)       \
  X(kEventCreate,       cuEventCreate,       cuEventCreate,       CUDA_VERSION)       \
  X(kEventRecord,       cuEventRecord,       cuEventRecord,       CUDA_VERSION)       \
  X(kEventQuery,        cuEventQuery,        cuEventQuery,        CUDA_VERSION)       \
  X(kEventDestroy,      cuEventDestroy,      cuEventDestroy,      CUDA_VERSION)
// clang-format on

enum ApiId {
#define TESSERA_APP_ENUM(id, exported, base, version) id,
  TESSERA_APP_APIS(TESSERA_APP_ENUM)
#undef TESSERA_APP_ENUM
      kApiCount
};

struct ApiEntry {
  const char* exported;
  const char* base;
  int version;
};

const ApiEntry kApi[kApiCount] = {
#define TESSERA_APP_ROW(id, exported, base, version) {TESSERA_APP_XSTR(exported), #base, (version)},
    TESSERA_APP_APIS(TESSERA_APP_ROW)
#undef TESSERA_APP_ROW
};

// The signature of each entry point, taken from the declaration cuda.h itself
// selected. A wrong argument list is then a compile error rather than a
// corrupted call.
#define TESSERA_APP_TYPE(id, exported, base, version) using Fn_##id = decltype(&exported);
TESSERA_APP_APIS(TESSERA_APP_TYPE)
#undef TESSERA_APP_TYPE

using FnGetProcAddressV1 = CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t);
using FnGetProcAddressV2 = CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t,
                                              CUdriverProcAddressQueryResult*);

// One resolved entry point set, plus the bookkeeping the tests read back.
class Api {
 public:
  explicit Api(Recorder* recorder) : recorder_(recorder) {}

  template <typename T>
  T call(ApiId id) {
    recorder_->record(kApi[id].exported);
    T fn = nullptr;
    std::memcpy(&fn, &fn_[id], sizeof(fn));
    return fn;
  }

  void* raw(ApiId id) const { return fn_[id]; }
  void set(ApiId id, void* fn) { fn_[id] = fn; }
  void set_is_shim_export(ApiId id, bool v) { is_shim_export_[id] = v; }
  bool is_shim_export(ApiId id) const { return is_shim_export_[id]; }

  bool complete() const {
    for (int i = 0; i < kApiCount; ++i) {
      if (fn_[i] == nullptr) {
        std::fprintf(stderr, "app: could not resolve %s\n", kApi[i].exported);
        return false;
      }
    }
    return true;
  }

  Recorder* recorder() const { return recorder_; }

 private:
  void* fn_[kApiCount] = {};
  bool is_shim_export_[kApiCount] = {};
  Recorder* recorder_;
};

// dlsym(RTLD_DEFAULT, name) is the address the application would reach by
// direct linkage: the shim's exported symbol when a shim is in the process,
// the driver's own otherwise. Comparing a resolved pointer against it is how
// the application reports whether an access path landed on the shim without
// knowing anything about Tessera.
void* global_symbol(const char* name) {
  return dlsym(RTLD_DEFAULT, name);
}

// ---------------------------------------------------------------------------
// Resolving one access path
// ---------------------------------------------------------------------------

bool resolve_direct(Api& api) {
#define TESSERA_APP_DIRECT(id, exported, base, version) \
  api.set(id, reinterpret_cast<void*>(&exported));      \
  api.set_is_shim_export(id, true);
  TESSERA_APP_APIS(TESSERA_APP_DIRECT)
#undef TESSERA_APP_DIRECT
  return api.complete();
}

bool resolve_dlopen(Api& api) {
  // What Triton's launcher stubs do: open the driver by SONAME and look every
  // entry point up by name. Under masquerade this handle IS the shim; under
  // LD_PRELOAD it is the real driver and the shim's dlsym hook is what
  // redirects the lookups it recognises.
  void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    std::fprintf(stderr, "app: dlopen(libcuda.so.1): %s\n", dlerror());
    return false;
  }
  for (int i = 0; i < kApiCount; ++i) {
    const auto id = static_cast<ApiId>(i);
    void* fn = dlsym(handle, kApi[i].exported);
    api.set(id, fn);
    api.set_is_shim_export(id, fn != nullptr && fn == global_symbol(kApi[i].exported));
  }
  return api.complete();
}

bool resolve_get_proc_address(Api& api, bool v2, Report* report) {
  // cudart resolves the lookup function through the lookup function before it
  // resolves anything else. A shim that does not hook that sees nothing.
  uint64_t failures = 0;
  FnGetProcAddressV1 lookup_v1 = nullptr;
  FnGetProcAddressV2 lookup_v2 = nullptr;

  if (v2) {
    void* self = nullptr;
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    api.recorder()->record("cuGetProcAddress_v2");
    const CUresult r = cuGetProcAddress_v2("cuGetProcAddress", &self, CUDA_VERSION,
                                           CU_GET_PROC_ADDRESS_DEFAULT, &status);
    if (r != CUDA_SUCCESS || self == nullptr || status != CU_GET_PROC_ADDRESS_SUCCESS) {
      fail("cuGetProcAddress_v2(cuGetProcAddress)", r);
      return false;
    }
    report->number("bootstrap_is_shim_export",
                   self == global_symbol("cuGetProcAddress_v2") ? 1 : 0);
    std::memcpy(&lookup_v2, &self, sizeof(lookup_v2));
  } else {
    void* self = nullptr;
    api.recorder()->record("cuGetProcAddress");
    // Below 12000 the lookup itself is the v1 entry point, which is what an
    // application of that vintage is about to call through.
    const CUresult r =
        cuGetProcAddress("cuGetProcAddress", &self, 11030, CU_GET_PROC_ADDRESS_DEFAULT);
    if (r != CUDA_SUCCESS || self == nullptr) {
      fail("cuGetProcAddress(cuGetProcAddress)", r);
      return false;
    }
    report->number("bootstrap_is_shim_export", self == global_symbol("cuGetProcAddress") ? 1 : 0);
    std::memcpy(&lookup_v1, &self, sizeof(lookup_v1));
  }

  for (int i = 0; i < kApiCount; ++i) {
    const auto id = static_cast<ApiId>(i);
    void* fn = nullptr;
    CUresult r = CUDA_SUCCESS;
    if (v2) {
      CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
      api.recorder()->record("cuGetProcAddress_v2");
      r = lookup_v2(kApi[i].base, &fn, kApi[i].version, CU_GET_PROC_ADDRESS_DEFAULT, &status);
      if (status != CU_GET_PROC_ADDRESS_SUCCESS) {
        ++failures;
      }
    } else {
      api.recorder()->record("cuGetProcAddress");
      r = lookup_v1(kApi[i].base, &fn, kApi[i].version, CU_GET_PROC_ADDRESS_DEFAULT);
    }
    if (r != CUDA_SUCCESS) {
      fail(kApi[i].base, r);
      ++failures;
    }
    api.set(id, fn);
    api.set_is_shim_export(id, fn != nullptr && fn == global_symbol(kApi[i].exported));
  }
  report->number("lookup_failures", failures);
  return failures == 0 && api.complete();
}

// ---------------------------------------------------------------------------
// The fake driver's control API, when it is reachable
//
// Under masquerade the fake is opened RTLD_LOCAL by the shim, so it is not in
// the application's global scope; opening the same absolute path again returns
// the same object, and its tf_* functions with it. Under LD_PRELOAD the fake
// is already the application's libcuda.so.1 and the same dlopen finds it. In
// both cases this is one simulation, seen from both sides.
// ---------------------------------------------------------------------------

struct FakeControl {
  void* handle = nullptr;
  decltype(&tf_symbol_count) symbol_count = nullptr;
  decltype(&tf_symbol_table_size) table_size = nullptr;
  decltype(&tf_symbol_name_at) name_at = nullptr;
  decltype(&tf_set_result) set_result = nullptr;
  decltype(&tf_clear_results) clear_results = nullptr;
  decltype(&tf_last_stream_was_per_thread_default) last_stream_per_thread = nullptr;

  bool ok() const { return symbol_count != nullptr && table_size != nullptr; }
};

template <typename T>
void bind(void* handle, const char* name, T* out) {
  void* p = dlsym(handle, name);
  std::memcpy(out, &p, sizeof(p));
}

FakeControl open_fake_control() {
  FakeControl fake;
  const char* path = std::getenv("TESSERA_REAL_LIBCUDA");
  if (path != nullptr && path[0] != '\0') {
    // Loading a library named by the environment is the mechanism under test,
    // not an oversight: the test harness sets this variable and the same
    // pattern is what the shim itself does (shim/src/state.cc carries the same
    // suppression for the same reason).
    // NOLINTNEXTLINE(clang-analyzer-optin.taint.GenericTaint)
    fake.handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  }
  if (fake.handle == nullptr) {
    fake.handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  }
  if (fake.handle == nullptr) {
    return fake;
  }
  bind(fake.handle, "tf_symbol_count", &fake.symbol_count);
  bind(fake.handle, "tf_symbol_table_size", &fake.table_size);
  bind(fake.handle, "tf_symbol_name_at", &fake.name_at);
  bind(fake.handle, "tf_set_result", &fake.set_result);
  bind(fake.handle, "tf_clear_results", &fake.clear_results);
  bind(fake.handle, "tf_last_stream_was_per_thread_default", &fake.last_stream_per_thread);
  return fake;
}

// The shim's observability ABI, when a shim is in the process at all. Reached
// through RTLD_DEFAULT: the masquerading libcuda.so.1 and the preloaded
// libtessera.so are both in the global scope, and in a control run neither is,
// which is exactly what the report should then say.
struct ShimControl {
  decltype(&tessera_get_symbol_count) symbol_count = nullptr;
  decltype(&tessera_get_stats_v1) get_stats = nullptr;
  decltype(&tessera_dump_stats) dump_stats = nullptr;

  bool ok() const { return symbol_count != nullptr && get_stats != nullptr; }
};

ShimControl open_shim_control() {
  ShimControl shim;
  bind(RTLD_DEFAULT, "tessera_get_symbol_count", &shim.symbol_count);
  bind(RTLD_DEFAULT, "tessera_get_stats_v1", &shim.get_stats);
  bind(RTLD_DEFAULT, "tessera_dump_stats", &shim.dump_stats);
  return shim;
}

// The shim's and the fake's view of every symbol the application recorded, so
// that a child process which cannot write the two stats files (a forked child
// writes neither) still reports all three numbers.
void add_side_counters(Report* report, const Recorder& recorder, const ShimControl& shim,
                       const FakeControl& fake) {
  report->number("shim_present", shim.ok() ? 1 : 0);
  report->number("fake_control_present", fake.ok() ? 1 : 0);
  if (shim.ok()) {
    report->begin_object("shim_symbols");
    for (int i = 0; i < recorder.size(); ++i) {
      report->object_number(recorder.name(i), shim.symbol_count(recorder.name(i)));
    }
    report->end_object();
  }
  if (fake.ok()) {
    report->begin_object("fake_symbols");
    for (int i = 0; i < recorder.size(); ++i) {
      report->object_number(recorder.name(i), fake.symbol_count(recorder.name(i)));
    }
    report->end_object();
  }
}

// ---------------------------------------------------------------------------
// The workload: identical for every access path
// ---------------------------------------------------------------------------

constexpr int kWorkloadLaunches = 3;
constexpr size_t kBufferBytes = 256;

int run_workload(Api& api, Blob* blob, const FakeControl& fake) {
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CUstream stream = nullptr;
  CUevent event = nullptr;
  CUdeviceptr buffer = 0;
  int driver_version = 0;
  int device_count = 0;
  char device_name[64];
  std::memset(device_name, 0, sizeof(device_name));
  size_t total_mem = 0;
  size_t free_mem = 0;
  size_t total_mem_again = 0;
  unsigned char host_in[kBufferBytes];
  unsigned char host_out[kBufferBytes];
  for (size_t i = 0; i < kBufferBytes; ++i) {
    host_in[i] = static_cast<unsigned char>((i * 7) + 1);
    host_out[i] = 0;
  }

  CUresult r = api.call<Fn_kInit>(kInit)(0);
  blob->add(r);
  if (r != CUDA_SUCCESS) {
    fail("cuInit", r);
    return 1;
  }
  blob->add(api.call<Fn_kDriverGetVersion>(kDriverGetVersion)(&driver_version));
  blob->add_u32(static_cast<uint32_t>(driver_version));
  blob->add(api.call<Fn_kDeviceGetCount>(kDeviceGetCount)(&device_count));
  blob->add_u32(static_cast<uint32_t>(device_count));
  blob->add(api.call<Fn_kDeviceGet>(kDeviceGet)(&device, 0));
  blob->add(api.call<Fn_kDeviceGetName>(kDeviceGetName)(device_name, sizeof(device_name), device));
  blob->add_bytes(device_name, sizeof(device_name));
  blob->add(api.call<Fn_kDeviceTotalMem>(kDeviceTotalMem)(&total_mem, device));
  blob->add_u64(total_mem);

  r = api.call<Fn_kCtxCreate>(kCtxCreate)(&context, 0, device);
  blob->add(r);
  if (r != CUDA_SUCCESS) {
    fail("cuCtxCreate", r);
    return 1;
  }

  blob->add(api.call<Fn_kMemAlloc>(kMemAlloc)(&buffer, kBufferBytes));
  blob->add(api.call<Fn_kMemsetD8>(kMemsetD8)(buffer, 0xA5, kBufferBytes));
  blob->add(api.call<Fn_kMemcpyHtoD>(kMemcpyHtoD)(buffer, host_in, kBufferBytes));
  blob->add(api.call<Fn_kStreamCreate>(kStreamCreate)(&stream, 0));
  blob->add(api.call<Fn_kModuleLoadData>(kModuleLoadData)(&module, "acceptance-module"));
  blob->add(
      api.call<Fn_kModuleGetFunction>(kModuleGetFunction)(&function, module, "acceptance_kernel"));

  for (int i = 0; i < kWorkloadLaunches; ++i) {
    blob->add(api.call<Fn_kLaunchKernel>(kLaunchKernel)(function, 1, 1, 1, 1, 1, 1, 0, stream,
                                                        nullptr, nullptr));
  }

  blob->add(
      api.call<Fn_kMemcpyDtoHAsync>(kMemcpyDtoHAsync)(host_out, buffer, kBufferBytes, stream));
  blob->add(api.call<Fn_kEventCreate>(kEventCreate)(&event, CU_EVENT_DEFAULT));
  blob->add(api.call<Fn_kEventRecord>(kEventRecord)(event, stream));
  blob->add(api.call<Fn_kStreamSynchronize>(kStreamSynchronize)(stream));
  blob->add(api.call<Fn_kEventQuery>(kEventQuery)(event));
  // The bytes the device gave back: bitwise pass-through of an out-parameter
  // that went device-ward and came back (I-3).
  blob->add_bytes(host_out, kBufferBytes);

  blob->add(api.call<Fn_kMemGetInfo>(kMemGetInfo)(&free_mem, &total_mem_again));
  blob->add_u64(total_mem_again);
  blob->add_u64(total_mem_again - free_mem);  // outstanding bytes: position-independent

  // ---- injected errors --------------------------------------------------
  // T4 asks for return codes and out-parameters to pass through bitwise,
  // including a failure. The fake returns the injected code after counting the
  // call and before touching anything, so an out-parameter that changed would
  // mean the shim wrote to it.
  if (fake.ok() && fake.set_result != nullptr) {
    constexpr int kSentinelInt = 0x5eed1234;
    fake.set_result("cuDriverGetVersion", CUDA_ERROR_INVALID_VALUE);
    fake.set_result(kApi[kMemAlloc].exported, CUDA_ERROR_OUT_OF_MEMORY);

    int sentinel = kSentinelInt;
    blob->add(api.call<Fn_kDriverGetVersion>(kDriverGetVersion)(&sentinel));
    blob->add_u32(static_cast<uint32_t>(sentinel));

    CUdeviceptr refused = 0xdeadbeef;
    blob->add(api.call<Fn_kMemAlloc>(kMemAlloc)(&refused, kBufferBytes));
    blob->add_u64(refused);

    fake.clear_results();
  }

  blob->add(api.call<Fn_kEventDestroy>(kEventDestroy)(event));
  blob->add(api.call<Fn_kStreamDestroy>(kStreamDestroy)(stream));
  blob->add(api.call<Fn_kMemFree>(kMemFree)(buffer));
  blob->add(api.call<Fn_kModuleUnload>(kModuleUnload)(module));
  blob->add(api.call<Fn_kCtxDestroy>(kCtxDestroy)(context));
  return 0;
}

int mode_workload(const char* path, Report* report) {
  Recorder recorder;
  Api api(&recorder);
  report->text("mode", "workload");
  report->text("path", path);

  bool resolved = false;
  if (std::strcmp(path, "direct") == 0) {
    resolved = resolve_direct(api);
  } else if (std::strcmp(path, "dlopen") == 0) {
    resolved = resolve_dlopen(api);
  } else if (std::strcmp(path, "gpa1") == 0) {
    resolved = resolve_get_proc_address(api, false, report);
  } else if (std::strcmp(path, "gpa2") == 0) {
    resolved = resolve_get_proc_address(api, true, report);
  } else {
    std::fprintf(stderr, "app: unknown access path '%s'\n", path);
    return 2;
  }
  if (!resolved) {
    return 3;
  }

  const FakeControl fake = open_fake_control();
  Blob blob;
  const int rc = run_workload(api, &blob, fake);

  report->number("workload_ok", rc == 0 ? 1 : 0);
  report->number("recorder_overflowed", recorder.overflowed() ? 1 : 0);
  report->counts("issued", recorder);
  report->begin_object("resolved_is_shim_export");
  for (int i = 0; i < kApiCount; ++i) {
    report->object_number(kApi[i].exported, api.is_shim_export(static_cast<ApiId>(i)) ? 1 : 0);
  }
  report->end_object();
  add_side_counters(report, recorder, open_shim_control(), fake);
  report->text("outparams", blob.hex());
  return rc;
}

// ---------------------------------------------------------------------------
// T2: the per-thread default stream, and version selection
// ---------------------------------------------------------------------------

struct PtszCase {
  const char* label;
  const char* base;
  int version;
  cuuint64_t flags;
  const char* expected_exported;
};

int mode_ptsz(Report* report) {
  report->text("mode", "ptsz");
  const FakeControl fake = open_fake_control();
  const ShimControl shim = open_shim_control();
  if (!fake.ok() || fake.last_stream_per_thread == nullptr) {
    std::fprintf(stderr, "app: the fake's control API is required for the ptsz mode\n");
    return 4;
  }

  Recorder recorder;
  Api api(&recorder);
  if (!resolve_direct(api)) {
    return 3;
  }
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  if (api.call<Fn_kInit>(kInit)(0) != CUDA_SUCCESS ||
      api.call<Fn_kDeviceGet>(kDeviceGet)(&device, 0) != CUDA_SUCCESS ||
      api.call<Fn_kCtxCreate>(kCtxCreate)(&context, 0, device) != CUDA_SUCCESS ||
      api.call<Fn_kModuleLoadData>(kModuleLoadData)(&module, "ptsz-module") != CUDA_SUCCESS ||
      api.call<Fn_kModuleGetFunction>(kModuleGetFunction)(&function, module, "ptsz_kernel") !=
          CUDA_SUCCESS) {
    std::fprintf(stderr, "app: ptsz setup failed\n");
    return 1;
  }

  // Stream-flag selection. The NULL stream argument is what makes the choice
  // observable: a _ptsz entry point resolves it to the calling thread's
  // per-thread default stream, its legacy twin to the context's.
  const PtszCase kStreamCases[] = {
      {"per_thread", "cuLaunchKernel", CUDA_VERSION, CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
       "cuLaunchKernel_ptsz"},
      {"legacy", "cuLaunchKernel", CUDA_VERSION, CU_GET_PROC_ADDRESS_LEGACY_STREAM,
       "cuLaunchKernel"},
      {"default_flag", "cuLaunchKernel", CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT,
       "cuLaunchKernel"},
  };

  report->begin_object("stream_case_ok");
  uint64_t stream_failures = 0;
  for (const PtszCase& c : kStreamCases) {
    void* resolved = nullptr;
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    recorder.record("cuGetProcAddress_v2");
    const CUresult r = cuGetProcAddress_v2(c.base, &resolved, c.version, c.flags, &status);
    const uint64_t shim_before = shim.ok() ? shim.symbol_count(c.expected_exported) : 0;
    const uint64_t fake_before = fake.symbol_count(c.expected_exported);

    bool ok = r == CUDA_SUCCESS && resolved != nullptr && status == CU_GET_PROC_ADDRESS_SUCCESS;
    ok = ok && resolved == global_symbol(c.expected_exported);
    if (ok) {
      Fn_kLaunchKernel launch = nullptr;
      std::memcpy(&launch, &resolved, sizeof(launch));
      recorder.record(c.expected_exported);
      // A NULL stream: which default stream it becomes is the proof.
      ok = launch(function, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr) == CUDA_SUCCESS;
      const bool want_per_thread = std::strcmp(c.label, "per_thread") == 0;
      ok = ok && (fake.last_stream_per_thread() != 0) == want_per_thread;
      ok = ok && fake.symbol_count(c.expected_exported) == fake_before + 1;
      if (shim.ok()) {
        ok = ok && shim.symbol_count(c.expected_exported) == shim_before + 1;
      }
    }
    if (!ok) {
      ++stream_failures;
    }
    report->object_number(c.label, ok ? 1 : 0);
  }
  report->end_object();

  // Version selection. Each case names the entry point the driver must choose
  // for that cudaVersion; the shim has to hand back its wrapper for that exact
  // variant, which is what makes the ABI the caller is about to use the right
  // one.
  const PtszCase kVersionCases[] = {
      {"cuMemAlloc@3000", "cuMemAlloc", 3000, CU_GET_PROC_ADDRESS_DEFAULT, "cuMemAlloc"},
      {"cuMemAlloc@12060", "cuMemAlloc", CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT,
       "cuMemAlloc_v2"},
      {"cuCtxCreate@3020", "cuCtxCreate", 3020, CU_GET_PROC_ADDRESS_DEFAULT, "cuCtxCreate_v2"},
      {"cuCtxCreate@11040", "cuCtxCreate", 11040, CU_GET_PROC_ADDRESS_DEFAULT, "cuCtxCreate_v3"},
      {"cuCtxCreate@12050", "cuCtxCreate", 12050, CU_GET_PROC_ADDRESS_DEFAULT, "cuCtxCreate_v4"},
  };

  report->begin_object("version_case_ok");
  uint64_t version_failures = 0;
  for (const PtszCase& c : kVersionCases) {
    void* resolved = nullptr;
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    recorder.record("cuGetProcAddress_v2");
    const CUresult r = cuGetProcAddress_v2(c.base, &resolved, c.version, c.flags, &status);
    bool ok = r == CUDA_SUCCESS && resolved != nullptr && status == CU_GET_PROC_ADDRESS_SUCCESS;
    ok = ok && resolved == global_symbol(c.expected_exported);

    // Calling the variant proves the pointer has the ABI its name promises:
    // the v1 entry points take 32-bit device pointers and a different context
    // argument list, so a wrong choice would be a corrupted call rather than a
    // wrong counter.
    const uint64_t fake_before = fake.symbol_count(c.expected_exported);
    if (ok) {
      recorder.record(c.expected_exported);
      if (std::strcmp(c.expected_exported, "cuMemAlloc") == 0) {
        using FnV1 = CUresult(CUDAAPI*)(unsigned int*, unsigned int);
        FnV1 fn = nullptr;
        std::memcpy(&fn, &resolved, sizeof(fn));
        unsigned int small = 0;
        ok = fn(&small, 128) == CUDA_SUCCESS && small != 0;
      } else if (std::strcmp(c.expected_exported, "cuMemAlloc_v2") == 0) {
        using FnV2 = CUresult(CUDAAPI*)(CUdeviceptr*, size_t);
        FnV2 fn = nullptr;
        std::memcpy(&fn, &resolved, sizeof(fn));
        CUdeviceptr p = 0;
        ok = fn(&p, 128) == CUDA_SUCCESS && p != 0;
      } else if (std::strcmp(c.expected_exported, "cuCtxCreate_v2") == 0) {
        Fn_kCtxCreate fn = nullptr;
        std::memcpy(&fn, &resolved, sizeof(fn));
        CUcontext ctx = nullptr;
        ok = fn(&ctx, 0, device) == CUDA_SUCCESS && ctx != nullptr;
      } else if (std::strcmp(c.expected_exported, "cuCtxCreate_v3") == 0) {
        using FnV3 =
            CUresult(CUDAAPI*)(CUcontext*, CUexecAffinityParam*, int, unsigned int, CUdevice);
        FnV3 fn = nullptr;
        std::memcpy(&fn, &resolved, sizeof(fn));
        CUcontext ctx = nullptr;
        ok = fn(&ctx, nullptr, 0, 0, device) == CUDA_SUCCESS && ctx != nullptr;
      } else if (std::strcmp(c.expected_exported, "cuCtxCreate_v4") == 0) {
        using FnV4 = CUresult(CUDAAPI*)(CUcontext*, CUctxCreateParams*, unsigned int, CUdevice);
        FnV4 fn = nullptr;
        std::memcpy(&fn, &resolved, sizeof(fn));
        CUcontext ctx = nullptr;
        ok = fn(&ctx, nullptr, 0, device) == CUDA_SUCCESS && ctx != nullptr;
      }
      ok = ok && fake.symbol_count(c.expected_exported) == fake_before + 1;
      if (shim.ok()) {
        ok = ok && shim.symbol_count(c.expected_exported) >= 1;
      }
    }
    if (!ok) {
      ++version_failures;
    }
    report->object_number(c.label, ok ? 1 : 0);
  }
  report->end_object();

  report->number("stream_failures", stream_failures);
  report->number("version_failures", version_failures);
  report->counts("issued", recorder);
  add_side_counters(report, recorder, shim, fake);
  return (stream_failures == 0 && version_failures == 0) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// T3: fork with a shim-internal lock held
//
// The shim has exactly one lock, and it is held only while the shim is loading
// the real driver (shim/src/state.cc). So the scenario has to be built around
// that window: a worker thread makes the process's first driver call, and this
// file's own dlopen -- which the shim's call to dlopen binds to, because the
// executable comes first in the global scope -- announces that the shim is
// inside its critical section and then stalls there. The main thread forks
// during the stall.
//
// Two things are then measured rather than assumed:
//   * how long fork() blocked. The shim's pthread_atfork prepare handler takes
//     that same lock, so a fork issued during the stall cannot return until
//     the initialising thread lets go. A fork that returns immediately means
//     no handler ran and the child would have inherited a locked mutex.
//   * how long the child took to complete a hooked call. Without the handlers
//     it would never complete one.
//
// The driver is warmed up first, on purpose: libcuda_fake builds its
// simulation behind a function-local static and registers its own fork
// handlers there, so a fork landing inside THAT window deadlocks the child on
// the driver's guard rather than on anything of Tessera's. That is a property
// of the driver underneath, not of the shim, and this test is about the shim's
// lock.
// ---------------------------------------------------------------------------

struct ForkScenario {
  int ready_pipe[2] = {-1, -1};
  const char* driver_path = nullptr;
  int stall_ms = 300;
  volatile bool armed = false;
  volatile bool stalled = false;
};

ForkScenario g_fork;

void* fork_worker(void* arg) {
  auto* api = static_cast<Api*>(arg);
  api->call<Fn_kInit>(kInit)(0);
  return nullptr;
}

int mode_fork(Report* report) {
  report->text("mode", "fork");
  const char* driver_path = std::getenv("TESSERA_REAL_LIBCUDA");
  if (driver_path == nullptr || driver_path[0] == '\0') {
    std::fprintf(stderr, "app: the fork mode needs TESSERA_REAL_LIBCUDA\n");
    return 4;
  }
  g_fork.driver_path = driver_path;
  if (::pipe(g_fork.ready_pipe) != 0) {
    return 4;
  }

  // Warm the driver (see above), through its own handle rather than through
  // any cu* call, so the shim is still uninitialised afterwards.
  const FakeControl fake = open_fake_control();
  report->number("driver_warmed", fake.ok() ? 1 : 0);
  if (fake.ok()) {
    static_cast<void>(fake.table_size());
  }

  Recorder recorder;
  Api api(&recorder);
  if (!resolve_direct(api)) {
    return 3;
  }

  g_fork.armed = true;
  pthread_t worker{};
  if (pthread_create(&worker, nullptr, fork_worker, &api) != 0) {
    return 4;
  }
  char byte = 0;
  if (::read(g_fork.ready_pipe[0], &byte, 1) != 1) {
    std::fprintf(stderr, "app: the shim never opened the driver\n");
    return 4;
  }

  const int64_t before_fork = now_us();
  const pid_t pid = ::fork();
  if (pid == 0) {
    // The child. Its first act is to look at the counters it inherited: the
    // shim resets them in its fork handler, so they must be zero here.
    const ShimControl child_shim = open_shim_control();
    tessera_stats_v1 inherited;
    std::memset(&inherited, 0, sizeof(inherited));
    if (child_shim.ok()) {
      child_shim.get_stats(&inherited);
    }
    uint64_t inherited_fake_calls = 0;
    if (fake.ok()) {
      for (uint64_t i = 0; i < fake.table_size(); ++i) {
        inherited_fake_calls += fake.symbol_count(fake.name_at(i));
      }
    }
    const int64_t child_start = now_us();
    Recorder child_recorder;
    Api child_api(&child_recorder);
    resolve_direct(child_api);
    const CUresult init = child_api.call<Fn_kInit>(kInit)(0);
    CUdevice device = 0;
    const CUresult got = child_api.call<Fn_kDeviceGet>(kDeviceGet)(&device, 0);
    CUcontext context = nullptr;
    const CUresult created = child_api.call<Fn_kCtxCreate>(kCtxCreate)(&context, 0, device);
    size_t total = 0;
    const CUresult hooked = child_api.call<Fn_kDeviceTotalMem>(kDeviceTotalMem)(&total, device);
    const int64_t child_us = now_us() - child_start;

    Report child_report;
    child_report.text("mode", "fork_child");
    child_report.number("inherited_hooked_calls", inherited.hooked_calls);
    child_report.number("inherited_fake_calls", inherited_fake_calls);
    child_report.number("inherited_launches", inherited.launches);
    child_report.number("hooked_call_us", static_cast<uint64_t>(child_us));
    child_report.number("init_ok", init == CUDA_SUCCESS ? 1 : 0);
    child_report.number("device_ok", got == CUDA_SUCCESS ? 1 : 0);
    child_report.number("ctx_ok", created == CUDA_SUCCESS ? 1 : 0);
    child_report.number("hooked_ok", hooked == CUDA_SUCCESS ? 1 : 0);
    child_report.number("total_mem_nonzero", total != 0 ? 1 : 0);
    child_report.counts("issued", child_recorder);
    add_side_counters(&child_report, child_recorder, child_shim, fake);
    const int written = child_report.write(env_or_empty("TESSERA_APP_CHILD_REPORT"));
    // _exit: the child must run neither the shim's nor the fake's exit dump,
    // and gtest is not in this process at all.
    ::_exit(written == 0 && init == CUDA_SUCCESS && hooked == CUDA_SUCCESS ? 0 : 1);
  }
  const int64_t fork_us = now_us() - before_fork;
  if (pid < 0) {
    return 4;
  }

  int status = 0;
  const int64_t wait_start = now_us();
  int child_exit = -1;
  bool child_exited = false;
  bool child_timed_out = false;
  for (;;) {
    const pid_t reaped = ::waitpid(pid, &status, WNOHANG);
    if (reaped == pid) {
      child_exited = WIFEXITED(status) != 0;
      child_exit = child_exited ? WEXITSTATUS(status) : 0;
      break;
    }
    if (reaped < 0) {
      break;
    }
    if (now_us() - wait_start > 2000000) {  // T3's two seconds
      child_timed_out = true;
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &status, 0);
      break;
    }
    const timespec pause{0, 200000};
    ::nanosleep(&pause, nullptr);
  }
  const int64_t child_wall_us = now_us() - wait_start;
  pthread_join(worker, nullptr);

  // The parent's counters, read after the child is gone: the child's calls
  // must not be in them.
  const ShimControl shim = open_shim_control();
  tessera_stats_v1 parent;
  std::memset(&parent, 0, sizeof(parent));
  if (shim.ok()) {
    shim.get_stats(&parent);
  }

  report->number("stall_ms", static_cast<uint64_t>(g_fork.stall_ms));
  report->number("stalled_in_shim_dlopen", g_fork.stalled ? 1 : 0);
  report->number("fork_blocked_us", static_cast<uint64_t>(fork_us));
  report->number("child_wall_us", static_cast<uint64_t>(child_wall_us));
  report->number("child_timed_out", child_timed_out ? 1 : 0);
  report->number("child_exited", child_exited ? 1 : 0);
  report->number("child_exit_code", static_cast<uint64_t>(child_exit < 0 ? 0 : child_exit));
  report->number("parent_hooked_calls", parent.hooked_calls);
  report->number("parent_launches", parent.launches);
  report->counts("issued", recorder);
  add_side_counters(report, recorder, shim, fake);
  return (child_exited && child_exit == 0 && !child_timed_out) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// T5: lazy initialisation
// ---------------------------------------------------------------------------

int mode_lazyinit(Report* report, bool call_the_driver_first) {
  report->text("mode", "lazyinit");
  report->number("called_the_driver_first", call_the_driver_first ? 1 : 0);
  const char* driver_path = env_or_empty("TESSERA_REAL_LIBCUDA");

  // The instrument's positive control. Every observation below is a claim that
  // something has NOT happened yet, so a run that makes it happen on purpose
  // is what shows the observations can come out the other way.
  Recorder warmup_recorder;
  Api warmup_api(&warmup_recorder);
  if (call_the_driver_first) {
    if (!resolve_direct(warmup_api)) {
      return 3;
    }
    warmup_api.call<Fn_kInit>(kInit)(0);
  }

  // Before the first driver call. Under masquerade the shim is the
  // application's libcuda.so.1 and has been loaded by the dynamic linker
  // already, so if it had initialised itself from a constructor the driver
  // would be mapped by now. Under LD_PRELOAD the driver is the application's
  // own libcuda.so.1 and is mapped either way, so there the measurement that
  // matters is the driver's own call counters, which are reachable because the
  // fake is in the global scope.
  const bool driver_mapped_before = object_is_mapped(driver_path);
  ShimControl shim = open_shim_control();
  tessera_stats_v1 before;
  std::memset(&before, 0, sizeof(before));
  if (shim.ok()) {
    shim.get_stats(&before);
  }

  uint64_t fake_calls_before = 0;
  bool fake_counts_before_available = false;
  decltype(&tf_symbol_count) global_symbol_count = nullptr;
  decltype(&tf_symbol_table_size) global_table_size = nullptr;
  decltype(&tf_symbol_name_at) global_name_at = nullptr;
  bind(RTLD_DEFAULT, "tf_symbol_count", &global_symbol_count);
  bind(RTLD_DEFAULT, "tf_symbol_table_size", &global_table_size);
  bind(RTLD_DEFAULT, "tf_symbol_name_at", &global_name_at);
  if (global_symbol_count != nullptr && global_table_size != nullptr && global_name_at != nullptr) {
    fake_counts_before_available = true;
    for (uint64_t i = 0; i < global_table_size(); ++i) {
      fake_calls_before += global_symbol_count(global_name_at(i));
    }
  }

  // The application's first driver call.
  Recorder recorder;
  Api api(&recorder);
  if (!resolve_direct(api)) {
    return 3;
  }
  const CUresult r = api.call<Fn_kInit>(kInit)(0);
  const bool driver_mapped_after = object_is_mapped(driver_path);

  uint64_t fake_calls_after = 0;
  if (fake_counts_before_available) {
    for (uint64_t i = 0; i < global_table_size(); ++i) {
      fake_calls_after += global_symbol_count(global_name_at(i));
    }
  }

  shim = open_shim_control();
  tessera_stats_v1 after;
  std::memset(&after, 0, sizeof(after));
  if (shim.ok()) {
    shim.get_stats(&after);
  }

  report->number("init_ok", r == CUDA_SUCCESS ? 1 : 0);
  report->number("driver_mapped_before_first_call", driver_mapped_before ? 1 : 0);
  report->number("driver_mapped_after_first_call", driver_mapped_after ? 1 : 0);
  report->number("fake_counts_available", fake_counts_before_available ? 1 : 0);
  report->number("fake_calls_before_first_call", fake_calls_before);
  report->number("fake_calls_after_first_call", fake_calls_after);
  report->number("shim_hooked_calls_before", before.hooked_calls);
  report->number("shim_calls_before", before.hooked_calls + before.forwarded_unhooked);
  report->number("shim_calls_after", after.hooked_calls + after.forwarded_unhooked);
  report->counts("issued", recorder);
  add_side_counters(report, recorder, open_shim_control(), FakeControl{});
  return r == CUDA_SUCCESS ? 0 : 1;
}

// ---------------------------------------------------------------------------
// T6: a misconfigured TESSERA_REAL_LIBCUDA
// ---------------------------------------------------------------------------

int mode_selfref(Report* report) {
  report->text("mode", "selfref");
  Recorder recorder;
  Api api(&recorder);
  if (!resolve_direct(api)) {
    return 3;
  }
  // Every one of these must come back with an error rather than recursing,
  // hanging or aborting (I-5). The parent's timeout is what catches a hang.
  const CUresult init = api.call<Fn_kInit>(kInit)(0);
  int version = 0;
  const CUresult unhooked = api.call<Fn_kDriverGetVersion>(kDriverGetVersion)(&version);
  CUdevice device = 0;
  size_t total = 0;
  const CUresult hooked = api.call<Fn_kDeviceTotalMem>(kDeviceTotalMem)(&total, device);

  report->number("init_result", static_cast<uint64_t>(init));
  report->number("unhooked_result", static_cast<uint64_t>(unhooked));
  report->number("hooked_result", static_cast<uint64_t>(hooked));
  report->number("init_failed", init != CUDA_SUCCESS ? 1 : 0);
  report->number("unhooked_failed", unhooked != CUDA_SUCCESS ? 1 : 0);
  report->number("hooked_failed", hooked != CUDA_SUCCESS ? 1 : 0);
  report->counts("issued", recorder);
  return (init != CUDA_SUCCESS && unhooked != CUDA_SUCCESS && hooked != CUDA_SUCCESS) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// T7: the hot path (I-4)
// ---------------------------------------------------------------------------

int mode_hotpath(Report* report, int launches) {
  report->text("mode", "hotpath");
  Recorder recorder;
  Api api(&recorder);
  if (!resolve_direct(api)) {
    return 3;
  }
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CUstream stream = nullptr;
  if (api.call<Fn_kInit>(kInit)(0) != CUDA_SUCCESS ||
      api.call<Fn_kDeviceGet>(kDeviceGet)(&device, 0) != CUDA_SUCCESS ||
      api.call<Fn_kCtxCreate>(kCtxCreate)(&context, 0, device) != CUDA_SUCCESS ||
      api.call<Fn_kModuleLoadData>(kModuleLoadData)(&module, "hotpath-module") != CUDA_SUCCESS ||
      api.call<Fn_kModuleGetFunction>(kModuleGetFunction)(&function, module, "hotpath_kernel") !=
          CUDA_SUCCESS ||
      api.call<Fn_kStreamCreate>(kStreamCreate)(&stream, 0) != CUDA_SUCCESS) {
    std::fprintf(stderr, "app: hotpath setup failed\n");
    return 1;
  }

  Fn_kLaunchKernel launch = nullptr;
  void* raw = api.raw(kLaunchKernel);
  std::memcpy(&launch, &raw, sizeof(launch));

  // Steady state: the shim's slot for this symbol is resolved, its counters
  // are warm, and the stack and the code are paged in.
  constexpr int kWarmup = 64;
  for (int i = 0; i < kWarmup; ++i) {
    launch(function, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr);
  }
  const ShimControl shim = open_shim_control();
  const uint64_t shim_launches_before = shim.ok() ? shim.symbol_count("cuLaunchKernel") : 0;

  // ---- the measured window ----------------------------------------------
  alloc_counting(true);
  const AllocCounts allocs_before = alloc_snapshot();
  syscall_marker("tessera-window-begin");
  uint64_t failures = 0;
  for (int i = 0; i < launches; ++i) {
    failures += launch(function, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr) != CUDA_SUCCESS;
  }
  syscall_marker("tessera-window-end");
  const AllocCounts allocs_after = alloc_snapshot();

  // ---- the instrument's own control -------------------------------------
  // One deliberate allocation and one deliberate syscall, so that a run which
  // reports zero for the window above has also just demonstrated that it can
  // report more than zero.
  syscall_marker("tessera-control-begin");
  void* volatile deliberate = std::malloc(64);
  if (deliberate != nullptr) {
    std::memset(deliberate, 1, 64);
  }
  syscall_marker("tessera-control-payload");
  std::free(deliberate);
  syscall_marker("tessera-control-end");
  const AllocCounts allocs_control = alloc_snapshot();
  alloc_counting(false);

  const AllocCounts window = alloc_delta(allocs_before, allocs_after);
  const AllocCounts control = alloc_delta(allocs_after, allocs_control);

  report->number("launches", static_cast<uint64_t>(launches));
  report->number("launch_failures", failures);
  report->number("alloc_counting_available", alloc_counting_available() ? 1 : 0);
  report->number("window_allocations", window.allocations());
  report->number("window_alloc_events", window.total());
  report->number("control_allocations", control.allocations());
  report->number("control_alloc_events", control.total());
  report->number("shim_launches_before_window", shim_launches_before);
  report->number("shim_launches_after_window", shim.ok() ? shim.symbol_count("cuLaunchKernel") : 0);
  report->number("shim_present", shim.ok() ? 1 : 0);

  // The null driver counts for itself, so the no-shim control run can still
  // show the launches arrived.
  unsigned long long (*driver_count)(const char*) = nullptr;
  bind(RTLD_DEFAULT, "tnull_call_count", &driver_count);
  report->number("driver_counts_available", driver_count != nullptr ? 1 : 0);
  report->number("driver_launches", driver_count != nullptr ? driver_count("cuLaunchKernel") : 0);
  return failures == 0 ? 0 : 1;
}

}  // namespace

// The shim calls dlopen to load the real driver, and an undefined symbol in a
// shared library binds to the executable's definition first, so this is the
// shim's dlopen. Every call but the one the fork scenario is waiting for is
// passed straight through.
//
// visibility("default") is required and not decorative: the project builds
// with -fvisibility=hidden, which hides this definition, and ENABLE_EXPORTS
// (-rdynamic) can only export what is not hidden. Without it the symbol never
// reaches .dynsym, the shim's dlopen binds to libc instead, the stall below
// never fires, and the fork scenario waits on its pipe forever.
extern "C" __attribute__((visibility("default"))) void* dlopen(const char* path, int flags) {
  static void* (*real)(const char*, int) = nullptr;
  if (real == nullptr) {
    void* p = dlsym(RTLD_NEXT, "dlopen");
    std::memcpy(&real, &p, sizeof(real));
    if (real == nullptr) {
      return nullptr;
    }
  }
  if (g_fork.armed && !g_fork.stalled && path != nullptr && g_fork.driver_path != nullptr &&
      std::strcmp(path, g_fork.driver_path) == 0) {
    g_fork.stalled = true;
    const char byte = 'x';
    const ssize_t written = ::write(g_fork.ready_pipe[1], &byte, 1);
    static_cast<void>(written);
    const timespec stall{g_fork.stall_ms / 1000, (g_fork.stall_ms % 1000) * 1000000L};
    ::nanosleep(&stall, nullptr);
  }
  return real(path, flags);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s workload <direct|dlopen|gpa1|gpa2> | ptsz | fork | "
                 "lazyinit [early] | selfref | hotpath [launches]\n",
                 argv[0]);
    return 2;
  }
  Report report;
  const char* mode = argv[1];
  int rc = 2;
  if (std::strcmp(mode, "workload") == 0 && argc == 3) {
    rc = mode_workload(argv[2], &report);
  } else if (std::strcmp(mode, "ptsz") == 0) {
    rc = mode_ptsz(&report);
  } else if (std::strcmp(mode, "fork") == 0) {
    rc = mode_fork(&report);
  } else if (std::strcmp(mode, "lazyinit") == 0) {
    rc = mode_lazyinit(&report, argc == 3 && std::strcmp(argv[2], "early") == 0);
  } else if (std::strcmp(mode, "selfref") == 0) {
    rc = mode_selfref(&report);
  } else if (std::strcmp(mode, "hotpath") == 0) {
    rc = mode_hotpath(&report, argc == 3 ? std::atoi(argv[2]) : 1000);
  } else {
    std::fprintf(stderr, "%s: unknown mode '%s'\n", argv[0], mode);
    return 2;
  }
  report.number("exit_code", static_cast<uint64_t>(rc));
  if (report.write(env_or_empty("TESSERA_APP_REPORT")) != 0) {
    std::fprintf(stderr, "app: could not write TESSERA_APP_REPORT\n");
    return 1;
  }
  // Returning from main runs both libraries' exit handlers, which is what
  // writes TESSERA_STATS_FILE and TESSERA_FAKE_STATS_FILE.
  return rc;
}
