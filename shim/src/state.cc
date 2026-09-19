// Lifecycle: loading the real driver, the fork handlers, logging, and the
// exit-time stats dump.
//
// Nothing in this file runs from a static constructor. A library masquerading
// as libcuda.so.1 is constructed at times we do not control, and calling the
// driver from there is how shims deadlock (ADR-001, test T5). The first hooked
// call or trampoline hit is what initialises us.

#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tessera/shim/stats.h"

#include "shim.h"

namespace tessera {
namespace {

enum InitState : int { kUninitialised = 0, kReady = 1, kFailed = 2 };

pthread_mutex_t g_init_mu = PTHREAD_MUTEX_INITIALIZER;
std::atomic<int> g_init_state{kUninitialised};
// Published as soon as the driver is open and has passed the self-load guard,
// which is before the state becomes kReady: on_driver_loaded() resolves the
// hooked symbols through it. Any other thread is still waiting on g_init_mu at
// that point, so nobody sees a half-built table.
std::atomic<void*> g_handle{nullptr};
std::atomic<bool> g_atfork_registered{false};
std::atomic<bool> g_suppress_exit_dump{false};

// Guards against a driver whose own initialisation re-enters a hooked symbol
// on this thread: that call fails rather than deadlocking on a lock this
// thread already holds (I-5).
thread_local bool t_initialising = false;

std::atomic<int> g_log_state{-1};  // -1 unknown, 0 off, 1 on

#if !defined(TESSERA_MULTIARCH_DIR)
#define TESSERA_MULTIARCH_DIR ""
#endif

// The documented fallback search, used only when TESSERA_REAL_LIBCUDA is
// unset. `tessera run` always sets it (it discovers the path with ldconfig);
// these are for a shim loaded by hand. The bare SONAME is last and is safe
// only because of the self-load guard below: under the masquerade strategy our
// own directory is first on LD_LIBRARY_PATH, so a bare dlopen finds us.
const char* const kFallbackPaths[] = {
    "/usr/lib/" TESSERA_MULTIARCH_DIR "/libcuda.so.1",
    "/usr/lib64/libcuda.so.1",
    "/usr/lib/libcuda.so.1",
    "libcuda.so.1",
};

// True when `handle` is this very shared object, which is what a
// TESSERA_REAL_LIBCUDA pointing back at the shim produces. Comparing the
// module the handle's cuInit lives in against our own is exact, and catches
// the case whatever the file was called (test T6).
bool handle_is_self(void* handle) noexcept {
  const DlsymFn lookup = real_dlsym();
  if (lookup == nullptr) {
    return false;
  }
  void* their_cu_init = lookup(handle, "cuInit");
  if (their_cu_init == nullptr) {
    return false;
  }
  Dl_info theirs;
  Dl_info ours;
  std::memset(&theirs, 0, sizeof(theirs));
  std::memset(&ours, 0, sizeof(ours));
  static const char kMarker = 0;
  if (dladdr(their_cu_init, &theirs) == 0 || dladdr(&kMarker, &ours) == 0) {
    return false;
  }
  return theirs.dli_fbase == ours.dli_fbase;
}

void* open_driver(const char* path, bool explicit_request) noexcept {
  // RTLD_LOCAL keeps the driver's symbols out of the global scope, so it
  // cannot interpose on us; RTLD_NOW surfaces a broken driver here rather
  // than at the first forwarded call (ADR-001).
  // The path comes from TESSERA_REAL_LIBCUDA and is meant to: `tessera run`
  // sets it, and a library it names is loaded with the caller's own
  // privileges. The guard that matters -- refusing an object that turns out to
  // be the shim itself -- is below.
  // NOLINTNEXTLINE(clang-analyzer-optin.taint.GenericTaint)
  void* handle = dlopen(path, RTLD_LOCAL | RTLD_NOW);
  if (handle == nullptr) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): glibc's dlerror is per-thread.
    const char* why = dlerror();
    TESSERA_LOG("cannot load real driver '%s': %s", path, why != nullptr ? why : "unknown");
    return nullptr;
  }
  if (handle_is_self(handle)) {
    TESSERA_LOG("%s'%s' resolves to the shim itself; refusing to forward to ourselves",
                explicit_request ? "TESSERA_REAL_LIBCUDA=" : "fallback ", path);
    dlclose(handle);
    return nullptr;
  }
  return handle;
}

void atfork_prepare() noexcept {
  pthread_mutex_lock(&g_init_mu);
}

void atfork_parent() noexcept {
  pthread_mutex_unlock(&g_init_mu);
}

// The child holds every shim lock (the prepare handler took them on this
// thread), so no lock can be held by a thread that did not survive the fork.
void atfork_child() noexcept {
  reset_after_fork();
  g_suppress_exit_dump.store(true, std::memory_order_relaxed);
  pthread_mutex_unlock(&g_init_mu);
}

void do_init() noexcept {
  if (!g_atfork_registered.exchange(true, std::memory_order_relaxed)) {
    pthread_atfork(atfork_prepare, atfork_parent, atfork_child);
  }

  void* handle = nullptr;
  const char* requested = env("TESSERA_REAL_LIBCUDA");
  if (requested != nullptr && requested[0] != '\0') {
    if (requested[0] != '/') {
      TESSERA_LOG(
          "TESSERA_REAL_LIBCUDA='%s' is not an absolute path; loading it anyway, but an "
          "absolute path is what keeps the loader from finding the shim again",
          requested);
    }
    handle = open_driver(requested, true);
    // Deliberately no fallback: an explicit setting that does not work is a
    // misconfiguration we want to be loud, not one we paper over.
  } else {
    for (const char* candidate : kFallbackPaths) {
      handle = open_driver(candidate, false);
      if (handle != nullptr) {
        TESSERA_LOG("TESSERA_REAL_LIBCUDA unset; using '%s'", candidate);
        break;
      }
    }
  }

  if (handle == nullptr) {
    g_init_state.store(kFailed, std::memory_order_release);
    return;
  }
  g_handle.store(handle, std::memory_order_release);
  on_driver_loaded();
  g_init_state.store(kReady, std::memory_order_release);
  TESSERA_LOG("initialised (pid %ld, abi %d)", static_cast<long>(getpid()), TESSERA_ABI_VERSION);
}

// Runs at process exit (or dlclose). Writing a file here is safe; calling the
// driver would not be, and we do not.
__attribute__((destructor)) void shim_fini() {
  if (g_suppress_exit_dump.load(std::memory_order_relaxed)) {
    return;
  }
  const char* path = env("TESSERA_STATS_FILE");
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  if (tessera_dump_stats(path) != 0) {
    TESSERA_LOG("could not write TESSERA_STATS_FILE='%s' (errno %d)", path, errno);
  }
}

}  // namespace

const char* env(const char* name) noexcept {
  // getenv races only against setenv in another thread. The shim reads its
  // configuration once, during lazy init, before it has done anything the
  // tenant can observe; a tenant that rewrites its own environment
  // concurrently would have the same race with the loader.
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  return std::getenv(name);
}

bool log_enabled() noexcept {
  int state = g_log_state.load(std::memory_order_relaxed);
  if (state < 0) {
    const char* value = env("TESSERA_LOG");
    state = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    g_log_state.store(state, std::memory_order_relaxed);
  }
  return state == 1;
}

// A variadic is what carries the printf format attribute that -Wformat=2
// checks the call sites against; a parameter pack would take that check away
// from the only code here that formats a string.
// NOLINTNEXTLINE(cert-dcl50-cpp)
void log_printf(const char* fmt, ...) {
  char buffer[512];
  const int prefix =
      std::snprintf(buffer, sizeof(buffer), "tessera[%ld]: ", static_cast<long>(getpid()));
  if (prefix < 0 || static_cast<size_t>(prefix) >= sizeof(buffer)) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  const int body =
      std::vsnprintf(buffer + prefix, sizeof(buffer) - static_cast<size_t>(prefix), fmt, args);
  va_end(args);
  if (body < 0) {
    return;
  }
  size_t length = static_cast<size_t>(prefix) + static_cast<size_t>(body);
  if (length >= sizeof(buffer) - 1) {
    length = sizeof(buffer) - 2;
  }
  buffer[length] = '\n';
  ++length;
  // Unbuffered and one write: the tenant's stderr may be shared with threads
  // we know nothing about.
  const ssize_t written = write(STDERR_FILENO, buffer, length);
  static_cast<void>(written);
}

void* real_driver_handle() noexcept {
  return g_handle.load(std::memory_order_acquire);
}

bool ensure_init() noexcept {
  const int state = g_init_state.load(std::memory_order_acquire);
  if (state == kReady) {
    return true;
  }
  if (state == kFailed) {
    return false;
  }
  if (t_initialising) {
    // Re-entered from the driver's own initialisation on this thread.
    return false;
  }
  t_initialising = true;
  pthread_mutex_lock(&g_init_mu);
  if (g_init_state.load(std::memory_order_relaxed) == kUninitialised) {
    do_init();
  }
  pthread_mutex_unlock(&g_init_mu);
  t_initialising = false;
  return g_init_state.load(std::memory_order_acquire) == kReady;
}

}  // namespace tessera
