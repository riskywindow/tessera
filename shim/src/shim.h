// libtessera's internal contract: the counters, the real-function slots, the
// hot path, and the lifecycle.
//
// Nothing here includes <cuda.h>. The typed wrappers do (hooks.cc,
// proc_address.cc); everything else works with indices and generic function
// pointers, which is what keeps the trampolines signature-free.

#ifndef TESSERA_SHIM_SRC_SHIM_H
#define TESSERA_SHIM_SRC_SHIM_H

#include <atomic>
#include <cstdint>

// Generated at configure time by shim/tools/gen_symbols.sh from the driver
// stub's export list: TESSERA_SYMBOL_COUNT and one TESSERA_SYM_<name> index
// per exported symbol.
#include "tessera_symbol_index.h"

#define TESSERA_HIDDEN __attribute__((visibility("hidden")))

namespace tessera {

// A function pointer of unknown signature. Every slot holds one of these, so
// the hot path never converts between object and function pointers.
using AnyFn = void (*)();

static_assert(sizeof(AnyFn) == 8, "the trampolines index 8-byte slots");
static_assert(sizeof(std::atomic<AnyFn>) == sizeof(AnyFn), "slot layout");
static_assert(sizeof(std::atomic<std::uint64_t>) == sizeof(std::uint64_t), "counter layout");

}  // namespace tessera

extern "C" {

// The two arrays the assembly trampolines address directly, indexed by symbol
// index. They must stay hidden: the trampolines reach them with a PC-relative
// reference, which the linker only allows for a symbol it can bind locally.
//
//   tessera_shim_counts  per exported symbol call count. A hooked wrapper
//                        bumps its own entry in C++; a trampoline bumps its
//                        entry with one atomic add in assembly.
//   tessera_shim_slots   the real driver's function for that symbol, resolved
//                        lazily on first use and never from a constructor.
TESSERA_HIDDEN extern std::atomic<std::uint64_t> tessera_shim_counts[TESSERA_SYMBOL_COUNT];
TESSERA_HIDDEN extern std::atomic<tessera::AnyFn> tessera_shim_slots[TESSERA_SYMBOL_COUNT];

// Called by a trampoline whose slot is still empty, with the symbol index.
// Always returns something callable: the real function, or a stub that returns
// CUDA_ERROR_NOT_FOUND, because the trampoline is about to jump to it (I-5:
// fail open, never crash the tenant).
TESSERA_HIDDEN tessera::AnyFn tessera_shim_resolve_tramp(unsigned index);

}  // extern "C"

namespace tessera {

// ---------------------------------------------------------------------------
// Aggregate counters (I-4: relaxed atomics, nothing else).
// ---------------------------------------------------------------------------

struct Stats {
  std::atomic<std::uint64_t> launches;
  std::atomic<std::uint64_t> graph_launches;
  std::atomic<std::uint64_t> hooked_calls;
  std::atomic<std::uint64_t> bypassed_lookups;
};

// forwarded_unhooked is not a field: it is the sum of the per-symbol counters
// of the symbols this build does not hook, computed when stats are read. That
// keeps the trampoline fast path to a single atomic add on one cache line
// instead of two.
TESSERA_HIDDEN extern Stats g_stats;

enum class Kind { kPlain, kLaunch, kGraphLaunch };

// THE HOT PATH. Everything a hooked wrapper does before forwarding:
// three relaxed atomic adds at a compile-time index and one acquire load.
// No string hashing, no map lookup, no lock, no allocation, no syscall (I-4).
// Returns the real function, or nullptr if it has not been resolved yet.
template <Kind K>
inline AnyFn hot_enter(unsigned index) noexcept {
  tessera_shim_counts[index].fetch_add(1, std::memory_order_relaxed);
  g_stats.hooked_calls.fetch_add(1, std::memory_order_relaxed);
  if constexpr (K == Kind::kLaunch) {
    g_stats.launches.fetch_add(1, std::memory_order_relaxed);
  } else if constexpr (K == Kind::kGraphLaunch) {
    g_stats.graph_launches.fetch_add(1, std::memory_order_relaxed);
  }
  return tessera_shim_slots[index].load(std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// Resolution (cold path)
// ---------------------------------------------------------------------------

// Initialises the shim if needed and looks this symbol up in the real driver,
// caching the result in its slot. Returns nullptr if the driver could not be
// loaded or does not export the symbol.
TESSERA_HIDDEN AnyFn resolve_slow(unsigned index) noexcept;

// The CUresult a hooked wrapper returns when it has no real function to
// forward to: CUDA_ERROR_NOT_INITIALIZED when the driver could not be loaded,
// CUDA_ERROR_NOT_FOUND when the driver simply does not export this symbol.
// Returned as int so this header stays free of <cuda.h>.
TESSERA_HIDDEN int unavailable_result(unsigned index) noexcept;

// ---------------------------------------------------------------------------
// Symbol table (generated from the driver stub's exports)
// ---------------------------------------------------------------------------

TESSERA_HIDDEN const char* symbol_name(unsigned index) noexcept;
TESSERA_HIDDEN bool symbol_is_hooked(unsigned index) noexcept;
// Binary search over the sorted name table; -1 when this build has no such
// exported symbol.
TESSERA_HIDDEN int symbol_index(const char* name) noexcept;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Loads the real driver on first use. Never called from a static constructor:
// a library masquerading as libcuda.so.1 is constructed at unpredictable
// times, and calling the driver from there is how shims deadlock (ADR-001).
// Returns false if the driver could not be loaded; the caller then fails the
// one call rather than aborting the tenant (I-5).
TESSERA_HIDDEN bool ensure_init() noexcept;

// The real driver's handle, or nullptr before a successful ensure_init().
TESSERA_HIDDEN void* real_driver_handle() noexcept;

// Called once by ensure_init(), immediately after the driver is loaded, to
// resolve every hooked variant's real address. cuGetProcAddress's identity
// check compares against those addresses, so they have to exist before the
// first lookup can be answered.
TESSERA_HIDDEN void on_driver_loaded() noexcept;

// Called from the pthread_atfork child handler: per-process state that a fork
// invalidates. Runs with the shim's locks held by this thread (the prepare
// handler took them), so no other thread can be holding one.
TESSERA_HIDDEN void reset_after_fork() noexcept;

// ---------------------------------------------------------------------------
// Environment, logging, dlsym
// ---------------------------------------------------------------------------

// getenv, in one place so that its thread-safety caveat is documented once.
TESSERA_HIDDEN const char* env(const char* name) noexcept;

TESSERA_HIDDEN bool log_enabled() noexcept;
TESSERA_HIDDEN void log_printf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// Never on the hot path: the condition is a relaxed load of a cached flag, but
// every call site is init, resolution or teardown.
#define TESSERA_LOG(...)                \
  do {                                  \
    if (tessera::log_enabled()) {       \
      tessera::log_printf(__VA_ARGS__); \
    }                                   \
  } while (0)

using DlsymFn = void* (*)(void*, const char*);
using DlvsymFn = void* (*)(void*, const char*, const char*);

// The real dlsym/dlvsym, found without calling either by name. The preload
// build interposes both, so a call by name would land back in our own hook;
// see native_dlsym.cc. Either may be nullptr if the bootstrap failed.
TESSERA_HIDDEN DlsymFn real_dlsym() noexcept;
TESSERA_HIDDEN DlvsymFn real_dlvsym() noexcept;

// dlsym on the real driver's handle, as a function pointer. Uses real_dlsym(),
// never the interposed one.
TESSERA_HIDDEN AnyFn driver_symbol(const char* name) noexcept;

// ---------------------------------------------------------------------------
// cuGetProcAddress support (proc_address.cc)
// ---------------------------------------------------------------------------

// One hooked entry point: its index, its exported name, the base name an
// application passes to cuGetProcAddress, and our wrapper.
struct HookRow {
  unsigned index;
  const char* name;
  const char* base;
  AnyFn wrapper;
};

TESSERA_HIDDEN const HookRow* hook_rows() noexcept;
TESSERA_HIDDEN unsigned hook_row_count() noexcept;

// Maps what the real driver returned for `symbol` to the shim's wrapper for
// the variant the driver picked, by pointer identity. Returns `real` unchanged
// when the base symbol is not hooked, and when it is hooked but no known
// variant matches (bumping bypassed_lookups in that case). See ADR-001.
TESSERA_HIDDEN void* map_proc_address(const char* symbol, void* real) noexcept;

}  // namespace tessera

#endif  // TESSERA_SHIM_SRC_SHIM_H
