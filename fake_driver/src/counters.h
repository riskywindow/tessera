// Per-exported-symbol call counters, forced results, and the symbol table that
// cuGetProcAddress resolves against.
//
// The counters live outside Sim::mu on purpose. Counting a call must not take
// a lock: the shim's hot-path tests measure the fake's overhead as well as
// their own, and a lock here would be noise. They are plain relaxed atomics,
// zero-initialised before any dynamic initialisation runs, so an entry point
// reached before the library's lazy init still counts.

#ifndef TESSERA_FAKE_COUNTERS_H
#define TESSERA_FAKE_COUNTERS_H

#include <atomic>
#include <cstdint>

#include <cuda.h>

namespace tf {

// One index per exported cu* entry point, generated from table.inc.
enum SymIndex : uint32_t {
#define TF_ROW(exported, base, version, kind) TF_SYM_##exported,
#include "table.inc"
#undef TF_ROW
  TF_SYM_COUNT
};

enum StreamKind : int {
  TF_STREAM_NONE = 0,
  TF_STREAM_LEGACY = 1,
  TF_STREAM_PER_THREAD = 2,
};

struct SymbolRow {
  const char* exported;
  const char* base;
  int version;
  StreamKind stream_kind;
  void (*fn)(void);
};

// Defined in entry_points.cc, where every entry point is in scope.
const SymbolRow* symbol_table();
uint32_t symbol_table_size();

// -1 when the name is not an exported symbol.
int64_t symbol_index(const char* exported_name);

extern std::atomic<uint64_t> g_counts[TF_SYM_COUNT];
// 0 means "no forced result"; otherwise the forced CUresult plus one.
extern std::atomic<int32_t> g_forced[TF_SYM_COUNT];
// Set by tf_set_fork_mode(1) together with Sim::in_forked_child.
extern std::atomic<int> g_fork_emulate_real;
extern std::atomic<int> g_in_forked_child;

void library_init_once();

// Counts the call and reports whether the entry point must return early.
// Returns true and writes *forced when a result is being injected, or when the
// call is happening in a forked child and fork emulation is on.
inline bool enter(SymIndex sym, CUresult* forced) {
  g_counts[sym].fetch_add(1, std::memory_order_relaxed);
  if (g_fork_emulate_real.load(std::memory_order_relaxed) != 0 &&
      g_in_forked_child.load(std::memory_order_relaxed) != 0) {
    *forced = CUDA_ERROR_NOT_INITIALIZED;
    return true;
  }
  const int32_t f = g_forced[sym].load(std::memory_order_relaxed);
  if (f != 0) {
    *forced = static_cast<CUresult>(f - 1);
    return true;
  }
  return false;
}

}  // namespace tf

// Every exported entry point starts with this. It counts the call, honours
// tf_set_result and tf_set_fork_mode, and makes sure the library's one-time
// initialisation has run.
#define TF_ENTER(sym)                                        \
  CUresult tf_forced_result_;                                \
  do {                                                       \
    ::tf::library_init_once();                               \
    if (::tf::enter(::tf::TF_SYM_##sym, &tf_forced_result_)) \
      return tf_forced_result_;                              \
  } while (0)

#endif  // TESSERA_FAKE_COUNTERS_H
