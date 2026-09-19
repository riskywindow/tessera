// The hooked entry points: ordinary typed C++ wrappers, one per exported
// variant, generated from shim/src/hook_table.inc.
//
// Each one is the hot path (I-4). It bumps its own counter and the aggregates
// with relaxed atomics at a COMPILE-TIME index, loads its slot, and tail-calls
// the real driver. No string hashing, no map lookup, no lock, no allocation,
// no syscall. M2 adds gating here and nowhere else.
//
// The signatures come from cuda.h itself, so the compiler checks every one of
// them against the declaration the driver's own sources use: a wrong parameter
// list is a build error, not a silent ABI mismatch that would corrupt a
// tenant's arguments.

#include "cuda_api.h"
// cuda_api.h configures and includes <cuda.h>; nothing may precede it.

#include "shim.h"

// NOLINTBEGIN(bugprone-macro-parentheses): `name`, `params` and `args` are a
// declarator and its parameter and argument lists; parentheses around them
// would not compile.
// One wrapper per row. `kind` selects which aggregate counter the call bumps,
// as a template argument, so the branch is resolved at compile time.
#define TESSERA_WRAP(name, base, kind, params, args)                                   \
  extern "C" CUresult CUDAAPI name params {                                            \
    using Fn = CUresult(CUDAAPI*) params;                                              \
    tessera::AnyFn real = tessera::hot_enter<tessera::Kind::kind>(TESSERA_SYM_##name); \
    if (real == nullptr) {                                                             \
      real = tessera::resolve_slow(TESSERA_SYM_##name);                                \
      if (real == nullptr) {                                                           \
        return static_cast<CUresult>(tessera::unavailable_result(TESSERA_SYM_##name)); \
      }                                                                                \
    }                                                                                  \
    return reinterpret_cast<Fn>(real) args;                                            \
  }

// cuGetProcAddress and cuGetProcAddress_v2 are written by hand in
// proc_address.cc: they do more than forward.
#define TESSERA_HOOK_MANUAL(name, base)

#include "hook_table.inc"

#undef TESSERA_WRAP
#undef TESSERA_HOOK_MANUAL
// NOLINTEND(bugprone-macro-parentheses)
