// cuGetProcAddress: the hook the rest depend on.
//
// The CUDA runtime does not call the exported cu* symbols at all. It resolves
// function pointers once, by name plus a cudaVersion plus flags -- including
// resolving cuGetProcAddress itself that way -- and calls through those
// pointers afterwards. A shim that exports cuLaunchKernel but does not hook
// the lookup sees nothing from a cudart application, which is nearly all of
// them (ADR-001).
//
// Resolution here is by IDENTITY, never by a hand-maintained version table:
//
//   1. forward the request to the real driver;
//   2. compare the pointer it returned against the real address of each
//      variant this build knows for that base name;
//   3. whichever matches tells us exactly which ABI the caller is about to
//      use, so return the corresponding wrapper.
//
// cudaVersion and the CU_GET_PROC_ADDRESS_{DEFAULT,LEGACY_STREAM,
// PER_THREAD_DEFAULT_STREAM} flags are therefore honoured by construction: the
// driver already applied them when it chose what to return.
//
// If nothing matches -- a newer driver handing back a variant this build has
// never heard of -- the REAL pointer goes back unchanged and
// bypassed_lookups is incremented. Guessing at an ABI would corrupt the
// tenant's arguments; under-counting is recoverable and visible.

#include "cuda_api.h"
// cuda_api.h configures and includes <cuda.h>; nothing may precede it.

#include <atomic>
#include <cstdint>
#include <cstring>

#include "shim.h"

namespace tessera {
namespace {

// Every hooked variant: index, exported name, the base name an application
// passes to cuGetProcAddress, and our wrapper.
//
// The wrapper addresses are link-time constants, so this table costs no
// run-time initialisation worth the name; nothing in it calls anything.
// NOLINTBEGIN(bugprone-macro-parentheses): `name` is pasted into an identifier
// and used as a declarator; parentheses would not compile.
#define TESSERA_WRAP(name, base, kind, params, args) \
  {TESSERA_SYM_##name, #name, #base, reinterpret_cast<tessera::AnyFn>(&name)},
#define TESSERA_HOOK_MANUAL(name, base) \
  {TESSERA_SYM_##name, #name, #base, reinterpret_cast<tessera::AnyFn>(&name)},
// NOLINTEND(bugprone-macro-parentheses)
const HookRow kRows[] = {
#include "hook_table.inc"
};
#undef TESSERA_WRAP
#undef TESSERA_HOOK_MANUAL

constexpr unsigned kRowCount = sizeof(kRows) / sizeof(kRows[0]);
static_assert(kRowCount == TESSERA_HOOKED_SYMBOL_COUNT,
              "the hook table and the generated symbol index disagree about how many "
              "symbols are hooked");

// A direct-mapped cache of answers, keyed by the pointer the driver returned.
// Each entry is a single pointer to an immutable row, so it is published with
// one atomic store and can never be read half-written: no lock, and nothing
// for fork() to find held.
constexpr unsigned kCacheSize = 256;
std::atomic<const HookRow*> g_cache[kCacheSize];

unsigned cache_slot(const void* key) noexcept {
  const std::uintptr_t value = reinterpret_cast<std::uintptr_t>(key);
  return static_cast<unsigned>((value >> 4) ^ (value >> 12)) & (kCacheSize - 1);
}

// The driver API hands function pointers back through void*, so the two have
// to be interchanged somewhere; doing it in one place keeps it honest.
void* as_void(AnyFn fn) noexcept {
  return reinterpret_cast<void*>(fn);
}

// The real driver's address for a hooked row, resolved in on_driver_loaded().
void* real_address(const HookRow& row) noexcept {
  return as_void(tessera_shim_slots[row.index].load(std::memory_order_acquire));
}

}  // namespace

const HookRow* hook_rows() noexcept {
  return kRows;
}

unsigned hook_row_count() noexcept {
  return kRowCount;
}

void* map_proc_address(const char* symbol, void* real) noexcept {
  if (symbol == nullptr || real == nullptr) {
    return real;
  }

  const unsigned slot = cache_slot(real);
  const HookRow* cached = g_cache[slot].load(std::memory_order_relaxed);
  if (cached != nullptr && real_address(*cached) == real &&
      std::strcmp(cached->base, symbol) == 0) {
    return as_void(cached->wrapper);
  }

  bool base_is_hooked = false;
  for (const HookRow& row : kRows) {
    if (std::strcmp(row.base, symbol) != 0) {
      continue;
    }
    base_is_hooked = true;
    if (real_address(row) == real) {
      g_cache[slot].store(&row, std::memory_order_relaxed);
      return as_void(row.wrapper);
    }
    if (as_void(row.wrapper) == real) {
      // The driver handed back one of OUR wrappers. This is not a bypass, it
      // is interposition working: under the masquerade strategy the shim is
      // the object called libcuda.so.1 and sits in the application's global
      // scope, so the real driver's own reference to its own exported symbol
      // -- the address its cuGetProcAddress table is built from -- resolves
      // to us unless it was linked with -Bsymbolic. Measured on this host:
      // with the shim as libcuda.so.1, the fake driver's
      // tf_symbol_address("cuLaunchKernel") returns the shim's wrapper, not
      // its own entry point. Either way the caller ends up in the wrapper,
      // which then forwards through its slot -- and the slot was filled by a
      // handle-specific dlsym, which interposition cannot touch.
      return real;
    }
  }

  if (base_is_hooked) {
    // The driver handed back a variant of a symbol we hook that we do not
    // recognise. Pass it through and say so: this counter is how H1 tells
    // "we captured everything" from "we captured what we knew about".
    g_stats.bypassed_lookups.fetch_add(1, std::memory_order_relaxed);
    TESSERA_LOG(
        "cuGetProcAddress('%s') returned an unrecognised variant at %p; forwarding the "
        "real pointer and counting a bypassed lookup",
        symbol, real);
  }
  return real;
}

}  // namespace tessera

extern "C" {

CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion,
                                  cuuint64_t flags) {
  using Fn = CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t);
  tessera::AnyFn real = tessera::hot_enter<tessera::Kind::kPlain>(TESSERA_SYM_cuGetProcAddress);
  if (real == nullptr) {
    real = tessera::resolve_slow(TESSERA_SYM_cuGetProcAddress);
    if (real == nullptr) {
      return static_cast<CUresult>(tessera::unavailable_result(TESSERA_SYM_cuGetProcAddress));
    }
  }
  const CUresult result = reinterpret_cast<Fn>(real)(symbol, pfn, cudaVersion, flags);
  if (result == CUDA_SUCCESS && pfn != nullptr && *pfn != nullptr) {
    *pfn = tessera::map_proc_address(symbol, *pfn);
  }
  return result;
}

CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion,
                                     cuuint64_t flags,
                                     CUdriverProcAddressQueryResult* symbolStatus) {
  using Fn =
      CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
  tessera::AnyFn real = tessera::hot_enter<tessera::Kind::kPlain>(TESSERA_SYM_cuGetProcAddress_v2);
  if (real == nullptr) {
    real = tessera::resolve_slow(TESSERA_SYM_cuGetProcAddress_v2);
    if (real == nullptr) {
      return static_cast<CUresult>(tessera::unavailable_result(TESSERA_SYM_cuGetProcAddress_v2));
    }
  }
  // symbolStatus is the driver's to fill: SUCCESS, SYMBOL_NOT_FOUND or
  // VERSION_NOT_SUFFICIENT. Substituting a wrapper does not change which
  // symbol was found, so whatever it wrote is passed through untouched.
  const CUresult result = reinterpret_cast<Fn>(real)(symbol, pfn, cudaVersion, flags, symbolStatus);
  if (result == CUDA_SUCCESS && pfn != nullptr && *pfn != nullptr) {
    *pfn = tessera::map_proc_address(symbol, *pfn);
  }
  return result;
}

}  // extern "C"
