// The dlsym hook. PRELOAD BUILD ONLY -- this is the one object that is not in
// the masquerading libcuda.so.1.
//
// Why it exists: dlsym on an explicit handle returns the symbol from THAT
// object, so LD_PRELOAD does not affect it. Triton's launcher stubs, some
// PyTorch paths and vLLM custom ops open the driver with dlopen and look
// symbols up by name; a preload-only shim is bypassed there. That was
// measured, not assumed: scripts/loader_semantics_probe.sh recorded the
// dlopen+dlsym path reaching the real library directly under LD_PRELOAD
// (results/m0/loader_semantics.json, row 2), which is the empirical reason
// masquerade is the primary strategy and this hook is the repair.
//
// RESIDUAL LIMITATION, stated plainly. glibc computes what RTLD_NEXT and
// RTLD_DEFAULT mean from the RETURN ADDRESS of the call into dlsym, so a
// forwarded lookup must leave that return address as the caller's. Two things
// follow:
//
//   * A lookup we do not intercept is TAIL-CALLED to the real dlsym, so the
//     return address stays the caller's and the pseudo-handles keep their
//     caller-relative meaning. The tail call is forced with the musttail
//     attribute where the compiler has it (clang 13+, gcc 15+). On an older
//     compiler it degrades to an ordinary call, and RTLD_NEXT/RTLD_DEFAULT
//     lookups made by the application would then resolve relative to
//     libtessera.so instead of the caller. That is a build-configuration
//     caveat of the secondary strategy, not of masquerade.
//   * A lookup we DO intercept is answered without calling the real dlsym on
//     the caller's behalf for a pseudo-handle: requests with RTLD_NEXT or
//     RTLD_DEFAULT are always passed straight through, precisely because we
//     cannot answer them correctly from here. A real handle does not depend on
//     the return address, so intercepting those is safe.

#include <dlfcn.h>

#include "shim.h"

#if defined(__has_attribute)
#if __has_attribute(musttail)
#define TESSERA_MUSTTAIL __attribute__((musttail))
#endif
#endif
#if !defined(TESSERA_MUSTTAIL)
#define TESSERA_MUSTTAIL
#endif

namespace tessera {
namespace {

// The shim's wrapper for a hooked symbol, or nullptr when the symbol is not
// one we hook.
void* hooked_wrapper(const char* name) noexcept {
  const int index = symbol_index(name);
  if (index < 0 || !symbol_is_hooked(static_cast<unsigned>(index))) {
    return nullptr;
  }
  const HookRow* rows = hook_rows();
  for (unsigned i = 0; i < hook_row_count(); ++i) {
    if (rows[i].index == static_cast<unsigned>(index)) {
      return reinterpret_cast<void*>(rows[i].wrapper);
    }
  }
  return nullptr;
}

bool is_pseudo_handle(void* handle) noexcept {
  return handle == RTLD_NEXT || handle == RTLD_DEFAULT;
}

// Only a driver entry point can be ours, and every one of them starts with
// "cu". Two character comparisons keep everything else -- including the
// lookups AddressSanitizer's runtime makes while it is still initialising,
// before any interceptor may run -- on the shortest possible path to the real
// dlsym.
bool could_be_a_driver_symbol(const char* name) noexcept {
  return name != nullptr && name[0] == 'c' && name[1] == 'u';
}

}  // namespace
}  // namespace tessera

extern "C" {

__attribute__((visibility("default"))) void* dlsym(void* handle, const char* name) noexcept {
  const tessera::DlsymFn real = tessera::real_dlsym();
  if (real == nullptr) {
    return nullptr;  // the bootstrap failed; there is nothing to forward to
  }
  if (tessera::could_be_a_driver_symbol(name) && !tessera::is_pseudo_handle(handle)) {
    void* wrapper = tessera::hooked_wrapper(name);
    // Substitute only when the object really provides the symbol: a handle
    // that does not have cuLaunchKernel must not suddenly appear to.
    if (wrapper != nullptr && real(handle, name) != nullptr) {
      return wrapper;
    }
  }
  TESSERA_MUSTTAIL return real(handle, name);
}

__attribute__((visibility("default"))) void* dlvsym(void* handle, const char* name,
                                                    const char* version) noexcept {
  const tessera::DlvsymFn real = tessera::real_dlvsym();
  if (real == nullptr) {
    return nullptr;
  }
  if (tessera::could_be_a_driver_symbol(name) && !tessera::is_pseudo_handle(handle)) {
    void* wrapper = tessera::hooked_wrapper(name);
    // The real libcuda.so.1 exports unversioned symbols, so a versioned lookup
    // of a driver entry point normally finds nothing and falls through. This
    // is here so that a driver which does version its symbols cannot escape.
    if (wrapper != nullptr && real(handle, name, version) != nullptr) {
      return wrapper;
    }
  }
  TESSERA_MUSTTAIL return real(handle, name, version);
}

}  // extern "C"
