// Finding the real dlsym and dlvsym, without calling either by name -- and
// without calling anything a sanitizer interposes on.
//
// WHY IT CANNOT JUST CALL dlvsym. The preload build interposes BOTH dlsym and
// dlvsym, and an interposing definition in an LD_PRELOAD'd object wins for
// that object's own calls too: calling dlvsym("dlsym") from here lands back in
// our own hook. That was measured, not assumed -- a two-file experiment on
// this host showed a preload's dlvsym intercepting its own bootstrap call.
//
// WHY IT WALKS THE LINK MAP BY HAND. The first caller of our dlsym hook is not
// the application; under AddressSanitizer it is the sanitizer runtime, during
// its own initialisation. Anything this bootstrap calls that ASan interposes
// on -- dl_iterate_phdr, dladdr, strcmp -- runs an interceptor before the
// runtime is ready, and ASan aborts with
// "CHECK failed: asan_posix.cpp:125 ((tsd_key_inited)) != (0)". That was
// measured too: with dl_iterate_phdr and dladdr here, the preload strategy
// died on every clang-asan run and the masquerade strategy (no dlsym hook)
// passed. So the bootstrap reads the loader's own data structures instead:
//
//   _r_debug.r_map   the link map the dynamic loader maintains, and the
//                    debugger interface it exports for exactly this purpose;
//   _DYNAMIC         this object's own dynamic section, which identifies which
//                    link-map entry is us without asking anyone;
//   str_equal        a four-line comparison, because strcmp is interposed.
//
// Nothing here calls into libc, so there is nothing for a sanitizer to
// intercept and nothing for the loader to interpose.
//
// ADR-001 describes the bootstrap as dlvsym(RTLD_NEXT, "dlsym",
// "GLIBC_2.34") with a fallback to the architecture's base version. That is
// what happens once the real dlvsym is in hand: the versioned lookup below is
// the ADR's, and the link-map walk is only how we get a dlvsym to make it
// with.

#include <dlfcn.h>
#include <elf.h>
#include <link.h>

#include <atomic>
#include <cstdint>

#include "shim.h"

// <link.h> declares both of the loader structures this file reads:
// _r_debug, the link map the dynamic loader maintains, and _DYNAMIC, which in
// any shared object is that object's own dynamic section -- so the reference
// below resolves to ours, which is what makes the self-check exact.

// NOLINTBEGIN(performance-no-int-to-ptr): reading another object's ELF tables
// is integer-to-pointer arithmetic by nature -- a load bias plus a link-time
// address is what every one of these is -- and there is no typed handle to
// take instead.
namespace tessera {
namespace {

bool str_equal(const char* a, const char* b) noexcept {
  while (*a != '\0' && *a == *b) {
    ++a;
    ++b;
  }
  return *a == *b;
}

std::uint32_t gnu_hash(const char* name) noexcept {
  std::uint32_t h = 5381;
  for (const char* p = name; *p != '\0'; ++p) {
    h = (h << 5) + h + static_cast<unsigned char>(*p);
  }
  return h;
}

// One object's dynamic-section pointers. On Linux/glibc the loader rewrites
// DT_* addresses in place by adding the load bias, but that is not guaranteed
// for every loader, so a value below the object's base is biased here instead.
struct Dynamic {
  const char* strtab = nullptr;
  const ElfW(Sym) * symtab = nullptr;
  const std::uint32_t* gnu_hash_tab = nullptr;
  const ElfW(Word) * sysv_hash_tab = nullptr;
};

Dynamic read_dynamic(const ElfW(Dyn) * dyn, ElfW(Addr) base) noexcept {
  Dynamic d;
  if (dyn == nullptr) {
    return d;
  }
  for (; dyn->d_tag != DT_NULL; ++dyn) {
    ElfW(Addr) value = dyn->d_un.d_ptr;
    if (value == 0) {
      continue;
    }
    if (value < base) {
      value += base;
    }
    switch (dyn->d_tag) {
      case DT_STRTAB:
        d.strtab = reinterpret_cast<const char*>(value);
        break;
      case DT_SYMTAB:
        d.symtab = reinterpret_cast<const ElfW(Sym)*>(value);
        break;
      case DT_GNU_HASH:
        d.gnu_hash_tab = reinterpret_cast<const std::uint32_t*>(value);
        break;
      case DT_HASH:
        d.sysv_hash_tab = reinterpret_cast<const ElfW(Word)*>(value);
        break;
      default:
        break;
    }
  }
  return d;
}

bool defines(const ElfW(Sym) & sym, const char* strtab, const char* want) noexcept {
  return sym.st_shndx != SHN_UNDEF && sym.st_value != 0 && str_equal(strtab + sym.st_name, want);
}

// Returns the symbol's index in this object, or 0 (STN_UNDEF, never a real
// definition) when it does not define it.
std::uint32_t lookup_gnu(const Dynamic& d, const char* want) noexcept {
  const std::uint32_t nbuckets = d.gnu_hash_tab[0];
  const std::uint32_t symoffset = d.gnu_hash_tab[1];
  const std::uint32_t bloom_size = d.gnu_hash_tab[2];
  if (nbuckets == 0) {
    return 0;
  }
  const ElfW(Addr)* bloom = reinterpret_cast<const ElfW(Addr)*>(&d.gnu_hash_tab[4]);
  const std::uint32_t* buckets = reinterpret_cast<const std::uint32_t*>(&bloom[bloom_size]);
  const std::uint32_t* chain = &buckets[nbuckets];

  const std::uint32_t hash = gnu_hash(want);
  std::uint32_t index = buckets[hash % nbuckets];
  if (index < symoffset) {
    return 0;
  }
  for (;; ++index) {
    const std::uint32_t entry = chain[index - symoffset];
    if (((entry ^ hash) >> 1) == 0 && defines(d.symtab[index], d.strtab, want)) {
      return index;
    }
    if ((entry & 1U) != 0) {
      return 0;
    }
  }
}

std::uint32_t lookup_sysv(const Dynamic& d, const char* want) noexcept {
  const std::uint32_t nchain = d.sysv_hash_tab[1];
  for (std::uint32_t i = 1; i < nchain; ++i) {
    if (defines(d.symtab[i], d.strtab, want)) {
      return i;
    }
  }
  return 0;
}

// The first definition of `name` in link-map order, excluding this object.
// This mirrors what the loader's global scope would have given us if we were
// not in it ourselves.
ElfW(Addr) find_elsewhere(const char* name) noexcept {
  for (const link_map* map = _r_debug.r_map; map != nullptr; map = map->l_next) {
    if (map->l_ld == _DYNAMIC) {
      continue;  // ourselves: the object whose interposition we are escaping
    }
    const Dynamic d = read_dynamic(map->l_ld, map->l_addr);
    if (d.strtab == nullptr || d.symtab == nullptr) {
      continue;
    }
    std::uint32_t index = 0;
    if (d.gnu_hash_tab != nullptr) {
      index = lookup_gnu(d, name);
    } else if (d.sysv_hash_tab != nullptr) {
      index = lookup_sysv(d, name);
    }
    if (index != 0) {
      return map->l_addr + d.symtab[index].st_value;
    }
  }
  return 0;
}

// The base version of the dl* symbols in glibc, per architecture: the version
// they carried before they moved into libc.so.6 in 2.34.
#if defined(__x86_64__)
constexpr const char* kBaseVersion = "GLIBC_2.2.5";
#elif defined(__aarch64__)
constexpr const char* kBaseVersion = "GLIBC_2.17";
#else
constexpr const char* kBaseVersion = "GLIBC_2.0";
#endif

std::atomic<DlvsymFn> g_dlvsym{nullptr};
std::atomic<DlsymFn> g_dlsym{nullptr};
std::atomic<bool> g_tried{false};

// Both lookups are pure functions of the process's link map, so two threads
// racing here compute the same answer and neither needs a lock -- which also
// means fork() can never catch this holding one.
void bootstrap() noexcept {
  const ElfW(Addr) vsym = find_elsewhere("dlvsym");
  DlvsymFn dlvsym_fn = nullptr;
  if (vsym != 0) {
    dlvsym_fn = reinterpret_cast<DlvsymFn>(vsym);
    g_dlvsym.store(dlvsym_fn, std::memory_order_release);
  }

  void* sym = nullptr;
  if (dlvsym_fn != nullptr) {
    // ADR-001's recipe, now that it cannot recurse into our own hook.
    sym = dlvsym_fn(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    if (sym == nullptr) {
      sym = dlvsym_fn(RTLD_NEXT, "dlsym", kBaseVersion);
    }
  }
  if (sym != nullptr) {
    g_dlsym.store(reinterpret_cast<DlsymFn>(sym), std::memory_order_release);
    return;
  }
  const ElfW(Addr) plain = find_elsewhere("dlsym");
  if (plain != 0) {
    g_dlsym.store(reinterpret_cast<DlsymFn>(plain), std::memory_order_release);
  }
}

void bootstrap_once() noexcept {
  if (g_tried.load(std::memory_order_acquire)) {
    return;
  }
  bootstrap();
  g_tried.store(true, std::memory_order_release);
}

}  // namespace

DlsymFn real_dlsym() noexcept {
  DlsymFn fn = g_dlsym.load(std::memory_order_acquire);
  if (fn != nullptr) {
    return fn;
  }
  bootstrap_once();
  return g_dlsym.load(std::memory_order_acquire);
}

DlvsymFn real_dlvsym() noexcept {
  DlvsymFn fn = g_dlvsym.load(std::memory_order_acquire);
  if (fn != nullptr) {
    return fn;
  }
  bootstrap_once();
  return g_dlvsym.load(std::memory_order_acquire);
}

AnyFn driver_symbol(const char* name) noexcept {
  void* handle = real_driver_handle();
  const DlsymFn lookup = real_dlsym();
  if (handle == nullptr || lookup == nullptr) {
    return nullptr;
  }
  void* address = lookup(handle, name);
  if (address == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<AnyFn>(address);
}
// NOLINTEND(performance-no-int-to-ptr)

}  // namespace tessera
