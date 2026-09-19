// The symbol table, the counters, the real-function slots, and the
// observability ABI other components read them through.

#include "cuda_api.h"
// cuda_api.h configures and includes <cuda.h>; nothing may precede it.

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tessera/shim/stats.h"

#include "shim.h"

// The arrays the trampolines address by index. Defined at namespace scope with
// C linkage and hidden visibility so the assembly can reach them with a
// PC-relative reference.
alignas(64) std::atomic<std::uint64_t> tessera_shim_counts[TESSERA_SYMBOL_COUNT];
alignas(64) std::atomic<tessera::AnyFn> tessera_shim_slots[TESSERA_SYMBOL_COUNT];

namespace tessera {

Stats g_stats;

namespace {

// Generated from the driver stub's export list, sorted by name.
#define TESSERA_SYMBOL(index, name, hooked) name,
const char* const kNames[TESSERA_SYMBOL_COUNT] = {
#include "tessera_symbols.inc"
};
#undef TESSERA_SYMBOL

#define TESSERA_SYMBOL(index, name, hooked) hooked,
const unsigned char kHooked[TESSERA_SYMBOL_COUNT] = {
#include "tessera_symbols.inc"
};
#undef TESSERA_SYMBOL

// What a trampoline jumps to when it has nothing to forward to. Both are
// entered with the caller's original arguments, which they ignore; every CUDA
// driver entry point returns a CUresult, so returning one is ABI-correct.
// I-5: a tenant gets an error, never a crash.
int unresolved_entry() {
  return static_cast<int>(CUDA_ERROR_NOT_FOUND);  // the driver has no such symbol
}

int uninitialised_entry() {
  return static_cast<int>(CUDA_ERROR_NOT_INITIALIZED);  // no driver at all
}

void zero_counters() noexcept {
  g_stats.launches.store(0, std::memory_order_relaxed);
  g_stats.graph_launches.store(0, std::memory_order_relaxed);
  g_stats.hooked_calls.store(0, std::memory_order_relaxed);
  g_stats.bypassed_lookups.store(0, std::memory_order_relaxed);
  for (std::uint64_t i = 0; i < TESSERA_SYMBOL_COUNT; ++i) {
    tessera_shim_counts[i].store(0, std::memory_order_relaxed);
  }
}

std::uint64_t forwarded_unhooked() noexcept {
  std::uint64_t total = 0;
  for (std::uint64_t i = 0; i < TESSERA_SYMBOL_COUNT; ++i) {
    if (kHooked[i] == 0) {
      total += tessera_shim_counts[i].load(std::memory_order_relaxed);
    }
  }
  return total;
}

// A small buffered writer over a file descriptor: no allocation, no stdio
// buffering to flush behind us, usable from the exit handler.
class FdWriter {
 public:
  explicit FdWriter(int fd) noexcept : fd_(fd) {}

  void put(const char* text, std::size_t length) noexcept {
    while (length > 0) {
      if (used_ == sizeof(buffer_)) {
        flush();
        if (!ok_) {
          return;
        }
      }
      const std::size_t room = sizeof(buffer_) - used_;
      const std::size_t take = length < room ? length : room;
      std::memcpy(buffer_ + used_, text, take);
      used_ += take;
      text += take;
      length -= take;
    }
  }

  void put(const char* text) noexcept { put(text, std::strlen(text)); }

  void put_u64(std::uint64_t value) noexcept {
    char digits[20];
    std::size_t n = 0;
    do {
      digits[n++] = static_cast<char>('0' + (value % 10));
      value /= 10;
    } while (value != 0);
    char reversed[20];
    for (std::size_t i = 0; i < n; ++i) {
      reversed[i] = digits[n - 1 - i];
    }
    put(reversed, n);
  }

  bool finish() noexcept {
    flush();
    return ok_;
  }

 private:
  void flush() noexcept {
    std::size_t offset = 0;
    while (ok_ && offset < used_) {
      const ssize_t written = ::write(fd_, buffer_ + offset, used_ - offset);
      if (written <= 0) {
        ok_ = false;
        return;
      }
      offset += static_cast<std::size_t>(written);
    }
    used_ = 0;
  }

  int fd_;
  std::size_t used_ = 0;
  bool ok_ = true;
  char buffer_[4096] = {};
};

}  // namespace

const char* symbol_name(unsigned index) noexcept {
  return index < TESSERA_SYMBOL_COUNT ? kNames[index] : nullptr;
}

bool symbol_is_hooked(unsigned index) noexcept {
  return index < TESSERA_SYMBOL_COUNT && kHooked[index] != 0;
}

int symbol_index(const char* name) noexcept {
  if (name == nullptr) {
    return -1;
  }
  int low = 0;
  int high = TESSERA_SYMBOL_COUNT - 1;
  while (low <= high) {
    const int mid = low + ((high - low) / 2);
    const int cmp = std::strcmp(name, kNames[mid]);
    if (cmp == 0) {
      return mid;
    }
    if (cmp < 0) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return -1;
}

AnyFn resolve_slow(unsigned index) noexcept {
  if (!ensure_init()) {
    return nullptr;
  }
  const AnyFn fn = driver_symbol(symbol_name(index));
  if (fn == nullptr) {
    TESSERA_LOG("the real driver does not export '%s'", symbol_name(index));
    return nullptr;
  }
  tessera_shim_slots[index].store(fn, std::memory_order_release);
  return fn;
}

int unavailable_result(unsigned index) noexcept {
  static_cast<void>(index);
  return static_cast<int>(real_driver_handle() == nullptr ? CUDA_ERROR_NOT_INITIALIZED
                                                          : CUDA_ERROR_NOT_FOUND);
}

void on_driver_loaded() noexcept {
  // Resolve every hooked variant now. cuGetProcAddress answers by comparing
  // the pointer the driver returned against these, so they have to be in place
  // before the first lookup; the trampolines stay lazy.
  const HookRow* rows = hook_rows();
  for (unsigned i = 0; i < hook_row_count(); ++i) {
    const AnyFn fn = driver_symbol(rows[i].name);
    if (fn != nullptr) {
      tessera_shim_slots[rows[i].index].store(fn, std::memory_order_release);
    }
  }
}

void reset_after_fork() noexcept {
  // The child is a new process: its counters start at zero, and the parent's
  // are untouched because they live in the parent's copy of these pages.
  zero_counters();
}

}  // namespace tessera

extern "C" {

tessera::AnyFn tessera_shim_resolve_tramp(unsigned index) {
  const tessera::AnyFn fn = tessera::resolve_slow(index);
  if (fn != nullptr) {
    return fn;
  }
  // A function pointer of one signature reinterpreted as another: the
  // trampoline is about to jump to it with the caller's arguments, which the
  // stub ignores. Which stub says why, in the only channel a trampoline has:
  // its return value.
  return reinterpret_cast<tessera::AnyFn>(tessera::real_driver_handle() == nullptr
                                              ? &tessera::uninitialised_entry
                                              : &tessera::unresolved_entry);
}

int tessera_abi_version(void) {
  return TESSERA_ABI_VERSION;
}

int tessera_get_stats_v1(tessera_stats_v1* out) {
  if (out == nullptr) {
    return -1;
  }
  out->launches = tessera::g_stats.launches.load(std::memory_order_relaxed);
  out->graph_launches = tessera::g_stats.graph_launches.load(std::memory_order_relaxed);
  out->hooked_calls = tessera::g_stats.hooked_calls.load(std::memory_order_relaxed);
  out->forwarded_unhooked = tessera::forwarded_unhooked();
  out->bypassed_lookups = tessera::g_stats.bypassed_lookups.load(std::memory_order_relaxed);
  return 0;
}

uint64_t tessera_get_symbol_count(const char* exported_symbol_name) {
  const int index = tessera::symbol_index(exported_symbol_name);
  if (index < 0) {
    return 0;
  }
  return tessera_shim_counts[index].load(std::memory_order_relaxed);
}

void tessera_reset_stats(void) {
  tessera::zero_counters();
}

int tessera_dump_stats(const char* path) {
  if (path == nullptr) {
    return -1;
  }
  const int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    return -1;
  }
  tessera_stats_v1 stats;
  std::memset(&stats, 0, sizeof(stats));
  tessera_get_stats_v1(&stats);

  tessera::FdWriter out(fd);
  out.put("{\"tessera_abi\":");
  out.put_u64(TESSERA_ABI_VERSION);
  out.put(",\"pid\":");
  out.put_u64(static_cast<std::uint64_t>(::getpid()));
  out.put(",\"launches\":");
  out.put_u64(stats.launches);
  out.put(",\"graph_launches\":");
  out.put_u64(stats.graph_launches);
  out.put(",\"hooked_calls\":");
  out.put_u64(stats.hooked_calls);
  out.put(",\"forwarded_unhooked\":");
  out.put_u64(stats.forwarded_unhooked);
  out.put(",\"bypassed_lookups\":");
  out.put_u64(stats.bypassed_lookups);
  out.put(",\"symbols\":{");
  bool first = true;
  for (unsigned i = 0; i < TESSERA_SYMBOL_COUNT; ++i) {
    const std::uint64_t count = tessera_shim_counts[i].load(std::memory_order_relaxed);
    if (count == 0) {
      continue;
    }
    if (!first) {
      out.put(",");
    }
    first = false;
    out.put("\"");
    out.put(tessera::symbol_name(i));
    out.put("\":");
    out.put_u64(count);
  }
  out.put("}}\n");
  const bool written = out.finish();
  const bool closed = ::close(fd) == 0;
  return (written && closed) ? 0 : -1;
}

}  // extern "C"
