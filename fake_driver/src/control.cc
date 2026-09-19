// The tf_* control API declared in tessera_fake/control.h.
//
// These are the only non-cu* symbols libcuda_fake exports. They never throw:
// like the driver entry points, they cross a C ABI boundary.

#include "tessera_fake/control.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "counters.h"
#include "sim.h"

namespace {

using tf::Function;
using tf::Sim;
using tf::Stream;

// A tf_* function that returns void has no channel to report a failure on, and
// none of them may let an exception cross the C ABI. Rather than swallow the
// failure, they count it: tf_control_failure_count() makes it visible, and a
// test can assert it stayed at zero.
std::atomic<uint64_t> g_control_failures{0};

template <class F>
void guarded_void(F&& f) noexcept {
  try {
    f();
  } catch (...) {
    g_control_failures.fetch_add(1, std::memory_order_relaxed);
  }
}

template <class T, class F>
T guarded_value(T fallback, F&& f) noexcept {
  try {
    return f();
  } catch (...) {
    g_control_failures.fetch_add(1, std::memory_order_relaxed);
    return fallback;
  }
}

}  // namespace

uint64_t tf_control_failure_count(void) {
  return g_control_failures.load(std::memory_order_relaxed);
}

// ===========================================================================
// Lifecycle
// ===========================================================================

void tf_reset(void) {
  guarded_void([] { Sim::get().reset(); });
}

// ===========================================================================
// Counters and the stats file
// ===========================================================================

uint64_t tf_symbol_count(const char* exported_symbol_name) {
  return guarded_value<uint64_t>(0, [&]() -> uint64_t {
    const int64_t i = tf::symbol_index(exported_symbol_name);
    if (i < 0) {
      return 0;
    }
    return tf::g_counts[i].load(std::memory_order_relaxed);
  });
}

int tf_dump_stats(const char* path) {
  return guarded_value<int>(-1, [&]() -> int {
    if (path == nullptr || path[0] == '\0') {
      errno = EINVAL;
      return -1;
    }
    // Built in memory first, so the file is written with one checked call.
    std::string text = "{";
    const uint32_t n = tf::symbol_table_size();
    const tf::SymbolRow* rows = tf::symbol_table();
    bool first = true;
    for (uint32_t i = 0; i < n; ++i) {
      const uint64_t count = tf::g_counts[i].load(std::memory_order_relaxed);
      if (count == 0) {
        continue;
      }
      if (!first) {
        text += ',';
      }
      text += '"';
      text += rows[i].exported;
      text += "\":";
      text += std::to_string(count);
      first = false;
    }
    text += "}\n";

    std::FILE* f = std::fopen(path, "w");
    if (f == nullptr) {
      return -1;
    }
    const bool written = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    const bool flushed = std::fflush(f) == 0;
    const bool closed = std::fclose(f) == 0;
    return written && flushed && closed ? 0 : -1;
  });
}

// ===========================================================================
// Kernels
// ===========================================================================

CUfunction tf_register_kernel(const char* name, void (*body)(void** kernel_params),
                              uint64_t duration_us) {
  return guarded_value<CUfunction>(nullptr, [&]() -> CUfunction {
    if (name == nullptr) {
      return nullptr;
    }
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    auto it = sim.functions_by_name.find(name);
    if (it != sim.functions_by_name.end()) {
      Function* existing = sim.functions.get(it->second);
      if (existing != nullptr) {
        existing->body = body;
        existing->duration_us = duration_us;
        return static_cast<CUfunction>(it->second);
      }
      sim.functions_by_name.erase(it);
    }
    auto f = std::make_unique<Function>();
    f->name = name;
    f->body = body;
    f->duration_us = duration_us;
    void* h = sim.functions.create(std::move(f));
    sim.functions_by_name.emplace(name, h);
    return static_cast<CUfunction>(h);
  });
}

void tf_set_kernel_duration_us(CUfunction f, uint64_t us) {
  guarded_void([&] {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    Function* fn = sim.functions.get(f);
    if (fn != nullptr) {
      fn->duration_us = us;
    }
  });
}

void tf_set_kernel_param_count(CUfunction f, uint32_t count) {
  guarded_void([&] {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    Function* fn = sim.functions.get(f);
    if (fn != nullptr) {
      fn->param_count = count;
    }
  });
}

void tf_set_default_kernel_duration_us(uint64_t us) {
  guarded_void([&] {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    sim.default_kernel_duration_us = us;
  });
}

// ===========================================================================
// The clock
// ===========================================================================

void tf_set_time_mode(int manual) {
  guarded_void([&] {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    sim.set_manual_locked(manual != 0);
    sim.cv_work.notify_all();
  });
}

void tf_advance_time_us(uint64_t us) {
  guarded_void([&] {
    Sim& sim = Sim::get();
    std::unique_lock<std::mutex> lk(sim.mu);
    sim.clock_base_us += us;
    if (sim.manual_time) {
      sim.quiesce_locked(lk);
    } else {
      sim.cv_work.notify_all();
    }
  });
}

uint64_t tf_now_us(void) {
  return guarded_value<uint64_t>(0, [] {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    return sim.now_us_locked();
  });
}

// ===========================================================================
// Error injection
// ===========================================================================

void tf_set_result(const char* exported_symbol_name, CUresult r) {
  guarded_void([&] {
    const int64_t i = tf::symbol_index(exported_symbol_name);
    if (i < 0) {
      return;
    }
    // CUDA_SUCCESS means "stop forcing": an entry point made to return success
    // without doing its work would leave its out-parameters uninitialised.
    const int32_t slot = r == CUDA_SUCCESS ? 0 : static_cast<int32_t>(r) + 1;
    tf::g_forced[i].store(slot, std::memory_order_relaxed);
  });
}

void tf_clear_results(void) {
  for (uint32_t i = 0; i < tf::TF_SYM_COUNT; ++i) {
    tf::g_forced[i].store(0, std::memory_order_relaxed);
  }
}

// ===========================================================================
// Fork
// ===========================================================================

void tf_set_fork_mode(int emulate_real) {
  guarded_void([&] {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    tf::g_fork_emulate_real.store(emulate_real != 0 ? 1 : 0, std::memory_order_relaxed);
  });
}

// ===========================================================================
// Streams
// ===========================================================================

uint64_t tf_stream_launch_count(CUstream s) {
  return guarded_value<uint64_t>(0, [&]() -> uint64_t {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    void* h = s;
    const uintptr_t raw = reinterpret_cast<uintptr_t>(s);
    if (raw == 0 || raw == tf::kStreamLegacyHandle) {
      h = sim.legacy_stream_of_locked(::tf::current_context());
    } else if (raw == tf::kStreamPerThreadHandle) {
      h = sim.per_thread_stream_of_locked(::tf::current_context());
    }
    const Stream* st = sim.streams.get(h);
    return st == nullptr ? 0 : st->launches;
  });
}

int tf_last_stream_was_per_thread_default(void) {
  return tf::last_stream_was_per_thread() ? 1 : 0;
}

CUstream tf_legacy_default_stream(void) {
  return guarded_value<CUstream>(nullptr, []() -> CUstream {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    return static_cast<CUstream>(sim.legacy_stream_of_locked(::tf::current_context()));
  });
}

CUstream tf_per_thread_default_stream(void) {
  return guarded_value<CUstream>(nullptr, []() -> CUstream {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    return static_cast<CUstream>(sim.per_thread_stream_of_locked(::tf::current_context()));
  });
}

// ===========================================================================
// The cuGetProcAddress table
// ===========================================================================

uint64_t tf_symbol_table_size(void) {
  return tf::symbol_table_size();
}

const char* tf_symbol_name_at(uint64_t i) {
  if (i >= tf::symbol_table_size()) {
    return nullptr;
  }
  return tf::symbol_table()[i].exported;
}

const char* tf_symbol_base_name_at(uint64_t i) {
  if (i >= tf::symbol_table_size()) {
    return nullptr;
  }
  return tf::symbol_table()[i].base;
}

int tf_symbol_version_at(uint64_t i) {
  if (i >= tf::symbol_table_size()) {
    return -1;
  }
  return tf::symbol_table()[i].version;
}

int tf_symbol_stream_kind_at(uint64_t i) {
  if (i >= tf::symbol_table_size()) {
    return -1;
  }
  switch (tf::symbol_table()[i].stream_kind) {
    case tf::TF_STREAM_LEGACY:
      return TF_STREAM_KIND_LEGACY;
    case tf::TF_STREAM_PER_THREAD:
      return TF_STREAM_KIND_PER_THREAD;
    case tf::TF_STREAM_NONE:
      break;
  }
  return TF_STREAM_KIND_NONE;
}

int64_t tf_symbol_index(const char* exported_symbol_name) {
  return tf::symbol_index(exported_symbol_name);
}

tf_fn_ptr tf_symbol_address(const char* exported_symbol_name) {
  const int64_t i = tf::symbol_index(exported_symbol_name);
  if (i < 0) {
    return nullptr;
  }
  return tf::symbol_table()[i].fn;
}

// ===========================================================================
// Device properties
// ===========================================================================

void tf_set_driver_version(int version) {
  guarded_void([&] {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    sim.driver_version = version;
  });
}

void tf_set_device_total_mem(uint64_t bytes) {
  guarded_void([&] {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    sim.device.total_mem = bytes;
  });
}

void tf_set_device_sm_count(int sm_count) {
  guarded_void([&] {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    sim.device.sm_count = sm_count;
  });
}
