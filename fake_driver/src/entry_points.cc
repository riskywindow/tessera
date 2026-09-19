// Every cu* symbol libcuda_fake exports.
//
// The legacy, _v2/_v3/_v4 and _ptsz/_ptds forms of an operation are SEPARATE
// exported symbols with separately observable behaviour, because that is what
// the shim has to cope with: a _ptsz entry point resolves a NULL stream to the
// calling thread's per-thread default stream, its legacy twin resolves the same
// argument to the context's legacy default stream, and both are counted apart.
//
// __CUDA_API_VERSION_INTERNAL makes cuda.h declare all of those names at once
// instead of hiding them behind its usual renaming macros, which is how the
// real driver's own sources see the header. __CUDA_API_PUSH_VISIBILITY_DEFAULT
// makes it wrap the declarations in `#pragma GCC visibility push(default)`, so
// these definitions are exported even though the build compiles with
// -fvisibility=hidden. __CUDA_API_VERSION_INTERNAL_ODR keeps the enumerations
// identical to the ones every other translation unit sees.

// clang-format off
// These three are cuda.h's own configuration macros, spelled the way NVIDIA
// spells them, and cuda.h has to be included here, under them, before anything
// else can drag it in. Neither the include sorter nor the reserved-identifier
// rule has a say over another project's header contract.
// NOLINTBEGIN(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
#define __CUDA_API_VERSION_INTERNAL 1
#define __CUDA_API_VERSION_INTERNAL_ODR 1
#define __CUDA_API_PUSH_VISIBILITY_DEFAULT 1
// NOLINTEND(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
#include <cuda.h>
// clang-format on

#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <new>
#include <string>
#include <utility>

#include <sys/mman.h>

#include "counters.h"
#include "sim.h"
#include "tessera_fake/control.h"

namespace tf {

std::atomic<uint64_t> g_counts[TF_SYM_COUNT];
std::atomic<int32_t> g_forced[TF_SYM_COUNT];
std::atomic<int> g_fork_emulate_real{0};
std::atomic<int> g_in_forked_child{0};

}  // namespace tf

namespace {

using tf::Allocation;
using tf::Context;
using tf::Event;
using tf::Function;
using tf::Graph;
using tf::GraphExec;
using tf::MemHandleObj;
using tf::Module;
using tf::Op;
using tf::Reservation;
using tf::Sim;
using tf::Stream;

// No exception may cross a C ABI boundary. Every entry point body runs inside
// this, so a bad_alloc from a container or a system_error from starting the
// worker thread becomes a CUresult instead of unwinding into the caller.
template <class F>
CUresult guarded(F&& f) noexcept {
  try {
    return f();
  } catch (const std::bad_alloc&) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}

// Take the simulation lock, and refuse the call if cuInit has not run.
//
// TF_LOCK's lock is const: use it in an entry point that only reads and writes
// simulation state. TF_WAIT_LOCK's is not, because an entry point that waits
// for the worker thread hands the lock to a condition variable.
#define TF_LOCK()                                \
  ::tf::Sim& sim = ::tf::Sim::get();             \
  const std::unique_lock<std::mutex> lk(sim.mu); \
  if (!sim.initialized)                          \
  return CUDA_ERROR_NOT_INITIALIZED

#define TF_WAIT_LOCK()                     \
  ::tf::Sim& sim = ::tf::Sim::get();       \
  std::unique_lock<std::mutex> lk(sim.mu); \
  if (!sim.initialized)                    \
  return CUDA_ERROR_NOT_INITIALIZED

// As above, and also bind the calling thread's current context. The macro
// parameter names a variable being declared, so it cannot be parenthesised.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define TF_LOCK_CTX(ctxvar)                \
  TF_LOCK();                               \
  void* ctxvar = ::tf::current_context();  \
  if (sim.contexts.get(ctxvar) == nullptr) \
  return CUDA_ERROR_INVALID_CONTEXT

#define TF_WAIT_LOCK_CTX(ctxvar)           \
  TF_WAIT_LOCK();                          \
  void* ctxvar = ::tf::current_context();  \
  if (sim.contexts.get(ctxvar) == nullptr) \
  return CUDA_ERROR_INVALID_CONTEXT
// NOLINTEND(bugprone-macro-parentheses)

void* host_ptr(CUdeviceptr p) {
  return ::tf::address_to_pointer(p);
}

// cuGetProcAddress hands the caller a function pointer through a void*. POSIX
// requires that to work, but the conversion between the two is not portable
// enough for -Wpedantic, so the bytes are copied rather than cast.
void store_function_pointer(void** out, void (*fn)(void)) {
  // NOLINTNEXTLINE(bugprone-bitwise-pointer-cast,bugprone-multi-level-implicit-pointer-conversion)
  std::memcpy(static_cast<void*>(out), static_cast<const void*>(&fn), sizeof(fn));
}

CUdeviceptr device_ptr(const void* p) {
  return static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(p));
}

// --- version-1 device pointers -------------------------------------------
// See Sim::v1_alias. Returns 0 for a token this process never handed out.
uintptr_t v1_resolve(Sim& sim, unsigned int token) {
  auto it = sim.v1_alias.find(token);
  return it == sim.v1_alias.end() ? 0 : it->second;
}

unsigned int v1_register(Sim& sim, void* host) {
  const unsigned int token = sim.next_v1_token;
  sim.next_v1_token += 4096;
  sim.v1_alias.emplace(token, reinterpret_cast<uintptr_t>(host));
  return token;
}

// --- common operation plumbing -------------------------------------------

// Resolves the stream argument of an entry point. `per_thread` is true for the
// _ptsz / _ptds entry points.
CUresult get_stream(Sim& sim, void* ctx, CUstream arg, bool per_thread, Stream** out,
                    void** handle) {
  Stream* s = sim.resolve_stream_locked(ctx, arg, per_thread, handle);
  if (s == nullptr) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  *out = s;
  return CUDA_SUCCESS;
}

// Queues an operation, or records it into the graph being captured on this
// stream. `blocking` drains the stream afterwards, which is what makes the
// synchronous copy and fill entry points synchronous.
CUresult submit(Sim& sim, std::unique_lock<std::mutex>& lk, Stream& s, void* stream_handle, Op op,
                bool blocking) {
  if (s.capture == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
    if (blocking) {
      return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    }
    Graph* g = sim.graphs.get(s.capture_graph);
    if (g == nullptr) {
      return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    }
    // A captured launch has not run: it is counted when the graph is replayed.
    g->nodes.push_back(std::move(op));
    return CUDA_SUCCESS;
  }
  if (op.kind == Op::Kind::kKernel) {
    ++s.launches;
  }
  sim.enqueue_locked(s, std::move(op));
  if (blocking) {
    sim.drain_stream_locked(lk, stream_handle);
  }
  return CUDA_SUCCESS;
}

// The device side of a copy or fill must be inside a live allocation, the way a
// real driver would reject a stray pointer.
bool device_range_ok(Sim& sim, CUdeviceptr p, size_t bytes) {
  return sim.range_is_device_locked(static_cast<uintptr_t>(p), bytes);
}

CUresult copy_impl(Sim& sim, std::unique_lock<std::mutex>& lk, void* ctx, CUstream arg,
                   bool per_thread, void* dst, const void* src, size_t bytes, bool dst_is_device,
                   bool src_is_device, bool blocking) {
  if (bytes == 0) {
    return CUDA_SUCCESS;
  }
  if (dst == nullptr || src == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dst_is_device && !device_range_ok(sim, device_ptr(dst), bytes)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (src_is_device && !device_range_ok(sim, device_ptr(src), bytes)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, arg, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  Op op;
  op.kind = Op::Kind::kCopy;
  op.dst = dst;
  op.src = src;
  op.bytes = bytes;
  return submit(sim, lk, *s, handle, std::move(op), blocking);
}

CUresult memset_impl(Sim& sim, std::unique_lock<std::mutex>& lk, void* ctx, CUstream arg,
                     bool per_thread, CUdeviceptr dst, uint32_t pattern, uint8_t width,
                     size_t count, bool blocking) {
  if (count == 0) {
    return CUDA_SUCCESS;
  }
  if (!device_range_ok(sim, dst, count * width)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, arg, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  Op op;
  op.kind = Op::Kind::kMemset;
  op.dst = host_ptr(dst);
  op.pattern = pattern;
  op.pattern_width = width;
  op.count = count;
  return submit(sim, lk, *s, handle, std::move(op), blocking);
}

CUresult launch_impl(Sim& sim, std::unique_lock<std::mutex>& lk, void* ctx, CUfunction f,
                     CUstream arg, bool per_thread, void** kernel_params) {
  Function* fn = sim.functions.get(f);
  if (fn == nullptr) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, arg, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  Op op;
  op.kind = Op::Kind::kKernel;
  op.duration_us = fn->duration_us;
  op.body = fn->body;
  if (fn->param_count != 0 && kernel_params != nullptr) {
    op.params.assign(kernel_params, kernel_params + fn->param_count);
    op.has_params = true;
  }
  return submit(sim, lk, *s, handle, std::move(op), false);
}

CUresult record_event_impl(Sim& sim, void* ctx, CUevent hEvent, CUstream arg, bool per_thread) {
  Event* e = sim.events.get(hEvent);
  if (e == nullptr) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, arg, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  if (s->capture == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
    // A recorded event inside a capture has no timing of its own; the fake
    // records it as an empty capture node so the graph stays replayable.
    return CUDA_SUCCESS;
  }
  e->recorded = true;
  e->complete = false;
  Op op;
  op.kind = Op::Kind::kEventRecord;
  op.event = hEvent;
  e->time_us = sim.enqueue_locked(*s, std::move(op));
  return CUDA_SUCCESS;
}

CUresult begin_capture_impl(Sim& sim, void* ctx, CUstream arg, bool per_thread,
                            CUstreamCaptureMode mode) {
  static_cast<void>(mode);
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, arg, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  if (s->legacy_default) {
    return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
  }
  if (s->capture == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
    return CUDA_ERROR_ILLEGAL_STATE;
  }
  s->capture_graph = sim.graphs.create(std::make_unique<Graph>());
  s->capture = CU_STREAM_CAPTURE_STATUS_ACTIVE;
  s->capture_id = sim.next_capture_id++;
  return CUDA_SUCCESS;
}

CUresult end_capture_impl(Sim& sim, void* ctx, CUstream arg, bool per_thread, CUgraph* phGraph) {
  if (phGraph == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, arg, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  if (s->capture != CU_STREAM_CAPTURE_STATUS_ACTIVE) {
    return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
  }
  *phGraph = static_cast<CUgraph>(s->capture_graph);
  s->capture = CU_STREAM_CAPTURE_STATUS_NONE;
  s->capture_graph = nullptr;
  s->capture_id = 0;
  return CUDA_SUCCESS;
}

CUresult is_capturing_impl(Sim& sim, void* ctx, CUstream arg, bool per_thread,
                           CUstreamCaptureStatus* out) {
  if (out == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, arg, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  *out = s->capture;
  return CUDA_SUCCESS;
}

CUresult graph_launch_impl(Sim& sim, void* ctx, CUgraphExec hGraphExec, CUstream arg,
                           bool per_thread) {
  GraphExec* ge = sim.graph_execs.get(hGraphExec);
  if (ge == nullptr) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, arg, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  for (const Op& node : ge->nodes) {
    Op op = node;
    if (op.kind == Op::Kind::kKernel) {
      ++s->launches;
    }
    sim.enqueue_locked(*s, std::move(op));
  }
  return CUDA_SUCCESS;
}

CUresult create_context(Sim& sim, CUcontext* pctx, unsigned int flags, CUdevice dev, bool primary) {
  if (pctx == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  auto c = std::make_unique<Context>();
  c->dev = dev;
  c->flags = flags;
  c->primary = primary;
  void* h = sim.contexts.create(std::move(c));
  sim.legacy_stream_of_locked(h);
  *pctx = static_cast<CUcontext>(h);
  return CUDA_SUCCESS;
}

CUresult destroy_context(Sim& sim, std::unique_lock<std::mutex>& lk, CUcontext ctx) {
  Context* c = sim.contexts.get(ctx);
  if (c == nullptr) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  sim.drain_context_locked(lk, ctx);
  std::vector<void*> victims;
  sim.streams.for_each([&](void* h, Stream& s) {
    if (s.ctx == ctx) {
      victims.push_back(h);
    }
  });
  for (void* h : victims) {
    sim.streams.destroy(h);
  }
  if (sim.primary_context == static_cast<void*>(ctx)) {
    sim.primary_context = nullptr;
  }
  sim.contexts.destroy(ctx);
  ::tf::forget_context(ctx);
  return CUDA_SUCCESS;
}

// --- the cuGetProcAddress table -------------------------------------------

CUresult resolve_proc(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags,
                      CUdriverProcAddressQueryResult* status) {
  const auto fail = [&](CUdriverProcAddressQueryResult s, CUresult r) {
    if (status != nullptr) {
      *status = s;
    }
    if (pfn != nullptr) {
      *pfn = nullptr;
    }
    return r;
  };
  if (symbol == nullptr || pfn == nullptr) {
    return fail(CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND, CUDA_ERROR_INVALID_VALUE);
  }
  const cuuint64_t stream_bits =
      flags & static_cast<cuuint64_t>(CU_GET_PROC_ADDRESS_LEGACY_STREAM |
                                      CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM);
  if (stream_bits == static_cast<cuuint64_t>(CU_GET_PROC_ADDRESS_LEGACY_STREAM |
                                             CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)) {
    return fail(CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND, CUDA_ERROR_INVALID_VALUE);
  }
  // CU_GET_PROC_ADDRESS_DEFAULT means "legacy" at the driver. cuda.h is what
  // turns a per-thread-default-stream compilation into the explicit flag.
  const bool want_per_thread =
      (stream_bits & static_cast<cuuint64_t>(CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)) != 0;

  int ceiling = 0;
  {
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    ceiling = sim.driver_version;
  }
  const int effective = cudaVersion < ceiling ? cudaVersion : ceiling;

  const tf::SymbolRow* rows = tf::symbol_table();
  const uint32_t n = tf::symbol_table_size();
  bool base_exists = false;
  bool variant_exists = false;
  const tf::SymbolRow* best = nullptr;
  for (uint32_t i = 0; i < n; ++i) {
    const tf::SymbolRow& row = rows[i];
    if (std::strcmp(row.base, symbol) != 0) {
      continue;
    }
    base_exists = true;
    const bool variant_ok = row.stream_kind == tf::TF_STREAM_NONE ||
                            (want_per_thread ? row.stream_kind == tf::TF_STREAM_PER_THREAD
                                             : row.stream_kind == tf::TF_STREAM_LEGACY);
    if (!variant_ok) {
      continue;
    }
    variant_exists = true;
    if (row.version > effective) {
      continue;
    }
    if (best == nullptr || row.version > best->version) {
      best = &row;
    }
  }
  if (!base_exists || !variant_exists) {
    return fail(CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND, CUDA_ERROR_NOT_FOUND);
  }
  if (best == nullptr) {
    return fail(CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT, CUDA_ERROR_NOT_FOUND);
  }
  store_function_pointer(pfn, best->fn);
  if (status != nullptr) {
    *status = CU_GET_PROC_ADDRESS_SUCCESS;
  }
  return CUDA_SUCCESS;
}

struct ErrorName {
  CUresult code;
  const char* name;
  const char* text;
};

const ErrorName kErrors[] = {
    {CUDA_SUCCESS, "CUDA_SUCCESS", "no error"},
    {CUDA_ERROR_INVALID_VALUE, "CUDA_ERROR_INVALID_VALUE", "invalid argument"},
    {CUDA_ERROR_OUT_OF_MEMORY, "CUDA_ERROR_OUT_OF_MEMORY", "out of memory"},
    {CUDA_ERROR_NOT_INITIALIZED, "CUDA_ERROR_NOT_INITIALIZED", "initialization error"},
    {CUDA_ERROR_DEINITIALIZED, "CUDA_ERROR_DEINITIALIZED", "driver shutting down"},
    {CUDA_ERROR_NO_DEVICE, "CUDA_ERROR_NO_DEVICE", "no CUDA-capable device is detected"},
    {CUDA_ERROR_INVALID_DEVICE, "CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal"},
    {CUDA_ERROR_INVALID_CONTEXT, "CUDA_ERROR_INVALID_CONTEXT", "invalid device context"},
    {CUDA_ERROR_INVALID_HANDLE, "CUDA_ERROR_INVALID_HANDLE", "invalid resource handle"},
    {CUDA_ERROR_NOT_FOUND, "CUDA_ERROR_NOT_FOUND", "named symbol not found"},
    {CUDA_ERROR_NOT_READY, "CUDA_ERROR_NOT_READY", "device not ready"},
    {CUDA_ERROR_NOT_SUPPORTED, "CUDA_ERROR_NOT_SUPPORTED", "operation not supported"},
    {CUDA_ERROR_NOT_PERMITTED, "CUDA_ERROR_NOT_PERMITTED", "operation not permitted"},
    {CUDA_ERROR_ILLEGAL_STATE, "CUDA_ERROR_ILLEGAL_STATE",
     "the operation is not permitted "
     "in the current state"},
    {CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED",
     "operation not permitted when stream is capturing"},
    {CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED",
     "operation failed due to a previous error during capture"},
    {CUDA_ERROR_UNKNOWN, "CUDA_ERROR_UNKNOWN", "unknown error"},
};

}  // namespace

// ===========================================================================
// Initialisation, version, error strings
// ===========================================================================

CUresult cuInit(unsigned int Flags) {
  TF_ENTER(cuInit);
  return guarded([&]() -> CUresult {
    static_cast<void>(Flags);
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    sim.initialized = true;
    return CUDA_SUCCESS;
  });
}

CUresult cuDriverGetVersion(int* driverVersion) {
  TF_ENTER(cuDriverGetVersion);
  return guarded([&]() -> CUresult {
    if (driverVersion == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    Sim& sim = Sim::get();
    const std::unique_lock<std::mutex> lk(sim.mu);
    *driverVersion = sim.driver_version;
    return CUDA_SUCCESS;
  });
}

CUresult cuGetErrorName(CUresult error, const char** pStr) {
  TF_ENTER(cuGetErrorName);
  return guarded([&]() -> CUresult {
    if (pStr == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    for (const ErrorName& e : kErrors) {
      if (e.code == error) {
        *pStr = e.name;
        return CUDA_SUCCESS;
      }
    }
    *pStr = nullptr;
    return CUDA_ERROR_INVALID_VALUE;
  });
}

CUresult cuGetErrorString(CUresult error, const char** pStr) {
  TF_ENTER(cuGetErrorString);
  return guarded([&]() -> CUresult {
    if (pStr == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    for (const ErrorName& e : kErrors) {
      if (e.code == error) {
        *pStr = e.text;
        return CUDA_SUCCESS;
      }
    }
    *pStr = nullptr;
    return CUDA_ERROR_INVALID_VALUE;
  });
}

// ===========================================================================
// Devices
// ===========================================================================

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
  TF_ENTER(cuDeviceGet);
  return guarded([&]() -> CUresult {
    if (device == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    if (ordinal != 0) {
      return CUDA_ERROR_INVALID_DEVICE;
    }
    *device = 0;
    return CUDA_SUCCESS;
  });
}

CUresult cuDeviceGetCount(int* count) {
  TF_ENTER(cuDeviceGetCount);
  return guarded([&]() -> CUresult {
    if (count == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    *count = 1;
    return CUDA_SUCCESS;
  });
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
  TF_ENTER(cuDeviceGetName);
  return guarded([&]() -> CUresult {
    if (name == nullptr || len <= 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    if (dev != 0) {
      return CUDA_ERROR_INVALID_DEVICE;
    }
    const std::string& n = sim.device.name;
    const size_t copy = std::min(n.size(), static_cast<size_t>(len) - 1);
    std::memcpy(name, n.data(), copy);
    name[copy] = '\0';
    return CUDA_SUCCESS;
  });
}

CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev) {
  TF_ENTER(cuDeviceGetUuid);
  return guarded([&]() -> CUresult {
    if (uuid == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    if (dev != 0) {
      return CUDA_ERROR_INVALID_DEVICE;
    }
    std::memset(uuid->bytes, 0, sizeof(uuid->bytes));
    std::memcpy(uuid->bytes, "TESSERAFAKEDEV0", 15);
    return CUDA_SUCCESS;
  });
}

CUresult cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice dev) {
  TF_ENTER(cuDeviceGetUuid_v2);
  return guarded([&]() -> CUresult {
    if (uuid == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    if (dev != 0) {
      return CUDA_ERROR_INVALID_DEVICE;
    }
    std::memset(uuid->bytes, 0, sizeof(uuid->bytes));
    std::memcpy(uuid->bytes, "TESSERAFAKEDEV0", 15);
    return CUDA_SUCCESS;
  });
}

CUresult cuDeviceTotalMem(unsigned int* bytes, CUdevice dev) {
  TF_ENTER(cuDeviceTotalMem);
  return guarded([&]() -> CUresult {
    if (bytes == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    if (dev != 0) {
      return CUDA_ERROR_INVALID_DEVICE;
    }
    *bytes = static_cast<unsigned int>(std::min<uint64_t>(sim.device.total_mem, UINT32_MAX));
    return CUDA_SUCCESS;
  });
}

CUresult cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev) {
  TF_ENTER(cuDeviceTotalMem_v2);
  return guarded([&]() -> CUresult {
    if (bytes == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    if (dev != 0) {
      return CUDA_ERROR_INVALID_DEVICE;
    }
    *bytes = static_cast<size_t>(sim.device.total_mem);
    return CUDA_SUCCESS;
  });
}

CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev) {
  TF_ENTER(cuDeviceComputeCapability);
  return guarded([&]() -> CUresult {
    if (major == nullptr || minor == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    if (dev != 0) {
      return CUDA_ERROR_INVALID_DEVICE;
    }
    *major = sim.device.cc_major;
    *minor = sim.device.cc_minor;
    return CUDA_SUCCESS;
  });
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
  TF_ENTER(cuDeviceGetAttribute);
  return guarded([&]() -> CUresult {
    if (pi == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    if (dev != 0) {
      return CUDA_ERROR_INVALID_DEVICE;
    }
    switch (attrib) {
      case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
        *pi = sim.device.sm_count;
        break;
      case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
        *pi = sim.device.cc_major;
        break;
      case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
        *pi = sim.device.cc_minor;
        break;
      case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
        *pi = 32;
        break;
      case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
        *pi = 1024;
        break;
      case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
        *pi = 48 * 1024;
        break;
      case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
        *pi = 1'500'000;
        break;
      case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
        *pi = 2;
        break;
      case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
      case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
      case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY:
      case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED:
      case CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH:
      case CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED:
      case CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED:
        *pi = 1;
        break;
      default:
        // Attributes the simulation has no opinion about read as zero rather
        // than failing, so a caller enumerating the whole enum still works.
        *pi = 0;
        break;
    }
    return CUDA_SUCCESS;
  });
}

// ===========================================================================
// Contexts
// ===========================================================================

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  TF_ENTER(cuDevicePrimaryCtxRetain);
  return guarded([&]() -> CUresult {
    if (pctx == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    if (dev != 0) {
      return CUDA_ERROR_INVALID_DEVICE;
    }
    if (sim.primary_context == nullptr) {
      CUcontext ctx = nullptr;
      const CUresult r = create_context(sim, &ctx, 0, dev, true);
      if (r != CUDA_SUCCESS) {
        return r;
      }
      sim.primary_context = ctx;
    }
    Context* c = sim.contexts.get(sim.primary_context);
    if (c == nullptr) {
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    ++c->primary_refs;
    *pctx = static_cast<CUcontext>(sim.primary_context);
    return CUDA_SUCCESS;
  });
}

namespace {
CUresult primary_ctx_release(Sim& sim, std::unique_lock<std::mutex>& lk, CUdevice dev) {
  if (dev != 0) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  Context* c = sim.contexts.get(sim.primary_context);
  if (c == nullptr) {
    return CUDA_SUCCESS;
  }
  if (c->primary_refs > 0) {
    --c->primary_refs;
  }
  if (c->primary_refs == 0) {
    return destroy_context(sim, lk, static_cast<CUcontext>(sim.primary_context));
  }
  return CUDA_SUCCESS;
}
}  // namespace

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
  TF_ENTER(cuDevicePrimaryCtxRelease);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK();
    return primary_ctx_release(sim, lk, dev);
  });
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  TF_ENTER(cuDevicePrimaryCtxRelease_v2);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK();
    return primary_ctx_release(sim, lk, dev);
  });
}

CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  TF_ENTER(cuCtxCreate);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    const CUresult r = create_context(sim, pctx, flags, dev, false);
    if (r == CUDA_SUCCESS) {
      ::tf::push_current(*pctx);
    }
    return r;
  });
}

CUresult cuCtxCreate_v2(CUcontext* pctx, unsigned int flags, CUdevice dev) {
  TF_ENTER(cuCtxCreate_v2);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    const CUresult r = create_context(sim, pctx, flags, dev, false);
    if (r == CUDA_SUCCESS) {
      ::tf::push_current(*pctx);
    }
    return r;
  });
}

CUresult cuCtxCreate_v3(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams,
                        unsigned int flags, CUdevice dev) {
  TF_ENTER(cuCtxCreate_v3);
  return guarded([&]() -> CUresult {
    if (numParams < 0 || (numParams > 0 && paramsArray == nullptr)) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    const CUresult r = create_context(sim, pctx, flags, dev, false);
    if (r == CUDA_SUCCESS) {
      ::tf::push_current(*pctx);
    }
    return r;
  });
}

CUresult cuCtxCreate_v4(CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags,
                        CUdevice dev) {
  TF_ENTER(cuCtxCreate_v4);
  return guarded([&]() -> CUresult {
    static_cast<void>(ctxCreateParams);
    TF_LOCK();
    const CUresult r = create_context(sim, pctx, flags, dev, false);
    if (r == CUDA_SUCCESS) {
      ::tf::push_current(*pctx);
    }
    return r;
  });
}

CUresult cuCtxDestroy(CUcontext ctx) {
  TF_ENTER(cuCtxDestroy);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK();
    return destroy_context(sim, lk, ctx);
  });
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
  TF_ENTER(cuCtxDestroy_v2);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK();
    return destroy_context(sim, lk, ctx);
  });
}

CUresult cuCtxPushCurrent(CUcontext ctx) {
  TF_ENTER(cuCtxPushCurrent);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    if (sim.contexts.get(ctx) == nullptr) {
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    ::tf::push_current(ctx);
    return CUDA_SUCCESS;
  });
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
  TF_ENTER(cuCtxPushCurrent_v2);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    if (sim.contexts.get(ctx) == nullptr) {
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    ::tf::push_current(ctx);
    return CUDA_SUCCESS;
  });
}

CUresult cuCtxPopCurrent(CUcontext* pctx) {
  TF_ENTER(cuCtxPopCurrent);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    void* top = ::tf::pop_current();
    if (top == nullptr) {
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (pctx != nullptr) {
      *pctx = static_cast<CUcontext>(top);
    }
    return CUDA_SUCCESS;
  });
}

CUresult cuCtxPopCurrent_v2(CUcontext* pctx) {
  TF_ENTER(cuCtxPopCurrent_v2);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    void* top = ::tf::pop_current();
    if (top == nullptr) {
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (pctx != nullptr) {
      *pctx = static_cast<CUcontext>(top);
    }
    return CUDA_SUCCESS;
  });
}

CUresult cuCtxGetCurrent(CUcontext* pctx) {
  TF_ENTER(cuCtxGetCurrent);
  return guarded([&]() -> CUresult {
    if (pctx == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    *pctx = static_cast<CUcontext>(::tf::current_context());
    return CUDA_SUCCESS;
  });
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  TF_ENTER(cuCtxSetCurrent);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    if (ctx != nullptr && sim.contexts.get(ctx) == nullptr) {
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    ::tf::set_current(ctx);
    return CUDA_SUCCESS;
  });
}

CUresult cuCtxGetDevice(CUdevice* device) {
  TF_ENTER(cuCtxGetDevice);
  return guarded([&]() -> CUresult {
    if (device == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK_CTX(ctx);
    *device = sim.contexts.get(ctx)->dev;
    return CUDA_SUCCESS;
  });
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) {
  TF_ENTER(cuCtxGetApiVersion);
  return guarded([&]() -> CUresult {
    if (version == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    void* target = ctx != nullptr ? static_cast<void*>(ctx) : ::tf::current_context();
    if (sim.contexts.get(target) == nullptr) {
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    *version = 3020;
    return CUDA_SUCCESS;
  });
}

CUresult cuCtxSynchronize(void) {
  TF_ENTER(cuCtxSynchronize);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    sim.drain_context_locked(lk, ctx);
    return CUDA_SUCCESS;
  });
}

CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  TF_ENTER(cuCtxGetStreamPriorityRange);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    if (leastPriority != nullptr) {
      *leastPriority = 0;
    }
    if (greatestPriority != nullptr) {
      *greatestPriority = -5;
    }
    return CUDA_SUCCESS;
  });
}

// ===========================================================================
// Modules and functions
// ===========================================================================

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
  TF_ENTER(cuModuleLoadData);
  return guarded([&]() -> CUresult {
    if (module == nullptr || image == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    *module = static_cast<CUmodule>(sim.modules.create(std::make_unique<Module>()));
    return CUDA_SUCCESS;
  });
}

CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
  TF_ENTER(cuModuleGetFunction);
  return guarded([&]() -> CUresult {
    if (hfunc == nullptr || name == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    Module* m = sim.modules.get(hmod);
    if (m == nullptr) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    auto it = sim.functions_by_name.find(name);
    if (it != sim.functions_by_name.end()) {
      *hfunc = static_cast<CUfunction>(it->second);
      return CUDA_SUCCESS;
    }
    auto f = std::make_unique<Function>();
    f->name = name;
    f->duration_us = sim.default_kernel_duration_us;
    void* h = sim.functions.create(std::move(f));
    sim.functions_by_name.emplace(name, h);
    m->functions.push_back(h);
    *hfunc = static_cast<CUfunction>(h);
    return CUDA_SUCCESS;
  });
}

CUresult cuModuleUnload(CUmodule hmod) {
  TF_ENTER(cuModuleUnload);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    if (sim.modules.destroy(hmod) == nullptr) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    return CUDA_SUCCESS;
  });
}

// ===========================================================================
// Launches
// ===========================================================================

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra) {
  TF_ENTER(cuLaunchKernel);
  return guarded([&]() -> CUresult {
    static_cast<void>(gridDimX);
    static_cast<void>(gridDimY);
    static_cast<void>(gridDimZ);
    static_cast<void>(blockDimX);
    static_cast<void>(blockDimY);
    static_cast<void>(blockDimZ);
    static_cast<void>(sharedMemBytes);
    static_cast<void>(extra);
    TF_WAIT_LOCK_CTX(ctx);
    return launch_impl(sim, lk, ctx, f, hStream, false, kernelParams);
  });
}

CUresult cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                             unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                             unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                             void** kernelParams, void** extra) {
  TF_ENTER(cuLaunchKernel_ptsz);
  return guarded([&]() -> CUresult {
    static_cast<void>(gridDimX);
    static_cast<void>(gridDimY);
    static_cast<void>(gridDimZ);
    static_cast<void>(blockDimX);
    static_cast<void>(blockDimY);
    static_cast<void>(blockDimZ);
    static_cast<void>(sharedMemBytes);
    static_cast<void>(extra);
    TF_WAIT_LOCK_CTX(ctx);
    return launch_impl(sim, lk, ctx, f, hStream, true, kernelParams);
  });
}

CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams,
                          void** extra) {
  TF_ENTER(cuLaunchKernelEx);
  return guarded([&]() -> CUresult {
    static_cast<void>(extra);
    if (config == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_WAIT_LOCK_CTX(ctx);
    return launch_impl(sim, lk, ctx, f, config->hStream, false, kernelParams);
  });
}

CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction f, void** kernelParams,
                               void** extra) {
  TF_ENTER(cuLaunchKernelEx_ptsz);
  return guarded([&]() -> CUresult {
    static_cast<void>(extra);
    if (config == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_WAIT_LOCK_CTX(ctx);
    return launch_impl(sim, lk, ctx, f, config->hStream, true, kernelParams);
  });
}

CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                   unsigned int gridDimZ, unsigned int blockDimX,
                                   unsigned int blockDimY, unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hStream,
                                   void** kernelParams) {
  TF_ENTER(cuLaunchCooperativeKernel);
  return guarded([&]() -> CUresult {
    static_cast<void>(gridDimX);
    static_cast<void>(gridDimY);
    static_cast<void>(gridDimZ);
    static_cast<void>(blockDimX);
    static_cast<void>(blockDimY);
    static_cast<void>(blockDimZ);
    static_cast<void>(sharedMemBytes);
    TF_WAIT_LOCK_CTX(ctx);
    return launch_impl(sim, lk, ctx, f, hStream, false, kernelParams);
  });
}

CUresult cuLaunchCooperativeKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                        unsigned int gridDimZ, unsigned int blockDimX,
                                        unsigned int blockDimY, unsigned int blockDimZ,
                                        unsigned int sharedMemBytes, CUstream hStream,
                                        void** kernelParams) {
  TF_ENTER(cuLaunchCooperativeKernel_ptsz);
  return guarded([&]() -> CUresult {
    static_cast<void>(gridDimX);
    static_cast<void>(gridDimY);
    static_cast<void>(gridDimZ);
    static_cast<void>(blockDimX);
    static_cast<void>(blockDimY);
    static_cast<void>(blockDimZ);
    static_cast<void>(sharedMemBytes);
    TF_WAIT_LOCK_CTX(ctx);
    return launch_impl(sim, lk, ctx, f, hStream, true, kernelParams);
  });
}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
  TF_ENTER(cuGraphLaunch);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return graph_launch_impl(sim, ctx, hGraphExec, hStream, false);
  });
}

CUresult cuGraphLaunch_ptsz(CUgraphExec hGraphExec, CUstream hStream) {
  TF_ENTER(cuGraphLaunch_ptsz);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return graph_launch_impl(sim, ctx, hGraphExec, hStream, true);
  });
}

// ===========================================================================
// Memory
// ===========================================================================

CUresult cuMemGetInfo(unsigned int* freeMem, unsigned int* totalMem) {
  TF_ENTER(cuMemGetInfo);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    const uint64_t total = sim.device.total_mem;
    const uint64_t avail = total - sim.allocated_bytes;
    if (freeMem != nullptr) {
      *freeMem = static_cast<unsigned int>(std::min<uint64_t>(avail, UINT32_MAX));
    }
    if (totalMem != nullptr) {
      *totalMem = static_cast<unsigned int>(std::min<uint64_t>(total, UINT32_MAX));
    }
    return CUDA_SUCCESS;
  });
}

CUresult cuMemGetInfo_v2(size_t* freeMem, size_t* totalMem) {
  TF_ENTER(cuMemGetInfo_v2);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    if (freeMem != nullptr) {
      *freeMem = static_cast<size_t>(sim.device.total_mem - sim.allocated_bytes);
    }
    if (totalMem != nullptr) {
      *totalMem = static_cast<size_t>(sim.device.total_mem);
    }
    return CUDA_SUCCESS;
  });
}

CUresult cuMemAlloc(unsigned int* dptr, unsigned int bytesize) {
  TF_ENTER(cuMemAlloc);
  return guarded([&]() -> CUresult {
    if (dptr == nullptr || bytesize == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    void* p = sim.device_alloc_locked(bytesize);
    if (p == nullptr) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *dptr = v1_register(sim, p);
    return CUDA_SUCCESS;
  });
}

CUresult cuMemAlloc_v2(CUdeviceptr* dptr, size_t bytesize) {
  TF_ENTER(cuMemAlloc_v2);
  return guarded([&]() -> CUresult {
    if (dptr == nullptr || bytesize == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    void* p = sim.device_alloc_locked(bytesize);
    if (p == nullptr) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *dptr = device_ptr(p);
    return CUDA_SUCCESS;
  });
}

CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
  TF_ENTER(cuMemAllocManaged);
  return guarded([&]() -> CUresult {
    static_cast<void>(flags);
    if (dptr == nullptr || bytesize == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    void* p = sim.device_alloc_locked(bytesize);
    if (p == nullptr) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    sim.allocs[reinterpret_cast<uintptr_t>(p)].managed = true;
    *dptr = device_ptr(p);
    return CUDA_SUCCESS;
  });
}

namespace {
constexpr size_t kPitchAlign = 512;
size_t pitch_for(size_t width) {
  return (width + kPitchAlign - 1) / kPitchAlign * kPitchAlign;
}
}  // namespace

CUresult cuMemAllocPitch(unsigned int* dptr, unsigned int* pPitch, unsigned int WidthInBytes,
                         unsigned int Height, unsigned int ElementSizeBytes) {
  TF_ENTER(cuMemAllocPitch);
  return guarded([&]() -> CUresult {
    static_cast<void>(ElementSizeBytes);
    if (dptr == nullptr || pPitch == nullptr || WidthInBytes == 0 || Height == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    const size_t pitch = pitch_for(WidthInBytes);
    void* p = sim.device_alloc_locked(pitch * Height);
    if (p == nullptr) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    sim.allocs[reinterpret_cast<uintptr_t>(p)].pitch = pitch;
    *pPitch = static_cast<unsigned int>(pitch);
    *dptr = v1_register(sim, p);
    return CUDA_SUCCESS;
  });
}

CUresult cuMemAllocPitch_v2(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes) {
  TF_ENTER(cuMemAllocPitch_v2);
  return guarded([&]() -> CUresult {
    static_cast<void>(ElementSizeBytes);
    if (dptr == nullptr || pPitch == nullptr || WidthInBytes == 0 || Height == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    const size_t pitch = pitch_for(WidthInBytes);
    void* p = sim.device_alloc_locked(pitch * Height);
    if (p == nullptr) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    sim.allocs[reinterpret_cast<uintptr_t>(p)].pitch = pitch;
    *pPitch = pitch;
    *dptr = device_ptr(p);
    return CUDA_SUCCESS;
  });
}

namespace {
CUresult free_device(Sim& sim, uintptr_t addr) {
  auto it = sim.allocs.find(addr);
  if (it == sim.allocs.end()) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (it->second.vmm) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  sim.allocated_bytes -= it->second.size;
  sim.allocs.erase(it);
  std::free(::tf::address_to_pointer(addr));
  return CUDA_SUCCESS;
}
}  // namespace

CUresult cuMemFree(unsigned int dptr) {
  TF_ENTER(cuMemFree);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    const uintptr_t addr = v1_resolve(sim, dptr);
    if (addr == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    const CUresult r = free_device(sim, addr);
    if (r == CUDA_SUCCESS) {
      sim.v1_alias.erase(dptr);
    }
    return r;
  });
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
  TF_ENTER(cuMemFree_v2);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    return free_device(sim, static_cast<uintptr_t>(dptr));
  });
}

namespace {
CUresult alloc_async(Sim& sim, void* ctx, CUdeviceptr* dptr, size_t bytesize, CUstream hStream,
                     bool per_thread) {
  if (dptr == nullptr || bytesize == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, hStream, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  void* p = sim.device_alloc_locked(bytesize);
  if (p == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  *dptr = device_ptr(p);
  return CUDA_SUCCESS;
}

CUresult free_async(Sim& sim, void* ctx, CUdeviceptr dptr, CUstream hStream, bool per_thread) {
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, hStream, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  const uintptr_t addr = static_cast<uintptr_t>(dptr);
  auto it = sim.allocs.find(addr);
  if (it == sim.allocs.end() || it->second.vmm) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Op op;
  op.kind = Op::Kind::kFree;
  op.free_ptr = addr;
  sim.enqueue_locked(*s, std::move(op));
  return CUDA_SUCCESS;
}
}  // namespace

CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) {
  TF_ENTER(cuMemAllocAsync);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return alloc_async(sim, ctx, dptr, bytesize, hStream, false);
  });
}

CUresult cuMemAllocAsync_ptsz(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) {
  TF_ENTER(cuMemAllocAsync_ptsz);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return alloc_async(sim, ctx, dptr, bytesize, hStream, true);
  });
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool,
                                 CUstream hStream) {
  TF_ENTER(cuMemAllocFromPoolAsync);
  return guarded([&]() -> CUresult {
    static_cast<void>(pool);
    TF_LOCK_CTX(ctx);
    return alloc_async(sim, ctx, dptr, bytesize, hStream, false);
  });
}

CUresult cuMemAllocFromPoolAsync_ptsz(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool,
                                      CUstream hStream) {
  TF_ENTER(cuMemAllocFromPoolAsync_ptsz);
  return guarded([&]() -> CUresult {
    static_cast<void>(pool);
    TF_LOCK_CTX(ctx);
    return alloc_async(sim, ctx, dptr, bytesize, hStream, true);
  });
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
  TF_ENTER(cuMemFreeAsync);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return free_async(sim, ctx, dptr, hStream, false);
  });
}

CUresult cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream) {
  TF_ENTER(cuMemFreeAsync_ptsz);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return free_async(sim, ctx, dptr, hStream, true);
  });
}

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) {
  TF_ENTER(cuDeviceGetDefaultMemPool);
  return guarded([&]() -> CUresult {
    if (pool_out == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    if (dev != 0) {
      return CUDA_ERROR_INVALID_DEVICE;
    }
    // One pool per device, represented by a stable handle; the fake has no
    // per-pool behaviour beyond identity.
    *pool_out = static_cast<CUmemoryPool>(tf::encode_handle(tf::HKind::kMemPool, 0, 1));
    return CUDA_SUCCESS;
  });
}

// ===========================================================================
// Virtual memory management
// ===========================================================================

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr,
                             unsigned long long flags) {
  TF_ENTER(cuMemAddressReserve);
  return guarded([&]() -> CUresult {
    static_cast<void>(alignment);
    static_cast<void>(addr);
    static_cast<void>(flags);
    if (ptr == nullptr || size == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    void* p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    Reservation r;
    r.size = size;
    sim.reservations.emplace(reinterpret_cast<uintptr_t>(p), r);
    *ptr = device_ptr(p);
    return CUDA_SUCCESS;
  });
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
  TF_ENTER(cuMemAddressFree);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    auto it = sim.reservations.find(static_cast<uintptr_t>(ptr));
    if (it == sim.reservations.end() || it->second.size != size) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    munmap(host_ptr(ptr), size);
    sim.reservations.erase(it);
    return CUDA_SUCCESS;
  });
}

CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size,
                     const CUmemAllocationProp* prop, unsigned long long flags) {
  TF_ENTER(cuMemCreate);
  return guarded([&]() -> CUresult {
    static_cast<void>(prop);
    static_cast<void>(flags);
    if (handle == nullptr || size == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    if (sim.allocated_bytes + size > sim.device.total_mem) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    auto h = std::make_unique<MemHandleObj>();
    h->size = size;
    h->refs = 1;
    h->fd = Sim::make_backing_fd(size);
    void* obj = sim.mem_handles.create(std::move(h));
    sim.allocated_bytes += size;
    *handle = static_cast<CUmemGenericAllocationHandle>(reinterpret_cast<uintptr_t>(obj));
    return CUDA_SUCCESS;
  });
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
  TF_ENTER(cuMemRelease);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    void* obj = ::tf::address_to_pointer(handle);
    MemHandleObj* h = sim.mem_handles.get(obj);
    if (h == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    if (--h->refs > 0) {
      return CUDA_SUCCESS;
    }
    sim.allocated_bytes -= h->size;
    if (h->fd >= 0) {
      close(h->fd);
    }
    sim.mem_handles.destroy(obj);
    return CUDA_SUCCESS;
  });
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle,
                  unsigned long long flags) {
  TF_ENTER(cuMemMap);
  return guarded([&]() -> CUresult {
    static_cast<void>(flags);
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    void* obj = ::tf::address_to_pointer(handle);
    MemHandleObj* h = sim.mem_handles.get(obj);
    if (h == nullptr || size == 0 || offset + size > h->size) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    void* target = host_ptr(ptr);
    void* p = nullptr;
    if (h->fd >= 0) {
      p = mmap(target, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, h->fd,
               static_cast<off_t>(offset));
    } else {
      p = mmap(target, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1,
               0);
    }
    if (p == MAP_FAILED) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    ++h->refs;
    Allocation a;
    a.size = size;
    a.vmm = true;
    a.vmm_handle = obj;
    sim.allocs[static_cast<uintptr_t>(ptr)] = a;
    return CUDA_SUCCESS;
  });
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
  TF_ENTER(cuMemUnmap);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    auto it = sim.allocs.find(static_cast<uintptr_t>(ptr));
    if (it == sim.allocs.end() || !it->second.vmm || it->second.size != size) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    // Put the reservation back the way cuMemAddressReserve left it.
    if (mmap(host_ptr(ptr), size, PROT_NONE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0) == MAP_FAILED) {
      return CUDA_ERROR_UNKNOWN;
    }
    void* obj = it->second.vmm_handle;
    sim.allocs.erase(it);
    // The mapping held a reference on the cuMemCreate allocation: real CUDA
    // keeps the backing alive until the last mapping goes away.
    MemHandleObj* h = sim.mem_handles.get(obj);
    if (h != nullptr && --h->refs <= 0) {
      sim.allocated_bytes -= h->size;
      if (h->fd >= 0) {
        close(h->fd);
      }
      sim.mem_handles.destroy(obj);
    }
    return CUDA_SUCCESS;
  });
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) {
  TF_ENTER(cuMemSetAccess);
  return guarded([&]() -> CUresult {
    if (desc == nullptr || count == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    auto it = sim.allocs.find(static_cast<uintptr_t>(ptr));
    if (it == sim.allocs.end() || !it->second.vmm || it->second.size != size) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
  });
}

// ===========================================================================
// Streams
// ===========================================================================

namespace {
CUresult create_stream(Sim& sim, void* ctx, CUstream* phStream, unsigned int Flags, int priority) {
  if (phStream == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto s = std::make_unique<Stream>();
  s->ctx = ctx;
  s->flags = Flags;
  s->priority = priority;
  *phStream = static_cast<CUstream>(sim.streams.create(std::move(s)));
  return CUDA_SUCCESS;
}

CUresult destroy_stream(Sim& sim, std::unique_lock<std::mutex>& lk, CUstream hStream) {
  const uintptr_t raw = reinterpret_cast<uintptr_t>(hStream);
  if (raw == 0 || raw == tf::kStreamLegacyHandle || raw == tf::kStreamPerThreadHandle) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  Stream* s = sim.streams.get(hStream);
  if (s == nullptr) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  if (s->legacy_default || s->per_thread_default) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  sim.drain_stream_locked(lk, hStream);
  if (sim.streams.destroy(hStream) == nullptr) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  return CUDA_SUCCESS;
}
}  // namespace

CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) {
  TF_ENTER(cuStreamCreate);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return create_stream(sim, ctx, phStream, Flags, 0);
  });
}

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) {
  TF_ENTER(cuStreamCreateWithPriority);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return create_stream(sim, ctx, phStream, flags, priority);
  });
}

CUresult cuStreamDestroy(CUstream hStream) {
  TF_ENTER(cuStreamDestroy);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    return destroy_stream(sim, lk, hStream);
  });
}

CUresult cuStreamDestroy_v2(CUstream hStream) {
  TF_ENTER(cuStreamDestroy_v2);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    return destroy_stream(sim, lk, hStream);
  });
}

namespace {
CUresult sync_stream(Sim& sim, std::unique_lock<std::mutex>& lk, void* ctx, CUstream hStream,
                     bool per_thread) {
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, hStream, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  if (s->capture == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
    return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
  }
  sim.drain_stream_locked(lk, handle);
  return CUDA_SUCCESS;
}

CUresult query_stream(Sim& sim, void* ctx, CUstream hStream, bool per_thread) {
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, hStream, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  return s->ops.empty() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
}  // namespace

CUresult cuStreamSynchronize(CUstream hStream) {
  TF_ENTER(cuStreamSynchronize);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return sync_stream(sim, lk, ctx, hStream, false);
  });
}

CUresult cuStreamSynchronize_ptsz(CUstream hStream) {
  TF_ENTER(cuStreamSynchronize_ptsz);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return sync_stream(sim, lk, ctx, hStream, true);
  });
}

CUresult cuStreamQuery(CUstream hStream) {
  TF_ENTER(cuStreamQuery);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return query_stream(sim, ctx, hStream, false);
  });
}

CUresult cuStreamQuery_ptsz(CUstream hStream) {
  TF_ENTER(cuStreamQuery_ptsz);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return query_stream(sim, ctx, hStream, true);
  });
}

namespace {
CUresult stream_priority(Sim& sim, void* ctx, CUstream hStream, bool per_thread, int* priority) {
  if (priority == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, hStream, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  *priority = s->priority;
  return CUDA_SUCCESS;
}

CUresult stream_flags(Sim& sim, void* ctx, CUstream hStream, bool per_thread, unsigned int* flags) {
  if (flags == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, hStream, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  *flags = s->flags;
  return CUDA_SUCCESS;
}

CUresult stream_wait_event(Sim& sim, void* ctx, CUstream hStream, bool per_thread, CUevent hEvent) {
  Event* e = sim.events.get(hEvent);
  if (e == nullptr) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  Stream* s = nullptr;
  void* handle = nullptr;
  const CUresult r = get_stream(sim, ctx, hStream, per_thread, &s, &handle);
  if (r != CUDA_SUCCESS) {
    return r;
  }
  if (e->recorded && !e->complete && e->time_us > s->ready_at_us) {
    s->ready_at_us = e->time_us;
  }
  return CUDA_SUCCESS;
}
}  // namespace

CUresult cuStreamGetPriority(CUstream hStream, int* priority) {
  TF_ENTER(cuStreamGetPriority);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return stream_priority(sim, ctx, hStream, false, priority);
  });
}

CUresult cuStreamGetPriority_ptsz(CUstream hStream, int* priority) {
  TF_ENTER(cuStreamGetPriority_ptsz);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return stream_priority(sim, ctx, hStream, true, priority);
  });
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags) {
  TF_ENTER(cuStreamGetFlags);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return stream_flags(sim, ctx, hStream, false, flags);
  });
}

CUresult cuStreamGetFlags_ptsz(CUstream hStream, unsigned int* flags) {
  TF_ENTER(cuStreamGetFlags_ptsz);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return stream_flags(sim, ctx, hStream, true, flags);
  });
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
  TF_ENTER(cuStreamWaitEvent);
  return guarded([&]() -> CUresult {
    static_cast<void>(Flags);
    TF_LOCK_CTX(ctx);
    return stream_wait_event(sim, ctx, hStream, false, hEvent);
  });
}

CUresult cuStreamWaitEvent_ptsz(CUstream hStream, CUevent hEvent, unsigned int Flags) {
  TF_ENTER(cuStreamWaitEvent_ptsz);
  return guarded([&]() -> CUresult {
    static_cast<void>(Flags);
    TF_LOCK_CTX(ctx);
    return stream_wait_event(sim, ctx, hStream, true, hEvent);
  });
}

// ===========================================================================
// Stream capture
// ===========================================================================

CUresult cuStreamBeginCapture(CUstream hStream) {
  TF_ENTER(cuStreamBeginCapture);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return begin_capture_impl(sim, ctx, hStream, false, CU_STREAM_CAPTURE_MODE_GLOBAL);
  });
}

CUresult cuStreamBeginCapture_ptsz(CUstream hStream) {
  TF_ENTER(cuStreamBeginCapture_ptsz);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return begin_capture_impl(sim, ctx, hStream, true, CU_STREAM_CAPTURE_MODE_GLOBAL);
  });
}

CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
  TF_ENTER(cuStreamBeginCapture_v2);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return begin_capture_impl(sim, ctx, hStream, false, mode);
  });
}

CUresult cuStreamBeginCapture_v2_ptsz(CUstream hStream, CUstreamCaptureMode mode) {
  TF_ENTER(cuStreamBeginCapture_v2_ptsz);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return begin_capture_impl(sim, ctx, hStream, true, mode);
  });
}

CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) {
  TF_ENTER(cuStreamEndCapture);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return end_capture_impl(sim, ctx, hStream, false, phGraph);
  });
}

CUresult cuStreamEndCapture_ptsz(CUstream hStream, CUgraph* phGraph) {
  TF_ENTER(cuStreamEndCapture_ptsz);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return end_capture_impl(sim, ctx, hStream, true, phGraph);
  });
}

CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus* captureStatus) {
  TF_ENTER(cuStreamIsCapturing);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return is_capturing_impl(sim, ctx, hStream, false, captureStatus);
  });
}

CUresult cuStreamIsCapturing_ptsz(CUstream hStream, CUstreamCaptureStatus* captureStatus) {
  TF_ENTER(cuStreamIsCapturing_ptsz);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return is_capturing_impl(sim, ctx, hStream, true, captureStatus);
  });
}

// ===========================================================================
// Graphs
// ===========================================================================

CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags) {
  TF_ENTER(cuGraphCreate);
  return guarded([&]() -> CUresult {
    static_cast<void>(flags);
    if (phGraph == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    *phGraph = static_cast<CUgraph>(sim.graphs.create(std::make_unique<Graph>()));
    return CUDA_SUCCESS;
  });
}

CUresult cuGraphDestroy(CUgraph hGraph) {
  TF_ENTER(cuGraphDestroy);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    if (sim.graphs.destroy(hGraph) == nullptr) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    return CUDA_SUCCESS;
  });
}

namespace {
CUresult instantiate(Sim& sim, CUgraphExec* phGraphExec, CUgraph hGraph) {
  if (phGraphExec == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Graph* g = sim.graphs.get(hGraph);
  if (g == nullptr) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  auto ge = std::make_unique<GraphExec>();
  ge->nodes = g->nodes;
  *phGraphExec = static_cast<CUgraphExec>(sim.graph_execs.create(std::move(ge)));
  return CUDA_SUCCESS;
}
}  // namespace

CUresult cuGraphInstantiate(CUgraphExec* phGraphExec, CUgraph hGraph, CUgraphNode* phErrorNode,
                            char* logBuffer, size_t bufferSize) {
  TF_ENTER(cuGraphInstantiate);
  return guarded([&]() -> CUresult {
    if (phErrorNode != nullptr) {
      *phErrorNode = nullptr;
    }
    if (logBuffer != nullptr && bufferSize > 0) {
      logBuffer[0] = '\0';
    }
    TF_LOCK();
    return instantiate(sim, phGraphExec, hGraph);
  });
}

CUresult cuGraphInstantiate_v2(CUgraphExec* phGraphExec, CUgraph hGraph, CUgraphNode* phErrorNode,
                               char* logBuffer, size_t bufferSize) {
  TF_ENTER(cuGraphInstantiate_v2);
  return guarded([&]() -> CUresult {
    if (phErrorNode != nullptr) {
      *phErrorNode = nullptr;
    }
    if (logBuffer != nullptr && bufferSize > 0) {
      logBuffer[0] = '\0';
    }
    TF_LOCK();
    return instantiate(sim, phGraphExec, hGraph);
  });
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec, CUgraph hGraph,
                                     unsigned long long flags) {
  TF_ENTER(cuGraphInstantiateWithFlags);
  return guarded([&]() -> CUresult {
    static_cast<void>(flags);
    TF_LOCK();
    return instantiate(sim, phGraphExec, hGraph);
  });
}

CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
  TF_ENTER(cuGraphExecDestroy);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    if (sim.graph_execs.destroy(hGraphExec) == nullptr) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    return CUDA_SUCCESS;
  });
}

// ===========================================================================
// Events
// ===========================================================================

CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags) {
  TF_ENTER(cuEventCreate);
  return guarded([&]() -> CUresult {
    if (phEvent == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK_CTX(ctx);
    static_cast<void>(ctx);
    auto e = std::make_unique<Event>();
    e->flags = Flags;
    *phEvent = static_cast<CUevent>(sim.events.create(std::move(e)));
    return CUDA_SUCCESS;
  });
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
  TF_ENTER(cuEventRecord);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return record_event_impl(sim, ctx, hEvent, hStream, false);
  });
}

CUresult cuEventRecord_ptsz(CUevent hEvent, CUstream hStream) {
  TF_ENTER(cuEventRecord_ptsz);
  return guarded([&]() -> CUresult {
    TF_LOCK_CTX(ctx);
    return record_event_impl(sim, ctx, hEvent, hStream, true);
  });
}

CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) {
  TF_ENTER(cuEventRecordWithFlags);
  return guarded([&]() -> CUresult {
    static_cast<void>(flags);
    TF_LOCK_CTX(ctx);
    return record_event_impl(sim, ctx, hEvent, hStream, false);
  });
}

CUresult cuEventRecordWithFlags_ptsz(CUevent hEvent, CUstream hStream, unsigned int flags) {
  TF_ENTER(cuEventRecordWithFlags_ptsz);
  return guarded([&]() -> CUresult {
    static_cast<void>(flags);
    TF_LOCK_CTX(ctx);
    return record_event_impl(sim, ctx, hEvent, hStream, true);
  });
}

CUresult cuEventQuery(CUevent hEvent) {
  TF_ENTER(cuEventQuery);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    Event* e = sim.events.get(hEvent);
    if (e == nullptr) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    if (!e->recorded) {
      return CUDA_SUCCESS;
    }
    return e->complete ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
  });
}

CUresult cuEventSynchronize(CUevent hEvent) {
  TF_ENTER(cuEventSynchronize);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK();
    if (sim.events.get(hEvent) == nullptr) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    sim.wait_event_locked(lk, hEvent);
    return CUDA_SUCCESS;
  });
}

CUresult cuEventDestroy(CUevent hEvent) {
  TF_ENTER(cuEventDestroy);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    if (sim.events.destroy(hEvent) == nullptr) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    return CUDA_SUCCESS;
  });
}

CUresult cuEventDestroy_v2(CUevent hEvent) {
  TF_ENTER(cuEventDestroy_v2);
  return guarded([&]() -> CUresult {
    TF_LOCK();
    if (sim.events.destroy(hEvent) == nullptr) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    return CUDA_SUCCESS;
  });
}

CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart, CUevent hEnd) {
  TF_ENTER(cuEventElapsedTime);
  return guarded([&]() -> CUresult {
    if (pMilliseconds == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    TF_LOCK();
    Event* a = sim.events.get(hStart);
    Event* b = sim.events.get(hEnd);
    if (a == nullptr || b == nullptr) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    if (!a->recorded || !b->recorded) {
      return CUDA_ERROR_INVALID_HANDLE;
    }
    if (!a->complete || !b->complete) {
      return CUDA_ERROR_NOT_READY;
    }
    const int64_t delta = static_cast<int64_t>(b->time_us) - static_cast<int64_t>(a->time_us);
    *pMilliseconds = static_cast<float>(static_cast<double>(delta) / 1000.0);
    return CUDA_SUCCESS;
  });
}

// ===========================================================================
// Copies and fills
// ===========================================================================

CUresult cuMemcpyHtoD(unsigned int dstDevice, const void* srcHost, unsigned int ByteCount) {
  TF_ENTER(cuMemcpyHtoD);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    const uintptr_t dst = v1_resolve(sim, dstDevice);
    if (dst == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    return copy_impl(sim, lk, ctx, nullptr, false, ::tf::address_to_pointer(dst), srcHost,
                     ByteCount, true, false, true);
  });
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  TF_ENTER(cuMemcpyHtoD_v2);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return copy_impl(sim, lk, ctx, nullptr, false, host_ptr(dstDevice), srcHost, ByteCount, true,
                     false, true);
  });
}

CUresult cuMemcpyHtoD_v2_ptds(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  TF_ENTER(cuMemcpyHtoD_v2_ptds);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return copy_impl(sim, lk, ctx, nullptr, true, host_ptr(dstDevice), srcHost, ByteCount, true,
                     false, true);
  });
}

CUresult cuMemcpyDtoH(void* dstHost, unsigned int srcDevice, unsigned int ByteCount) {
  TF_ENTER(cuMemcpyDtoH);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    const uintptr_t src = v1_resolve(sim, srcDevice);
    if (src == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    return copy_impl(sim, lk, ctx, nullptr, false, dstHost, ::tf::address_to_pointer(src),
                     ByteCount, false, true, true);
  });
}

CUresult cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  TF_ENTER(cuMemcpyDtoH_v2);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return copy_impl(sim, lk, ctx, nullptr, false, dstHost, host_ptr(srcDevice), ByteCount, false,
                     true, true);
  });
}

CUresult cuMemcpyDtoH_v2_ptds(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  TF_ENTER(cuMemcpyDtoH_v2_ptds);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return copy_impl(sim, lk, ctx, nullptr, true, dstHost, host_ptr(srcDevice), ByteCount, false,
                     true, true);
  });
}

CUresult cuMemcpyDtoD(unsigned int dstDevice, unsigned int srcDevice, unsigned int ByteCount) {
  TF_ENTER(cuMemcpyDtoD);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    const uintptr_t dst = v1_resolve(sim, dstDevice);
    const uintptr_t src = v1_resolve(sim, srcDevice);
    if (dst == 0 || src == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    return copy_impl(sim, lk, ctx, nullptr, false, ::tf::address_to_pointer(dst),
                     ::tf::address_to_pointer(src), ByteCount, true, true, true);
  });
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  TF_ENTER(cuMemcpyDtoD_v2);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return copy_impl(sim, lk, ctx, nullptr, false, host_ptr(dstDevice), host_ptr(srcDevice),
                     ByteCount, true, true, true);
  });
}

CUresult cuMemcpyDtoD_v2_ptds(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  TF_ENTER(cuMemcpyDtoD_v2_ptds);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return copy_impl(sim, lk, ctx, nullptr, true, host_ptr(dstDevice), host_ptr(srcDevice),
                     ByteCount, true, true, true);
  });
}

CUresult cuMemcpyHtoDAsync(unsigned int dstDevice, const void* srcHost, unsigned int ByteCount,
                           CUstream hStream) {
  TF_ENTER(cuMemcpyHtoDAsync);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    const uintptr_t dst = v1_resolve(sim, dstDevice);
    if (dst == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    return copy_impl(sim, lk, ctx, hStream, false, ::tf::address_to_pointer(dst), srcHost,
                     ByteCount, true, false, false);
  });
}

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount,
                              CUstream hStream) {
  TF_ENTER(cuMemcpyHtoDAsync_v2);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return copy_impl(sim, lk, ctx, hStream, false, host_ptr(dstDevice), srcHost, ByteCount, true,
                     false, false);
  });
}

CUresult cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount,
                                   CUstream hStream) {
  TF_ENTER(cuMemcpyHtoDAsync_v2_ptsz);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return copy_impl(sim, lk, ctx, hStream, true, host_ptr(dstDevice), srcHost, ByteCount, true,
                     false, false);
  });
}

CUresult cuMemcpyDtoHAsync(void* dstHost, unsigned int srcDevice, unsigned int ByteCount,
                           CUstream hStream) {
  TF_ENTER(cuMemcpyDtoHAsync);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    const uintptr_t src = v1_resolve(sim, srcDevice);
    if (src == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    return copy_impl(sim, lk, ctx, hStream, false, dstHost, ::tf::address_to_pointer(src),
                     ByteCount, false, true, false);
  });
}

CUresult cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                              CUstream hStream) {
  TF_ENTER(cuMemcpyDtoHAsync_v2);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return copy_impl(sim, lk, ctx, hStream, false, dstHost, host_ptr(srcDevice), ByteCount, false,
                     true, false);
  });
}

CUresult cuMemcpyDtoHAsync_v2_ptsz(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                                   CUstream hStream) {
  TF_ENTER(cuMemcpyDtoHAsync_v2_ptsz);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return copy_impl(sim, lk, ctx, hStream, true, dstHost, host_ptr(srcDevice), ByteCount, false,
                     true, false);
  });
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
  TF_ENTER(cuMemcpyAsync);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return copy_impl(sim, lk, ctx, hStream, false, host_ptr(dst), host_ptr(src), ByteCount, true,
                     true, false);
  });
}

CUresult cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
  TF_ENTER(cuMemcpyAsync_ptsz);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return copy_impl(sim, lk, ctx, hStream, true, host_ptr(dst), host_ptr(src), ByteCount, true,
                     true, false);
  });
}

CUresult cuMemsetD8(unsigned int dstDevice, unsigned char uc, unsigned int N) {
  TF_ENTER(cuMemsetD8);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    const uintptr_t dst = v1_resolve(sim, dstDevice);
    if (dst == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    return memset_impl(sim, lk, ctx, nullptr, false, static_cast<CUdeviceptr>(dst), uc, 1, N, true);
  });
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  TF_ENTER(cuMemsetD8_v2);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return memset_impl(sim, lk, ctx, nullptr, false, dstDevice, uc, 1, N, true);
  });
}

CUresult cuMemsetD8_v2_ptds(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  TF_ENTER(cuMemsetD8_v2_ptds);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return memset_impl(sim, lk, ctx, nullptr, true, dstDevice, uc, 1, N, true);
  });
}

CUresult cuMemsetD32(unsigned int dstDevice, unsigned int ui, unsigned int N) {
  TF_ENTER(cuMemsetD32);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    const uintptr_t dst = v1_resolve(sim, dstDevice);
    if (dst == 0) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    return memset_impl(sim, lk, ctx, nullptr, false, static_cast<CUdeviceptr>(dst), ui, 4, N, true);
  });
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  TF_ENTER(cuMemsetD32_v2);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return memset_impl(sim, lk, ctx, nullptr, false, dstDevice, ui, 4, N, true);
  });
}

CUresult cuMemsetD32_v2_ptds(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  TF_ENTER(cuMemsetD32_v2_ptds);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return memset_impl(sim, lk, ctx, nullptr, true, dstDevice, ui, 4, N, true);
  });
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
  TF_ENTER(cuMemsetD8Async);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return memset_impl(sim, lk, ctx, hStream, false, dstDevice, uc, 1, N, false);
  });
}

CUresult cuMemsetD8Async_ptsz(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
  TF_ENTER(cuMemsetD8Async_ptsz);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return memset_impl(sim, lk, ctx, hStream, true, dstDevice, uc, 1, N, false);
  });
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
  TF_ENTER(cuMemsetD32Async);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return memset_impl(sim, lk, ctx, hStream, false, dstDevice, ui, 4, N, false);
  });
}

CUresult cuMemsetD32Async_ptsz(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
  TF_ENTER(cuMemsetD32Async_ptsz);
  return guarded([&]() -> CUresult {
    TF_WAIT_LOCK_CTX(ctx);
    return memset_impl(sim, lk, ctx, hStream, true, dstDevice, ui, 4, N, false);
  });
}

// ===========================================================================
// The lookup function itself. cudart resolves everything, including this
// function, through it, so it must work before cuInit and it must be able to
// find itself.
// ===========================================================================

CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags) {
  TF_ENTER(cuGetProcAddress);
  return guarded(
      [&]() -> CUresult { return resolve_proc(symbol, pfn, cudaVersion, flags, nullptr); });
}

CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags,
                             CUdriverProcAddressQueryResult* symbolStatus) {
  TF_ENTER(cuGetProcAddress_v2);
  return guarded(
      [&]() -> CUresult { return resolve_proc(symbol, pfn, cudaVersion, flags, symbolStatus); });
}

// ===========================================================================
// The table itself, built from the same list that generated the symbol enum
// and the linker version script.
// ===========================================================================

namespace tf {
namespace {

const SymbolRow kSymbols[] = {
#define TF_ROW(exported, base, version, kind) \
  {#exported, #base, version, kind, reinterpret_cast<void (*)(void)>(exported)},
#include "table.inc"
#undef TF_ROW
};

}  // namespace

const SymbolRow* symbol_table() {
  return kSymbols;
}

uint32_t symbol_table_size() {
  return static_cast<uint32_t>(std::size(kSymbols));
}

int64_t symbol_index(const char* exported_name) {
  if (exported_name == nullptr) {
    return -1;
  }
  for (uint32_t i = 0; i < symbol_table_size(); ++i) {
    if (std::strcmp(kSymbols[i].exported, exported_name) == 0) {
      return static_cast<int64_t>(i);
    }
  }
  return -1;
}

}  // namespace tf
