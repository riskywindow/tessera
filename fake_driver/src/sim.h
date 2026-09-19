// The simulated GPU behind libcuda_fake.
//
// One process-wide Sim holds every context, stream, event, module, graph and
// allocation, plus the simulated clock and the worker thread that completes
// queued work in order. Sim::mu guards all of it; the per-symbol call counters
// in counters.h are the only state outside the lock.
//
// Nothing here throws out of a cu* entry point: the entry points in
// entry_points.cc are the only callers, and they wrap everything that can
// allocate.

#ifndef TESSERA_FAKE_SIM_H
#define TESSERA_FAKE_SIM_H

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda.h>

namespace tf {

// The two reserved stream handles cuda.h spells CU_STREAM_LEGACY and
// CU_STREAM_PER_THREAD, as plain integers so that no C-style cast from a
// system-header macro leaks into first-party code.
inline constexpr uintptr_t kStreamLegacyHandle = 1;
inline constexpr uintptr_t kStreamPerThreadHandle = 2;

// ---------------------------------------------------------------------------
// Opaque handles.
//
// CUcontext, CUstream and friends are integers dressed as pointers: a kind tag,
// a generation and a slot index. Nothing ever dereferences them, so a stale
// handle is reported as CUDA_ERROR_INVALID_HANDLE instead of being a
// use-after-free. CUdeviceptr is the exception: it really is a host address, so
// that a simulated kernel can compute into it.
// ---------------------------------------------------------------------------

enum class HKind : uint8_t {
  kContext = 1,
  kStream = 2,
  kEvent = 3,
  kModule = 4,
  kFunction = 5,
  kGraph = 6,
  kGraphExec = 7,
  kMemHandle = 8,
  kMemPool = 9,
};

// The one place this library turns an integer into a pointer. It is not an
// accident here: CUdeviceptr really is a host address, because a simulated
// kernel has to compute into it, and the opaque handles really are integers
// dressed as pointers, because nothing ever dereferences them.
inline void* address_to_pointer(uint64_t v) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<void*>(static_cast<uintptr_t>(v));
}

inline void* encode_handle(HKind k, uint32_t idx, uint16_t gen) {
  const uint64_t v = (UINT64_C(0xF0) << 56) | (static_cast<uint64_t>(k) << 48) |
                     (static_cast<uint64_t>(gen) << 32) | idx;
  return address_to_pointer(v);
}

inline bool decode_handle(const void* h, HKind k, uint32_t* idx, uint16_t* gen) {
  const uint64_t v = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(h));
  if ((v >> 56) != UINT64_C(0xF0)) {
    return false;
  }
  if (static_cast<HKind>((v >> 48) & 0xFF) != k) {
    return false;
  }
  *gen = static_cast<uint16_t>((v >> 32) & 0xFFFF);
  *idx = static_cast<uint32_t>(v & 0xFFFFFFFF);
  return true;
}

// A slot table of objects addressed by opaque handle.
template <class T>
class Table {
 public:
  explicit Table(HKind kind) : kind_(kind) {}

  void* create(std::unique_ptr<T> obj) {
    uint32_t idx;
    if (!free_.empty()) {
      idx = free_.back();
      free_.pop_back();
    } else {
      idx = static_cast<uint32_t>(slots_.size());
      slots_.emplace_back();
      gens_.push_back(1);
    }
    slots_[idx] = std::move(obj);
    return encode_handle(kind_, idx, gens_[idx]);
  }

  T* get(const void* h) const {
    uint32_t idx = 0;
    uint16_t gen = 0;
    if (!decode_handle(h, kind_, &idx, &gen)) {
      return nullptr;
    }
    if (idx >= slots_.size() || gens_[idx] != gen) {
      return nullptr;
    }
    return slots_[idx].get();
  }

  // Removes the object and invalidates every copy of its handle.
  std::unique_ptr<T> destroy(const void* h) {
    uint32_t idx = 0;
    uint16_t gen = 0;
    if (!decode_handle(h, kind_, &idx, &gen)) {
      return nullptr;
    }
    if (idx >= slots_.size() || gens_[idx] != gen || !slots_[idx]) {
      return nullptr;
    }
    std::unique_ptr<T> obj = std::move(slots_[idx]);
    gens_[idx] = static_cast<uint16_t>(gens_[idx] + 1);
    if (gens_[idx] == 0) {
      gens_[idx] = 1;
    }
    free_.push_back(idx);
    return obj;
  }

  template <class F>
  void for_each(F&& f) {
    for (uint32_t i = 0; i < slots_.size(); ++i) {
      if (slots_[i]) {
        f(encode_handle(kind_, i, gens_[i]), *slots_[i]);
      }
    }
  }

  void clear() {
    slots_.clear();
    gens_.clear();
    free_.clear();
  }

 private:
  HKind kind_;
  std::vector<std::unique_ptr<T>> slots_;
  std::vector<uint16_t> gens_;
  std::vector<uint32_t> free_;
};

// ---------------------------------------------------------------------------
// Queued work.
// ---------------------------------------------------------------------------

struct Op {
  enum class Kind : uint8_t { kKernel, kCopy, kMemset, kEventRecord, kFree };

  Kind kind = Kind::kKernel;
  uint64_t duration_us = 0;  // how long it occupies the stream
  uint64_t start_us = 0;
  uint64_t end_us = 0;

  // kKernel
  void (*body)(void**) = nullptr;
  std::vector<void*> params;
  bool has_params = false;

  // kCopy
  void* dst = nullptr;
  const void* src = nullptr;
  size_t bytes = 0;

  // kMemset
  uint32_t pattern = 0;
  uint8_t pattern_width = 0;  // 1 or 4 bytes
  size_t count = 0;

  // kEventRecord
  void* event = nullptr;

  // kFree
  uintptr_t free_ptr = 0;
};

// ---------------------------------------------------------------------------
// Simulated objects.
// ---------------------------------------------------------------------------

struct Function {
  std::string name;
  void (*body)(void**) = nullptr;
  uint64_t duration_us = 0;
  uint32_t param_count = 0;
};

struct Module {
  std::vector<void*> functions;
};

struct Event {
  unsigned int flags = 0;
  bool recorded = false;
  bool complete = false;
  uint64_t time_us = 0;
};

struct Stream {
  void* ctx = nullptr;
  unsigned int flags = 0;
  int priority = 0;
  bool legacy_default = false;
  bool per_thread_default = false;

  uint64_t ready_at_us = 0;
  std::deque<Op> ops;
  uint64_t launches = 0;
  uint64_t seq_enqueued = 0;
  uint64_t seq_retired = 0;

  CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
  void* capture_graph = nullptr;
  uint64_t capture_id = 0;
};

struct Context {
  CUdevice dev = 0;
  unsigned int flags = 0;
  bool primary = false;
  int primary_refs = 0;
  void* legacy_stream = nullptr;
  std::unordered_map<uint64_t, void*> per_thread_streams;
  std::vector<void*> owned_streams;
};

struct Graph {
  std::vector<Op> nodes;
};

struct GraphExec {
  std::vector<Op> nodes;
};

struct MemHandleObj {
  size_t size = 0;
  int refs = 0;
  int fd = -1;  // memfd backing the allocation, or -1 for anonymous backing
};

struct Allocation {
  size_t size = 0;
  size_t pitch = 0;
  bool managed = false;
  bool vmm = false;            // mapped through cuMemMap, so cuMemFree must refuse it
  void* vmm_handle = nullptr;  // the cuMemCreate handle this mapping holds a reference to
};

struct Reservation {
  size_t size = 0;
};

struct DeviceProps {
  std::string name;
  uint64_t total_mem = 0;
  int sm_count = 0;
  int cc_major = 0;
  int cc_minor = 0;
};

// ---------------------------------------------------------------------------
// The simulation.
// ---------------------------------------------------------------------------

// Sim is a plain aggregate on purpose. It is not a public abstraction: it is
// the library's own state, and entry_points.cc manipulates it directly under
// Sim::mu rather than through a wall of one-line accessors that would each
// have to document the same locking rule. Nothing outside this library can
// reach it -- the header is private to fake_driver/src and every symbol in it
// is hidden.
struct Sim {
  static Sim& get();

  // Guards every member below. There is no second lock.
  std::mutex mu;
  std::condition_variable cv_work;  // wakes the worker
  std::condition_variable cv_done;  // wakes threads waiting for completion

  bool initialized = false;
  int driver_version = 0;
  DeviceProps device;
  uint64_t default_kernel_duration_us = 0;

  // Clock.
  bool manual_time = false;
  uint64_t clock_base_us = 0;
  uint64_t clock_epoch_ns = 0;

  // Object tables.
  Table<Context> contexts{HKind::kContext};
  Table<Stream> streams{HKind::kStream};
  Table<Event> events{HKind::kEvent};
  Table<Module> modules{HKind::kModule};
  Table<Function> functions{HKind::kFunction};
  Table<Graph> graphs{HKind::kGraph};
  Table<GraphExec> graph_execs{HKind::kGraphExec};
  Table<MemHandleObj> mem_handles{HKind::kMemHandle};

  std::unordered_map<std::string, void*> functions_by_name;
  void* primary_context = nullptr;

  // Device memory: key is the host address that CUdeviceptr carries.
  std::map<uintptr_t, Allocation> allocs;
  std::map<uintptr_t, Reservation> reservations;
  uint64_t allocated_bytes = 0;

  // The version-1 entry points take 32-bit device pointers, which cannot hold
  // a host address. Each allocation made through a v1 entry point therefore
  // gets a 32-bit token, and the v1 entry points translate it back. A token is
  // only meaningful to the v1 entry points, exactly as a v1 CUdeviceptr was
  // only meaningful to the v1 API.
  // The low byte is deliberately not 0: every real allocation this library
  // hands out is 256-byte aligned, so a token can never be mistaken for one.
  static constexpr uint32_t kFirstV1Token = 0x10000040;
  std::unordered_map<uint32_t, uintptr_t> v1_alias;
  uint32_t next_v1_token = kFirstV1Token;

  uint64_t next_capture_id = 1;

  // Worker. `worker_gen` is bumped to retire the current worker: the loop runs
  // only while the generation it started with is still current, which lets
  // stop_worker() release the lock to join without racing a restart.
  static constexpr size_t kMaxAbandonedWorkers = 64;
  bool worker_running = false;
  uint64_t worker_gen = 0;
  std::thread* worker = nullptr;
  // Worker threads lost to fork. They do not exist in this process and must
  // never be joined; they are kept reachable so LeakSanitizer stays quiet, in a
  // fixed-size array so the fork handler allocates nothing.
  std::thread* abandoned_workers[kMaxAbandonedWorkers] = {};
  size_t n_abandoned = 0;

  // Fork.
  bool in_forked_child = false;
  bool exit_dump_enabled = true;

  // --- clock -------------------------------------------------------------
  uint64_t now_us_locked() const;
  void set_manual_locked(bool manual);
  void jump_to_locked(uint64_t target_us);

  // --- worker ------------------------------------------------------------
  void ensure_worker_locked();
  void stop_worker(std::unique_lock<std::mutex>& lk);

  // --- streams -----------------------------------------------------------
  // Appends `op` to the stream, filling in start_us/end_us from the stream's
  // current tail. Returns the completion time.
  uint64_t enqueue_locked(Stream& s, Op op);
  // Waits for everything currently queued on `s`. Jumps the clock in manual
  // mode. Returns false only if the simulation is shutting down.
  bool drain_stream_locked(std::unique_lock<std::mutex>& lk, void* stream_handle);
  bool drain_context_locked(std::unique_lock<std::mutex>& lk, void* ctx_handle);
  bool wait_event_locked(std::unique_lock<std::mutex>& lk, void* event_handle);
  void quiesce_locked(std::unique_lock<std::mutex>& lk);

  // --- contexts ----------------------------------------------------------
  void* legacy_stream_of_locked(void* ctx_handle);
  void* per_thread_stream_of_locked(void* ctx_handle);
  // Resolves a stream argument. `per_thread` says whether the entry point that
  // was called is a _ptsz/_ptds one. Records the answer for
  // tf_last_stream_was_per_thread_default().
  Stream* resolve_stream_locked(void* ctx_handle, CUstream arg, bool per_thread, void** out_handle);

  // --- memory ------------------------------------------------------------
  // Finds the allocation containing [addr, addr+bytes).
  bool range_is_device_locked(uintptr_t addr, size_t bytes) const;
  void free_all_memory_locked();
  // Allocates zeroed, 256-byte-aligned host memory and charges it against the
  // simulated device's memory budget. Returns nullptr when the budget is used
  // up or the host allocation fails.
  void* device_alloc_locked(size_t bytes);
  // A memfd sized to `bytes`, so that one cuMemCreate allocation mapped at two
  // addresses really does alias. Returns -1 when memfd is unavailable, in which
  // case cuMemMap falls back to an anonymous mapping.
  static int make_backing_fd(size_t bytes);

  // --- lifecycle ---------------------------------------------------------
  void reset();

  // Only Sim::get() constructs one, and nothing ever destroys it.
  Sim();
  void worker_loop(uint64_t gen);
  void retire_head_locked(Stream& s);
};

// The per-thread context stack and the thread's identity.
uint64_t this_thread_id();

// The calling thread's CUDA context stack. CUDA keeps one per thread and so
// do we; none of this touches Sim, so none of it needs the simulation lock.
void* current_context();
void push_current(void* ctx);
void* pop_current();
void set_current(void* ctx);
void forget_context(void* ctx);  // a destroyed context leaves this thread's stack
void clear_context_stack();

// Set by resolve_stream_locked; read by tf_last_stream_was_per_thread_default.
void set_last_stream_was_per_thread(bool v);
bool last_stream_was_per_thread();

}  // namespace tf

#endif  // TESSERA_FAKE_SIM_H
