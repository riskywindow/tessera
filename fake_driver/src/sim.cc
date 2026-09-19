#include "sim.h"

#include <pthread.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <new>
#include <utility>

#include <sys/mman.h>
#include <sys/syscall.h>

#include "counters.h"
#include "tessera_fake/control.h"

namespace tf {
namespace {

constexpr uint64_t kDefaultTotalMem = UINT64_C(16) << 30;  // 16 GiB
constexpr int kDefaultSmCount = 58;
constexpr int kDefaultCcMajor = 8;
constexpr int kDefaultCcMinor = 9;
constexpr const char* kDeviceName = "Tessera Fake Device";
constexpr size_t kAllocAlignment = 256;

uint64_t mono_ns() {
  struct timespec ts = {};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    // CLOCK_MONOTONIC cannot fail on Linux; if it somehow did, a clock that
    // stands still is far better than one that reads uninitialised memory.
    return 0;
  }
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000u + static_cast<uint64_t>(ts.tv_nsec);
}

std::atomic<uint64_t> g_next_thread_id{1};
std::atomic<int> g_last_stream_per_thread{0};

std::vector<void*>& context_stack() {
  static thread_local std::vector<void*> stack;
  return stack;
}

void atfork_prepare();
void atfork_parent();
void atfork_child();
void at_exit_handler();

}  // namespace

uint64_t this_thread_id() {
  static const thread_local uint64_t id = g_next_thread_id.fetch_add(1, std::memory_order_relaxed);
  return id;
}

void set_last_stream_was_per_thread(bool v) {
  g_last_stream_per_thread.store(v ? 1 : 0, std::memory_order_relaxed);
}

bool last_stream_was_per_thread() {
  return g_last_stream_per_thread.load(std::memory_order_relaxed) != 0;
}

// ---------------------------------------------------------------------------
// Construction. The Sim is allocated once and never destroyed: a driver that
// tears itself down during static destruction would be a source of crashes in
// exactly the tests this library exists to support. The pointer stays in a
// namespace-scope variable so LeakSanitizer still sees it as reachable.
// ---------------------------------------------------------------------------

Sim::Sim() {
  driver_version = CUDA_VERSION;
  device.name = kDeviceName;
  device.total_mem = kDefaultTotalMem;
  device.sm_count = kDefaultSmCount;
  device.cc_major = kDefaultCcMajor;
  device.cc_minor = kDefaultCcMinor;
  clock_epoch_ns = mono_ns();
}

Sim& Sim::get() {
  static Sim* instance = [] {
    Sim* s = new Sim();
    // Neither registration can fail in any way this library could recover
    // from, but an unregistered atfork handler would mean a child that
    // deadlocks, so say so rather than continuing quietly.
    if (pthread_atfork(atfork_prepare, atfork_parent, atfork_child) != 0 ||
        std::atexit(at_exit_handler) != 0) {
      const int written =
          std::fputs("libcuda_fake: could not install fork/exit handlers\n", stderr);
      static_cast<void>(written);
      std::abort();
    }
    return s;
  }();
  return *instance;
}

void library_init_once() {
  static_cast<void>(Sim::get());
}

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------

uint64_t Sim::now_us_locked() const {
  if (manual_time) {
    return clock_base_us;
  }
  const uint64_t elapsed_ns = mono_ns() - clock_epoch_ns;
  return clock_base_us + elapsed_ns / 1000u;
}

void Sim::set_manual_locked(bool manual) {
  const uint64_t now = now_us_locked();
  manual_time = manual;
  clock_base_us = now;
  clock_epoch_ns = mono_ns();
}

void Sim::jump_to_locked(uint64_t target_us) {
  if (!manual_time) {
    return;
  }
  if (target_us > clock_base_us) {
    clock_base_us = target_us;
  }
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

void Sim::ensure_worker_locked() {
  if (worker_running) {
    return;
  }
  if (worker != nullptr) {
    // A worker lost to fork. It does not exist in this process; never join it.
    if (n_abandoned < kMaxAbandonedWorkers) {
      abandoned_workers[n_abandoned++] = worker;
    }
    worker = nullptr;
  }
  worker_running = true;
  const uint64_t gen = worker_gen;
  worker = new std::thread([this, gen] { this->worker_loop(gen); });
}

void Sim::stop_worker(std::unique_lock<std::mutex>& lk) {
  if (worker == nullptr) {
    worker_running = false;
    return;
  }
  std::thread* t = worker;
  worker = nullptr;
  worker_running = false;
  ++worker_gen;
  cv_work.notify_all();
  lk.unlock();
  t->join();
  delete t;
  lk.lock();
}

void Sim::worker_loop(uint64_t gen) {
  std::unique_lock<std::mutex> lk(mu);
  while (gen == worker_gen) {
    const uint64_t now = now_us_locked();
    Stream* due = nullptr;
    uint64_t earliest = UINT64_MAX;
    streams.for_each([&](void*, Stream& s) {
      if (s.ops.empty()) {
        return;
      }
      const uint64_t end = s.ops.front().end_us;
      if (end < earliest) {
        earliest = end;
        due = &s;
      }
    });
    if (due == nullptr) {
      cv_work.wait(lk);
      continue;
    }
    if (earliest <= now) {
      retire_head_locked(*due);
      cv_done.notify_all();
      continue;
    }
    if (manual_time) {
      cv_work.wait(lk);
    } else {
      cv_work.wait_for(lk, std::chrono::microseconds(earliest - now));
    }
  }
  cv_done.notify_all();
}

// Runs one operation's effect and removes it from its stream. The simulation
// lock is held throughout, which is what gives a test that reads a buffer after
// cuStreamSynchronize a happens-before edge to the kernel body that wrote it.
// It is also why a kernel body must not call back into the driver.
void Sim::retire_head_locked(Stream& s) {
  Op& op = s.ops.front();
  switch (op.kind) {
    case Op::Kind::kKernel:
      if (op.body != nullptr) {
        op.body(op.has_params ? op.params.data() : nullptr);
      }
      break;
    case Op::Kind::kCopy:
      if (op.bytes != 0) {
        std::memcpy(op.dst, op.src, op.bytes);
      }
      break;
    case Op::Kind::kMemset:
      if (op.pattern_width == 1) {
        std::memset(op.dst, static_cast<int>(op.pattern & 0xFF), op.count);
      } else {
        auto* words = static_cast<uint32_t*>(op.dst);
        for (size_t i = 0; i < op.count; ++i) {
          words[i] = op.pattern;
        }
      }
      break;
    case Op::Kind::kEventRecord: {
      Event* e = events.get(op.event);
      if (e != nullptr) {
        e->complete = true;
        e->time_us = op.end_us;
      }
      break;
    }
    case Op::Kind::kFree: {
      auto it = allocs.find(op.free_ptr);
      if (it != allocs.end()) {
        allocated_bytes -= it->second.size;
        allocs.erase(it);
        std::free(address_to_pointer(op.free_ptr));
      }
      break;
    }
  }
  s.ops.pop_front();
  ++s.seq_retired;
}

// ---------------------------------------------------------------------------
// Stream queueing and waiting
// ---------------------------------------------------------------------------

uint64_t Sim::enqueue_locked(Stream& s, Op op) {
  const uint64_t now = now_us_locked();
  const uint64_t start = s.ready_at_us > now ? s.ready_at_us : now;
  op.start_us = start;
  op.end_us = start + op.duration_us;
  s.ready_at_us = op.end_us;
  ++s.seq_enqueued;
  const uint64_t end = op.end_us;
  s.ops.push_back(std::move(op));
  ensure_worker_locked();
  cv_work.notify_all();
  return end;
}

bool Sim::drain_stream_locked(std::unique_lock<std::mutex>& lk, void* stream_handle) {
  Stream* s = streams.get(stream_handle);
  if (s == nullptr) {
    return false;
  }
  const uint64_t target_seq = s->seq_enqueued;
  if (manual_time) {
    jump_to_locked(s->ready_at_us);
  }
  ensure_worker_locked();
  cv_work.notify_all();
  cv_done.wait(lk, [&] {
    const Stream* cur = streams.get(stream_handle);
    return cur == nullptr || cur->seq_retired >= target_seq || !worker_running;
  });
  return true;
}

bool Sim::drain_context_locked(std::unique_lock<std::mutex>& lk, void* ctx_handle) {
  std::vector<std::pair<void*, uint64_t>> targets;
  uint64_t latest = 0;
  streams.for_each([&](void* h, Stream& s) {
    if (s.ctx != ctx_handle) {
      return;
    }
    targets.emplace_back(h, s.seq_enqueued);
    if (s.ready_at_us > latest) {
      latest = s.ready_at_us;
    }
  });
  if (targets.empty()) {
    return true;
  }
  if (manual_time) {
    jump_to_locked(latest);
  }
  ensure_worker_locked();
  cv_work.notify_all();
  cv_done.wait(lk, [&] {
    if (!worker_running) {
      return true;
    }
    for (const auto& t : targets) {
      const Stream* cur = streams.get(t.first);
      if (cur != nullptr && cur->seq_retired < t.second) {
        return false;
      }
    }
    return true;
  });
  return true;
}

bool Sim::wait_event_locked(std::unique_lock<std::mutex>& lk, void* event_handle) {
  Event* e = events.get(event_handle);
  if (e == nullptr) {
    return false;
  }
  if (!e->recorded || e->complete) {
    return true;
  }
  if (manual_time) {
    jump_to_locked(e->time_us);
  }
  ensure_worker_locked();
  cv_work.notify_all();
  cv_done.wait(lk, [&] {
    const Event* cur = events.get(event_handle);
    return cur == nullptr || cur->complete || !worker_running;
  });
  return true;
}

void Sim::quiesce_locked(std::unique_lock<std::mutex>& lk) {
  ensure_worker_locked();
  cv_work.notify_all();
  cv_done.wait(lk, [&] {
    if (!worker_running) {
      return true;
    }
    const uint64_t now = now_us_locked();
    bool due = false;
    streams.for_each([&](void*, Stream& s) {
      if (!s.ops.empty() && s.ops.front().end_us <= now) {
        due = true;
      }
    });
    return !due;
  });
}

// ---------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------

void* current_context() {
  const std::vector<void*>& stack = context_stack();
  return stack.empty() ? nullptr : stack.back();
}

void push_current(void* ctx) {
  context_stack().push_back(ctx);
}

void* pop_current() {
  std::vector<void*>& stack = context_stack();
  if (stack.empty()) {
    return nullptr;
  }
  void* top = stack.back();
  stack.pop_back();
  return top;
}

void set_current(void* ctx) {
  std::vector<void*>& stack = context_stack();
  if (ctx == nullptr) {
    if (!stack.empty()) {
      stack.pop_back();
    }
    return;
  }
  if (stack.empty()) {
    stack.push_back(ctx);
  } else {
    stack.back() = ctx;
  }
}

void forget_context(void* ctx) {
  std::vector<void*>& stack = context_stack();
  stack.erase(std::remove(stack.begin(), stack.end(), ctx), stack.end());
}

void clear_context_stack() {
  context_stack().clear();
}

void* Sim::legacy_stream_of_locked(void* ctx_handle) {
  Context* c = contexts.get(ctx_handle);
  if (c == nullptr) {
    return nullptr;
  }
  if (c->legacy_stream == nullptr) {
    auto s = std::make_unique<Stream>();
    s->ctx = ctx_handle;
    s->legacy_default = true;
    c->legacy_stream = streams.create(std::move(s));
  }
  return c->legacy_stream;
}

void* Sim::per_thread_stream_of_locked(void* ctx_handle) {
  Context* c = contexts.get(ctx_handle);
  if (c == nullptr) {
    return nullptr;
  }
  const uint64_t tid = this_thread_id();
  auto it = c->per_thread_streams.find(tid);
  if (it != c->per_thread_streams.end()) {
    return it->second;
  }
  auto s = std::make_unique<Stream>();
  s->ctx = ctx_handle;
  s->per_thread_default = true;
  void* h = streams.create(std::move(s));
  c->per_thread_streams.emplace(tid, h);
  return h;
}

Stream* Sim::resolve_stream_locked(void* ctx_handle, CUstream arg, bool per_thread,
                                   void** out_handle) {
  const uintptr_t raw = reinterpret_cast<uintptr_t>(arg);
  void* h = nullptr;
  if (raw == 0) {
    h = per_thread ? per_thread_stream_of_locked(ctx_handle) : legacy_stream_of_locked(ctx_handle);
  } else if (raw == kStreamLegacyHandle) {
    h = legacy_stream_of_locked(ctx_handle);
  } else if (raw == kStreamPerThreadHandle) {
    h = per_thread_stream_of_locked(ctx_handle);
  } else {
    h = arg;
  }
  Stream* s = streams.get(h);
  if (s == nullptr) {
    return nullptr;
  }
  set_last_stream_was_per_thread(s->per_thread_default);
  if (out_handle != nullptr) {
    *out_handle = h;
  }
  return s;
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

bool Sim::range_is_device_locked(uintptr_t addr, size_t bytes) const {
  if (allocs.empty()) {
    return false;
  }
  auto it = allocs.upper_bound(addr);
  if (it == allocs.begin()) {
    return false;
  }
  --it;
  const uintptr_t base = it->first;
  const size_t size = it->second.size;
  return addr >= base && bytes <= size && addr - base <= size - bytes;
}

void Sim::free_all_memory_locked() {
  for (auto& entry : allocs) {
    if (!entry.second.vmm) {
      std::free(address_to_pointer(entry.first));
    }
  }
  allocs.clear();
  for (auto& entry : reservations) {
    munmap(address_to_pointer(entry.first), entry.second.size);
  }
  reservations.clear();
  mem_handles.for_each([](void*, MemHandleObj& h) {
    if (h.fd >= 0) {
      close(h.fd);
    }
    h.fd = -1;
  });
  allocated_bytes = 0;
}

void* Sim::device_alloc_locked(size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  if (allocated_bytes + bytes > device.total_mem) {
    return nullptr;
  }
  const size_t rounded = (bytes + kAllocAlignment - 1) / kAllocAlignment * kAllocAlignment;
  void* p = nullptr;
  if (posix_memalign(&p, kAllocAlignment, rounded) != 0) {
    return nullptr;
  }
  std::memset(p, 0, rounded);
  Allocation a;
  a.size = rounded;
  allocs.emplace(reinterpret_cast<uintptr_t>(p), a);
  allocated_bytes += rounded;
  return p;
}

int Sim::make_backing_fd(size_t bytes) {
#if defined(SYS_memfd_create)
  const long fd = syscall(SYS_memfd_create, "tessera_fake_vmm", 0);
  if (fd >= 0) {
    if (ftruncate(static_cast<int>(fd), static_cast<off_t>(bytes)) == 0) {
      return static_cast<int>(fd);
    }
    close(static_cast<int>(fd));
  }
#else
  static_cast<void>(bytes);
#endif
  return -1;
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

void Sim::reset() {
  std::unique_lock<std::mutex> lk(mu);
  stop_worker(lk);

  streams.for_each([](void*, Stream& s) { s.ops.clear(); });
  free_all_memory_locked();

  graph_execs.clear();
  graphs.clear();
  events.clear();
  streams.clear();
  contexts.clear();
  modules.clear();
  functions.clear();
  mem_handles.clear();
  functions_by_name.clear();
  v1_alias.clear();
  next_v1_token = kFirstV1Token;
  primary_context = nullptr;
  next_capture_id = 1;

  initialized = false;
  driver_version = CUDA_VERSION;
  device.name = kDeviceName;
  device.total_mem = kDefaultTotalMem;
  device.sm_count = kDefaultSmCount;
  device.cc_major = kDefaultCcMajor;
  device.cc_minor = kDefaultCcMinor;
  default_kernel_duration_us = 0;

  manual_time = false;
  clock_base_us = 0;
  clock_epoch_ns = mono_ns();

  for (uint32_t i = 0; i < TF_SYM_COUNT; ++i) {
    g_counts[i].store(0, std::memory_order_relaxed);
    g_forced[i].store(0, std::memory_order_relaxed);
  }
  g_fork_emulate_real.store(0, std::memory_order_relaxed);
  set_last_stream_was_per_thread(false);

  lk.unlock();
  clear_context_stack();
}

// ---------------------------------------------------------------------------
// fork and exit
// ---------------------------------------------------------------------------

namespace {

void atfork_prepare() {
  Sim::get().mu.lock();
}

void atfork_parent() {
  Sim::get().mu.unlock();
}

// Runs in the child, single-threaded, with the simulation lock held by this
// thread. The worker thread did not survive the fork, so it is abandoned (never
// joined) and recreated lazily on the child's next call. Without this the
// child's first cuStreamSynchronize would wait forever on work no thread is
// left to complete.
void atfork_child() {
  Sim& s = Sim::get();
  if (s.worker != nullptr) {
    if (s.n_abandoned < Sim::kMaxAbandonedWorkers) {
      s.abandoned_workers[s.n_abandoned++] = s.worker;
    }
    s.worker = nullptr;
  }
  s.worker_running = false;
  ++s.worker_gen;
  // The condition variables were copied mid-wait: their internal bookkeeping
  // still counts the worker that did not survive the fork, which can cost the
  // child a wakeup. Re-constructing them in place resets them to their
  // static-initializer state. They are deliberately NOT destroyed first:
  // pthread_cond_destroy blocks until every recorded waiter has left, and the
  // waiter recorded here is a thread that does not exist in this process, so
  // destroying would hang the child inside fork(). There is nothing to release
  // either, because a glibc condition variable owns no other resource.
  new (&s.cv_work) std::condition_variable();
  new (&s.cv_done) std::condition_variable();
  s.in_forked_child = true;
  s.exit_dump_enabled = false;
  g_in_forked_child.store(1, std::memory_order_relaxed);
  for (uint32_t i = 0; i < TF_SYM_COUNT; ++i) {
    g_counts[i].store(0, std::memory_order_relaxed);
  }
  s.mu.unlock();
}

void at_exit_handler() {
  Sim& s = Sim::get();
  if (s.exit_dump_enabled) {
    // Reading the environment is only unsafe against a concurrent setenv. This
    // runs from an atexit handler, after main has returned, and the value is
    // used immediately and never kept.
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char* path = std::getenv("TESSERA_FAKE_STATS_FILE");
    if (path != nullptr && path[0] != '\0') {
      tf_dump_stats(path);
    }
  }
  std::unique_lock<std::mutex> lk(s.mu);
  s.stop_worker(lk);
}

}  // namespace
}  // namespace tf
