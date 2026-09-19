// The per-tenant shared-memory page: the only channel on the shim's hot path.
//
// One 4 KiB page per tenant, created by tesserad at
// /dev/shm/tessera/<gpu>/<tenant> and mmap'd MAP_SHARED by the shim. The shim
// reads and updates it with relaxed/acq-rel atomics only: no locks, no
// syscalls, no allocation (I-4).
//
// Layout rules:
//   - Fields the daemon writes and the shim reads on every launch live on the
//     first cache line. Telemetry the shim writes lives on a separate line, so
//     the daemon's refills never invalidate the shim's counter line.
//   - The layout is versioned. A shim that sees an unknown layout_version or a
//     bad magic fails open (I-5) rather than guessing.
//   - Every field is an atomic of a lock-free type, and the struct is standard
//     layout, because two processes built by two compilers map the same bytes.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace tessera {

inline constexpr std::size_t kTenantPageSize = 4096;
inline constexpr std::size_t kCacheLine = 64;
// "TSRA": identifies the page and catches a stale or foreign mapping.
inline constexpr std::uint32_t kTenantPageMagic = 0x54535241U;
inline constexpr std::uint32_t kTenantPageLayoutVersion = 1U;

// Bits in TenantPage::policy_flags.
enum PolicyFlag : std::uint32_t {
  // Tenant is exempt from gating (the interactive class).
  kPolicyUngated = 1U << 0,
  // Daemon is shutting down; shim should fail open without waiting.
  kPolicyDrain = 1U << 1,
  // Memory quota enforcement is active.
  kPolicyMemQuota = 1U << 2,
};

struct TenantPage {
  // --- Line 0: written by the daemon, read by the shim on every launch. ---
  alignas(kCacheLine) std::atomic<std::uint32_t> magic;
  std::atomic<std::uint32_t> layout_version;
  // Estimated GPU microseconds this tenant may still launch. May dip negative:
  // a launch commits its estimate before it knows the true cost.
  std::atomic<std::int64_t> credits_us;
  // Bumped by the daemon on every refill; the address the shim futex-waits on.
  std::atomic<std::uint32_t> futex_word;
  // Bumped every daemon tick. If it stops advancing, the shim fails open.
  std::atomic<std::uint32_t> daemon_epoch;
  // CLOCK_MONOTONIC nanoseconds at the last tick. Lets a shim that has just
  // mapped the page judge staleness without first observing two epochs.
  std::atomic<std::int64_t> epoch_mono_ns;
  std::atomic<std::uint32_t> policy_flags;
  // Priority class for streams this tenant creates (lever 2, shared mode).
  std::atomic<std::int32_t> stream_priority;

  // --- Line 1: memory accounting, written by the shim, read by the daemon. ---
  alignas(kCacheLine) std::atomic<std::int64_t> mem_used;
  // Written by the daemon; read by the shim to virtualize cuMemGetInfo.
  std::atomic<std::int64_t> mem_quota;

  // --- Line 2: telemetry, written by the shim only. ---
  alignas(kCacheLine) std::atomic<std::uint64_t> launched;
  std::atomic<std::uint64_t> gated;
  std::atomic<std::uint64_t> stalled_us;
  std::atomic<std::uint64_t> sampled;
};

static_assert(sizeof(TenantPage) <= kTenantPageSize,
              "TenantPage must fit in one 4 KiB page");
static_assert(alignof(TenantPage) == kCacheLine);
static_assert(std::is_standard_layout_v<TenantPage>,
              "TenantPage is mapped by separately compiled processes");

// The hot path may not fall back to a lock for any field.
static_assert(std::atomic<std::int32_t>::is_always_lock_free);
static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
static_assert(std::atomic<std::int64_t>::is_always_lock_free);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);

// The daemon must be able to futex-wait on futex_word: the kernel requires a
// naturally aligned 32-bit word.
static_assert(offsetof(TenantPage, futex_word) % sizeof(std::uint32_t) == 0);

// Fields the shim reads per launch must share one cache line with each other,
// and must not share one with the telemetry the shim writes.
static_assert(offsetof(TenantPage, credits_us) / kCacheLine ==
                  offsetof(TenantPage, daemon_epoch) / kCacheLine,
              "per-launch reads should touch a single cache line");
static_assert(offsetof(TenantPage, launched) / kCacheLine !=
                  offsetof(TenantPage, credits_us) / kCacheLine,
              "shim telemetry must not false-share with daemon refills");

}  // namespace tessera
