// Shared scaffolding for the M0 acceptance matrix (T1-T7 of docs/gates/M0.md).
//
// Every acceptance test is the same shape: run tests/shim/acceptance_app_main
// in a child process whose environment decides which library its driver calls
// land in, then compare the three JSON files that come back -- the
// application's own record of what it issued, the shim's counters, and the
// fake driver's counters. The tests themselves link neither the shim nor the
// fake, so nothing they measure can come from their own address space.
//
// The three arms:
//
//   kMasquerade  the shim is libcuda.so.1, first on LD_LIBRARY_PATH; it loads
//                the fake as the real driver through TESSERA_REAL_LIBCUDA
//   kPreload     the shim is LD_PRELOAD'd as libtessera.so and the fake is the
//                libcuda.so.1 the application links against
//   kNoShim      no shim at all: the application talks straight to the fake
//
// kNoShim is not a third strategy, it is the negative control. Every
// interception claim is paired with it, because an assertion that holds with
// and without the shim proves nothing about the shim (docs/gates/M0.md,
// "METHOD REQUIREMENTS").
#ifndef TESSERA_TESTS_SHIM_ACCEPTANCE_FIXTURE_H
#define TESSERA_TESTS_SHIM_ACCEPTANCE_FIXTURE_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "support/counters.h"
#include "support/spawn.h"

#if !defined(TESSERA_ACCEPTANCE_APP) || !defined(TESSERA_SHIM_MASQ_DIR) ||    \
    !defined(TESSERA_SHIM_PRELOAD_PATH) || !defined(TESSERA_FAKE_MASQ_DIR) || \
    !defined(TESSERA_FAKE_MASQ_LIB)
#error "acceptance tests must be declared with _tessera_acceptance_test()"
#endif

namespace tessera_test {

enum class Strategy { kMasquerade, kPreload, kNoShim };

inline const char* strategy_name(Strategy strategy) {
  switch (strategy) {
    case Strategy::kMasquerade:
      return "masquerade";
    case Strategy::kPreload:
      return "preload";
    case Strategy::kNoShim:
      return "no-shim (control)";
  }
  return "?";
}

inline bool has_shim(Strategy strategy) {
  return strategy != Strategy::kNoShim;
}

// Everything one run produced.
struct RunOutcome {
  Strategy strategy = Strategy::kNoShim;
  SpawnResult spawn;
  std::string app_json;
  std::string shim_json;
  std::string fake_json;
  std::string child_json;

  // cu* entry points only, keyed by the EXPORTED symbol name each call
  // reached, so the three maps are directly comparable.
  CounterMap issued;        // what the application says it called
  CounterMap shim_symbols;  // what the shim counted
  CounterMap fake_symbols;  // what the driver underneath actually received

  bool shim_file_written = false;

  std::optional<uint64_t> app_number(const std::string& key) const {
    return json_number(app_json, key);
  }
  std::optional<uint64_t> child_number(const std::string& key) const {
    return json_number(child_json, key);
  }
  std::optional<uint64_t> shim_number(const std::string& key) const {
    return json_number(shim_json, key);
  }

  // A run is only evidence if the application actually did its work.
  bool app_completed() const { return spawn.ok() && app_number("exit_code").value_or(1) == 0; }

  std::string describe() const {
    return std::string(strategy_name(strategy)) + ": " + spawn.command +
           (spawn.output.empty() ? "" : "\n--- child output ---\n" + spawn.output);
  }
};

struct RunRequest {
  Strategy strategy = Strategy::kNoShim;
  std::vector<std::string> args;
  std::vector<std::string> env;     // extra overrides, layered last
  std::vector<std::string> prefix;  // e.g. {"strace", "-f", ...}
  int timeout_ms = 120000;
  // Where the shim should find the real driver. Only the self-reference test
  // (T6) changes it.
  std::string real_libcuda = TESSERA_FAKE_MASQ_LIB;
};

inline RunOutcome run_acceptance_app(const RunRequest& request) {
  RunOutcome outcome;
  outcome.strategy = request.strategy;

  const std::string app_report = spawn_detail::unique_path("app");
  const std::string child_report = spawn_detail::unique_path("child");
  const std::string shim_stats = spawn_detail::unique_path("shim");
  const std::string fake_stats = spawn_detail::unique_path("fake");

  Spawn spawn;
  spawn.program = TESSERA_ACCEPTANCE_APP;
  spawn.args = request.args;
  spawn.prefix = request.prefix;
  spawn.timeout_ms = request.timeout_ms;

  spawn.env = {
      "TESSERA_APP_REPORT=" + app_report,
      "TESSERA_APP_CHILD_REPORT=" + child_report,
      "TESSERA_FAKE_STATS_FILE=" + fake_stats,
      "TESSERA_REAL_LIBCUDA=" + request.real_libcuda,
  };

  switch (request.strategy) {
    case Strategy::kMasquerade:
      // The shim's directory holds only libcuda.so.1 and its symlink, so this
      // one entry decides what DT_NEEDED libcuda.so.1 resolves to.
      spawn.env.push_back("LD_LIBRARY_PATH=" TESSERA_SHIM_MASQ_DIR);
      spawn.env.push_back("LD_PRELOAD=");
      spawn.env.push_back("TESSERA_STATS_FILE=" + shim_stats);
      break;
    case Strategy::kPreload:
      spawn.env.push_back("LD_LIBRARY_PATH=" TESSERA_FAKE_MASQ_DIR);
      spawn.env.push_back("LD_PRELOAD=" TESSERA_SHIM_PRELOAD_PATH);
      spawn.env.push_back("TESSERA_STATS_FILE=" + shim_stats);
      break;
    case Strategy::kNoShim:
      spawn.env.push_back("LD_LIBRARY_PATH=" TESSERA_FAKE_MASQ_DIR);
      spawn.env.push_back("LD_PRELOAD=");
      // No TESSERA_STATS_FILE: there is no shim to write one.
      break;
  }
  for (const std::string& entry : request.env) {
    spawn.env.push_back(entry);
  }

  outcome.spawn = run_child(spawn);
  outcome.app_json = read_text_file(app_report);
  outcome.child_json = read_text_file(child_report);
  outcome.fake_json = read_text_file(fake_stats);
  outcome.shim_json = read_text_file(shim_stats);
  outcome.shim_file_written = !outcome.shim_json.empty();

  outcome.issued = driver_counts(outcome.app_json, "issued");
  outcome.shim_symbols = driver_counts(outcome.shim_json, "symbols");
  outcome.fake_symbols = driver_counts(outcome.fake_json, "");

  ::unlink(app_report.c_str());
  ::unlink(child_report.c_str());
  ::unlink(shim_stats.c_str());
  ::unlink(fake_stats.c_str());
  return outcome;
}

inline RunOutcome run_workload(Strategy strategy, const std::string& path) {
  RunRequest request;
  request.strategy = strategy;
  request.args = {"workload", path};
  return run_acceptance_app(request);
}

// The four access paths of ADR-001. Named as the application names them.
inline const std::vector<std::string>& access_paths() {
  static const std::vector<std::string> paths = {"direct", "dlopen", "gpa1", "gpa2"};
  return paths;
}

}  // namespace tessera_test

#endif  // TESSERA_TESTS_SHIM_ACCEPTANCE_FIXTURE_H
