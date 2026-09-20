// Launching a child process from a test, the same way on every configuration.
//
// The acceptance suite's claims are about what happens to an *application*
// that does not know Tessera exists (I-2), so the application has to be a real
// process with a real environment: LD_LIBRARY_PATH, LD_PRELOAD and
// TESSERA_REAL_LIBCUDA are the interposition strategy, and they can only be
// set on a process that has not started yet.
//
// Three things every caller would otherwise get wrong, so they live here:
//
//   1. THE EMULATOR. A cross build's test binaries are aarch64 and run under
//      qemu-aarch64 because this host has no binfmt registration. CTest knows
//      that for the test binary itself; nothing tells the child. Every command
//      is therefore prefixed with TESSERA_TEST_EMULATOR (a space-separated
//      command prefix, empty for native builds), which cmake/TesseraTesting
//      passes in as a compile definition.
//   2. THE SANITIZER RUNTIME. gcc links libasan/libtsan as shared objects, and
//      the runtime must be the FIRST entry in LD_PRELOAD or the process dies
//      during startup. TESSERA_SANITIZER_PRELOAD holds that path (empty for
//      clang, which links its runtime statically, and for unsanitized builds);
//      it is prepended to whatever LD_PRELOAD the caller asked for.
//   3. A DEADLINE. T6 asserts that a misconfiguration produces an *error*
//      rather than a hang, which is only a test if the hang is caught here and
//      reported as a failure. Every run has a timeout; a child that overruns it
//      is killed and the result says so.
//
// Nothing here is Tessera-specific beyond those three rules.

#ifndef TESSERA_TESTS_SUPPORT_SPAWN_H
#define TESSERA_TESTS_SUPPORT_SPAWN_H

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/wait.h>

extern "C" char** environ;

#if !defined(TESSERA_TEST_EMULATOR)
#error "TESSERA_TEST_EMULATOR is not defined; link the test with tessera_add_gtest"
#endif
#if !defined(TESSERA_SANITIZER_PRELOAD)
#error "TESSERA_SANITIZER_PRELOAD is not defined; link the test with tessera_add_gtest"
#endif

namespace tessera_test {

// What to run.
struct Spawn {
  std::string program;              // absolute path of the target binary
  std::vector<std::string> args;    // argv[1...]
  std::vector<std::string> env;     // "KEY=VALUE" overrides on this process's environment
  std::vector<std::string> prefix;  // command words before the emulator, e.g. {"strace","-f"}
  int timeout_ms = 60000;           // hard deadline; the child is killed at it
  std::string output_path;          // where stdout+stderr go; a temp file when empty
};

// What happened.
struct SpawnResult {
  bool spawned = false;    // posix_spawn itself succeeded
  bool exited = false;     // exited normally (rather than dying on a signal)
  bool timed_out = false;  // the deadline expired and the child was killed
  int exit_code = -1;      // valid when exited
  int term_signal = 0;     // valid when !exited
  int64_t wall_us = 0;     // wall time from spawn to reap
  std::string command;     // the full command line, for failure messages
  std::string output;      // the child's merged stdout and stderr

  bool ok() const { return spawned && exited && !timed_out && exit_code == 0; }
};

namespace spawn_detail {

inline int64_t now_us() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (static_cast<int64_t>(ts.tv_sec) * 1000000) + (ts.tv_nsec / 1000);
}

// TESSERA_TEST_EMULATOR is a space-separated command prefix ("qemu-aarch64 -L
// /usr/aarch64-linux-gnu"), empty on a native build.
inline std::vector<std::string> emulator_words() {
  std::vector<std::string> words;
  std::istringstream text{std::string(TESSERA_TEST_EMULATOR)};
  std::string word;
  while (text >> word) {
    words.push_back(word);
  }
  return words;
}

inline std::string key_of(const std::string& entry) {
  const size_t eq = entry.find('=');
  return eq == std::string::npos ? entry : entry.substr(0, eq);
}

inline std::string unique_path(const char* tag) {
  static std::atomic<uint64_t> counter{0};
  std::ostringstream path;
  const char* dir = ::getenv("TMPDIR");
  path << (dir != nullptr && dir[0] != '\0' ? dir : "/tmp") << "/tessera_" << tag << "_"
       << static_cast<long>(::getpid()) << "_" << counter.fetch_add(1, std::memory_order_relaxed);
  return path.str();
}

inline std::string read_file(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return {};
  }
  std::string text;
  char buffer[4096];
  for (;;) {
    const ssize_t n = ::read(fd, buffer, sizeof(buffer));
    if (n <= 0) {
      break;
    }
    text.append(buffer, static_cast<size_t>(n));
  }
  ::close(fd);
  return text;
}

// The sanitizer runtime must precede every other preloaded object. Applied to
// whatever LD_PRELOAD the caller asked for, including none.
inline std::vector<std::string> with_sanitizer_preload(std::vector<std::string> env) {
  const std::string runtime = TESSERA_SANITIZER_PRELOAD;
  if (runtime.empty()) {
    return env;
  }
  for (std::string& entry : env) {
    if (key_of(entry) == "LD_PRELOAD") {
      const std::string value = entry.substr(std::strlen("LD_PRELOAD="));
      entry = "LD_PRELOAD=" + runtime + (value.empty() ? "" : " " + value);
      return env;
    }
  }
  env.push_back("LD_PRELOAD=" + runtime);
  return env;
}

}  // namespace spawn_detail

// Runs `request` to completion (or to its deadline) and returns what happened.
// The child's environment is this process's, with `request.env` layered over
// it by key, so a test changes only the variables it names.
inline SpawnResult run_child(const Spawn& request) {
  SpawnResult result;
  const std::vector<std::string> env = spawn_detail::with_sanitizer_preload(request.env);

  // argv: [prefix...] [emulator...] program [args...]
  std::vector<std::string> words = request.prefix;
  for (const std::string& word : spawn_detail::emulator_words()) {
    words.push_back(word);
  }
  words.push_back(request.program);
  for (const std::string& arg : request.args) {
    words.push_back(arg);
  }

  std::ostringstream printable;
  std::vector<char*> argv;
  argv.reserve(words.size() + 1);
  for (std::string& word : words) {
    printable << word << ' ';
    argv.push_back(word.data());
  }
  argv.push_back(nullptr);
  result.command = printable.str();

  // Overrides win; everything else is inherited unchanged.
  std::vector<std::string> storage = env;
  std::vector<char*> envp;
  for (char** entry = environ; *entry != nullptr; ++entry) {
    const std::string key = spawn_detail::key_of(*entry);
    bool overridden = false;
    for (const std::string& override_entry : storage) {
      if (spawn_detail::key_of(override_entry) == key) {
        overridden = true;
        break;
      }
    }
    if (!overridden) {
      envp.push_back(*entry);
    }
  }
  for (std::string& entry : storage) {
    envp.push_back(entry.data());
  }
  envp.push_back(nullptr);

  const std::string output_path =
      request.output_path.empty() ? spawn_detail::unique_path("child_output") : request.output_path;

  posix_spawn_file_actions_t actions;
  if (posix_spawn_file_actions_init(&actions) != 0) {
    return result;
  }
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, output_path.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC, 0600);
  posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);

  const int64_t started = spawn_detail::now_us();
  pid_t pid = 0;
  const int rc = posix_spawn(&pid, argv[0], &actions, nullptr, argv.data(), envp.data());
  posix_spawn_file_actions_destroy(&actions);
  if (rc != 0) {
    result.output = "posix_spawn: " + std::string(std::strerror(rc));
    return result;
  }
  result.spawned = true;

  // waitpid with a deadline: poll rather than install a SIGCHLD handler, which
  // would be process-wide state in a test binary that runs many cases.
  int status = 0;
  const int64_t deadline = started + (static_cast<int64_t>(request.timeout_ms) * 1000);
  for (;;) {
    const pid_t reaped = ::waitpid(pid, &status, WNOHANG);
    if (reaped == pid) {
      break;
    }
    if (reaped < 0) {
      result.spawned = false;
      break;
    }
    if (spawn_detail::now_us() >= deadline) {
      result.timed_out = true;
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &status, 0);
      break;
    }
    const timespec pause{0, 1000000};  // 1 ms
    ::nanosleep(&pause, nullptr);
  }
  result.wall_us = spawn_detail::now_us() - started;

  if (WIFEXITED(status)) {
    result.exited = true;
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.term_signal = WTERMSIG(status);
  }
  result.output = spawn_detail::read_file(output_path);
  if (request.output_path.empty()) {
    ::unlink(output_path.c_str());
  }
  return result;
}

}  // namespace tessera_test

#endif  // TESSERA_TESTS_SUPPORT_SPAWN_H
