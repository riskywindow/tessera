// Reading the three counter sources the acceptance matrix compares.
//
// T1 asks for one thing: for every hooked symbol the application exercised,
// the calls the APPLICATION issued, the SHIM's per-symbol counter and the
// FAKE DRIVER's per-symbol counter are the same number. Those three numbers
// arrive as three small JSON files, because the application runs in a child
// process and each side writes its own at exit:
//
//   the app   $TESSERA_APP_REPORT       written by tests/shim/acceptance_app_main.cc
//   the shim  $TESSERA_STATS_FILE       written by libtessera   (shim/README.md)
//   the fake  $TESSERA_FAKE_STATS_FILE  written by libcuda_fake (fake_driver/README.md)
//
// All three are flat objects of "name": <unsigned> pairs, possibly nested one
// level under a named member. A full JSON parser would be more machinery than
// that shape deserves, and a wrong one would be a second thing to debug, so
// this reads exactly that shape and says so when it sees anything else.

#ifndef TESSERA_TESTS_SUPPORT_COUNTERS_H
#define TESSERA_TESTS_SUPPORT_COUNTERS_H

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace tessera_test {

using CounterMap = std::map<std::string, uint64_t>;

// open/read rather than stdio: these files are a few hundred bytes, and a
// descriptor has one failure mode instead of three.
inline std::string read_text_file(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return {};
  }
  std::string text;
  char buffer[8192];
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

// The unsigned number after "key": at any nesting depth, or nothing.
inline std::optional<uint64_t> json_number(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\":";
  const size_t at = json.find(needle);
  if (at == std::string::npos) {
    return std::nullopt;
  }
  size_t i = at + needle.size();
  if (i >= json.size() || json[i] < '0' || json[i] > '9') {
    return std::nullopt;
  }
  uint64_t value = 0;
  for (; i < json.size() && json[i] >= '0' && json[i] <= '9'; ++i) {
    value = (value * 10) + static_cast<uint64_t>(json[i] - '0');
  }
  return value;
}

// The string value of "key", or nothing.
inline std::optional<std::string> json_string(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\":\"";
  const size_t at = json.find(needle);
  if (at == std::string::npos) {
    return std::nullopt;
  }
  const size_t start = at + needle.size();
  const size_t end = json.find('"', start);
  if (end == std::string::npos) {
    return std::nullopt;
  }
  return json.substr(start, end - start);
}

// Every "name": <unsigned> pair of one object. `member` names the object to
// read ("symbols" in the shim's file, "issued" in the app's); an empty
// `member` reads the whole file, which is the fake's shape. Members whose
// value is not a number end the object, so a trailing string field is not a
// parse error.
inline CounterMap json_counts(const std::string& json, const std::string& member) {
  CounterMap counts;
  size_t i = 0;
  if (!member.empty()) {
    const std::string needle = "\"" + member + "\":{";
    const size_t at = json.find(needle);
    if (at == std::string::npos) {
      return counts;
    }
    i = at + needle.size();
  }
  while (i < json.size()) {
    const size_t key_start = json.find('"', i);
    if (key_start == std::string::npos) {
      break;
    }
    const size_t key_end = json.find('"', key_start + 1);
    if (key_end == std::string::npos) {
      break;
    }
    const std::string key = json.substr(key_start + 1, key_end - key_start - 1);
    size_t value_start = key_end + 1;
    if (value_start >= json.size() || json[value_start] != ':') {
      break;
    }
    ++value_start;
    uint64_t value = 0;
    size_t j = value_start;
    for (; j < json.size() && json[j] >= '0' && json[j] <= '9'; ++j) {
      value = (value * 10) + static_cast<uint64_t>(json[j] - '0');
    }
    if (j == value_start) {
      break;  // not a number: this object has ended
    }
    counts[key] = value;
    i = j;
  }
  return counts;
}

// Only the cu* entry points: the app's report and the shim's file also carry
// bookkeeping of their own, and only driver symbols are comparable.
inline CounterMap driver_counts(const std::string& json, const std::string& member) {
  CounterMap counts;
  for (const auto& [name, value] : json_counts(json, member)) {
    if (name.compare(0, 2, "cu") == 0) {
      counts[name] = value;
    }
  }
  return counts;
}

inline uint64_t count_of(const CounterMap& counts, const std::string& name) {
  const auto found = counts.find(name);
  return found == counts.end() ? 0 : found->second;
}

}  // namespace tessera_test

#endif  // TESSERA_TESTS_SUPPORT_COUNTERS_H
