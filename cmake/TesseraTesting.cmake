# GoogleTest (test-only dependency), pinned by hash. A pre-downloaded tarball
# in $TESSERA_DEPS is used when present so repeated configures and offline
# builds do not hit the network.

include(FetchContent)
include(GoogleTest)

set(_gtest_ver "1.17.0")
set(_gtest_sha256 "65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c")
set(_deps "$ENV{TESSERA_DEPS}")
if(NOT _deps)
  set(_deps "$ENV{HOME}/.cache/tessera-deps")
endif()
set(_gtest_local "${_deps}/googletest-${_gtest_ver}.tar.gz")
if(EXISTS "${_gtest_local}")
  set(_gtest_url "file://${_gtest_local}")
else()
  set(_gtest_url
    "https://github.com/google/googletest/releases/download/v${_gtest_ver}/googletest-${_gtest_ver}.tar.gz")
endif()

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
  URL "${_gtest_url}"
  URL_HASH "SHA256=${_gtest_sha256}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SYSTEM
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(googletest)

# The emulator used to run target binaries (qemu-aarch64 for cross builds).
# Tests that spawn child processes must launch them through this prefix; it is
# passed to test code as the TESSERA_TEST_EMULATOR compile definition
# (a ';'-free, space-separated command prefix, empty for native builds).
set(TESSERA_TEST_EMULATOR "")
if(CMAKE_CROSSCOMPILING AND CMAKE_CROSSCOMPILING_EMULATOR)
  list(JOIN CMAKE_CROSSCOMPILING_EMULATOR " " TESSERA_TEST_EMULATOR)
endif()

# tessera_add_gtest(<name>
#   SOURCES <src>...
#   [LIBS <lib>...]
#   [LABELS <label>...]         # e.g. gpu (GPU tests only run through Modal)
#   [ENVIRONMENT <K=V>...]
#   [TIMEOUT <seconds>])        # default 120
function(tessera_add_gtest name)
  cmake_parse_arguments(ARG "" "TIMEOUT" "SOURCES;LIBS;LABELS;ENVIRONMENT" ${ARGN})
  if(NOT ARG_TIMEOUT)
    set(ARG_TIMEOUT 120)
  endif()
  add_executable(${name} ${ARG_SOURCES})
  target_link_libraries(${name} PRIVATE GTest::gtest GTest::gmock GTest::gtest_main
    tessera::warnings ${ARG_LIBS})
  target_compile_definitions(${name} PRIVATE
    "TESSERA_TEST_EMULATOR=\"${TESSERA_TEST_EMULATOR}\""
    "TESSERA_SANITIZER_PRELOAD=\"${TESSERA_SANITIZER_PRELOAD}\"")
  # Build the property list conditionally: an empty value would otherwise make
  # CTest swallow the following keyword as that property's value.
  set(_props TIMEOUT ${ARG_TIMEOUT})
  if(ARG_LABELS)
    list(APPEND _props LABELS "${ARG_LABELS}")
  endif()
  if(ARG_ENVIRONMENT)
    list(APPEND _props ENVIRONMENT "${ARG_ENVIRONMENT}")
  endif()
  gtest_discover_tests(${name}
    DISCOVERY_TIMEOUT 60
    PROPERTIES ${_props})
endfunction()
