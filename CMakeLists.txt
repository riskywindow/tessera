cmake_minimum_required(VERSION 3.28)

project(tessera
  VERSION 0.0.1
  DESCRIPTION "Userspace GPU hypervisor with latency SLOs"
  LANGUAGES C CXX ASM)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not supported; use a preset (cmake --preset gcc-debug).")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Hidden by default; only cu* driver symbols and tessera_* are exported (exports.map).
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(TESSERA_BUILD_TESTS "Build CPU tests" ON)
option(TESSERA_WERROR "Treat warnings as errors" ON)
set(TESSERA_SANITIZE "" CACHE STRING
  "Sanitizers to enable: empty, 'address;undefined', or 'thread'")

include(TesseraWarnings)
include(TesseraSanitizers)
include(TesseraCuda)

if(TESSERA_BUILD_TESTS)
  enable_testing()
  include(TesseraTesting)
endif()

# Components. Each directory owns its own CMakeLists.txt; directories that do
# not exist yet (later milestones) are skipped.
foreach(dir IN ITEMS common fake_driver shim daemon cli bench tests)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
    add_subdirectory(${dir})
  endif()
endforeach()
