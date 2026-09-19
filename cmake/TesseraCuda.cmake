# Locates the CUDA driver-API headers (cuda.h, cudaTypedefs.h) and the driver
# stub library (stubs/libcuda.so). Nothing links against the stub: it is used
# only as the authoritative list of symbols the real driver exports, so the
# shim can forward all of them.
#
# Search order:
#   1. -DTESSERA_CUDA_ROOT=<dir containing include/cuda.h>
#   2. $TESSERA_DEPS/cuda-12.6/<target>   (scripts/fetch_cuda_headers.sh)
#   3. $CUDA_HOME, /usr/local/cuda         (the Modal image; CI)
#
# Provides:
#   tessera::cuda_headers        INTERFACE target, SYSTEM include dir
#   TESSERA_CUDA_INCLUDE_DIR     path containing cuda.h
#   TESSERA_CUDA_STUB_LIBCUDA    path to stubs/libcuda.so for the target arch
#   TESSERA_CUDA_HEADER_VERSION  integer CUDA_VERSION from cuda.h (e.g. 12060)
#   TESSERA_CUDA_TARGET          NVIDIA target dir name (x86_64-linux | sbsa-linux)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(TESSERA_CUDA_TARGET "sbsa-linux")
else()
  set(TESSERA_CUDA_TARGET "x86_64-linux")
endif()

set(TESSERA_CUDA_ROOT "" CACHE PATH "Directory containing include/cuda.h and lib/stubs/libcuda.so")

set(_deps "$ENV{TESSERA_DEPS}")
if(NOT _deps)
  set(_deps "$ENV{HOME}/.cache/tessera-deps")
endif()

set(_roots)
if(TESSERA_CUDA_ROOT)
  list(APPEND _roots "${TESSERA_CUDA_ROOT}")
endif()
list(APPEND _roots "${_deps}/cuda-12.6/${TESSERA_CUDA_TARGET}")
if(DEFINED ENV{CUDA_HOME})
  list(APPEND _roots "$ENV{CUDA_HOME}/targets/${TESSERA_CUDA_TARGET}" "$ENV{CUDA_HOME}")
endif()
list(APPEND _roots "/usr/local/cuda/targets/${TESSERA_CUDA_TARGET}" "/usr/local/cuda")

# NO_CMAKE_FIND_ROOT_PATH: the cross toolchains restrict find_* to the target
# sysroot, but these headers live on the build host.
find_path(TESSERA_CUDA_INCLUDE_DIR cuda.h
  PATHS ${_roots}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

find_file(TESSERA_CUDA_STUB_LIBCUDA libcuda.so
  PATHS ${_roots}
  PATH_SUFFIXES lib/stubs lib64/stubs
  NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

if(NOT TESSERA_CUDA_INCLUDE_DIR OR NOT TESSERA_CUDA_STUB_LIBCUDA)
  message(FATAL_ERROR
    "tessera: CUDA driver headers/stub not found (searched: ${_roots}).\n"
    "Run scripts/fetch_cuda_headers.sh (add 'aarch64' for cross builds) "
    "or pass -DTESSERA_CUDA_ROOT=<dir>.")
endif()

file(STRINGS "${TESSERA_CUDA_INCLUDE_DIR}/cuda.h" _ver_line
  REGEX "^#define CUDA_VERSION [0-9]+")
string(REGEX MATCH "[0-9]+" TESSERA_CUDA_HEADER_VERSION "${_ver_line}")
if(NOT TESSERA_CUDA_HEADER_VERSION OR TESSERA_CUDA_HEADER_VERSION LESS 12040)
  message(FATAL_ERROR
    "tessera: cuda.h at ${TESSERA_CUDA_INCLUDE_DIR} is CUDA_VERSION "
    "'${TESSERA_CUDA_HEADER_VERSION}'; 12040 (CUDA 12.4) or newer is required.")
endif()

add_library(tessera_cuda_headers INTERFACE)
add_library(tessera::cuda_headers ALIAS tessera_cuda_headers)
target_include_directories(tessera_cuda_headers SYSTEM INTERFACE "${TESSERA_CUDA_INCLUDE_DIR}")

message(STATUS "tessera: cuda.h ${TESSERA_CUDA_HEADER_VERSION} at ${TESSERA_CUDA_INCLUDE_DIR}")
message(STATUS "tessera: driver export list from ${TESSERA_CUDA_STUB_LIBCUDA}")
