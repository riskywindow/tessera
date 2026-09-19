# libtessera: the CUDA driver-API shim.
#
# Two shared libraries are built from one set of objects:
#
#   <build>/shim/libtessera.so          SONAME libtessera.so   (LD_PRELOAD)
#   <build>/shim/masq/libcuda.so.1      SONAME libcuda.so.1    (masquerade)
#   <build>/shim/masq/libcuda.so     -> libcuda.so.1
#
# masq/ holds nothing else, because `tessera run` puts exactly that directory
# first on LD_LIBRARY_PATH. Only the dlsym hook differs between the two builds:
# it is compiled into the preload copy alone.

find_package(Threads REQUIRED)

# --- the export list, read from the driver stub for the TARGET architecture --
#
# nm is not optional: without it there is no export list, and a shim that does
# not export everything the driver exports fails at symbol resolution inside
# somebody else's application.
set(_nm_candidates)
if(CMAKE_NM)
  list(APPEND _nm_candidates "${CMAKE_NM}")
endif()
list(APPEND _nm_candidates nm llvm-nm)
find_program(TESSERA_NM NAMES ${_nm_candidates})
if(NOT TESSERA_NM)
  message(FATAL_ERROR
    "tessera: nm not found (looked for: ${_nm_candidates}).\n"
    "The shim generates its export list from ${TESSERA_CUDA_STUB_LIBCUDA} with nm; "
    "install binutils or llvm and re-configure.")
endif()

set(TESSERA_SHIM_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated"
  CACHE INTERNAL "Generated shim symbol tables")
set(_hook_table "${CMAKE_CURRENT_SOURCE_DIR}/src/hook_table.inc")
set(_gen_script "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_symbols.sh")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_hook_table}" "${_gen_script}")

execute_process(
  COMMAND "${_gen_script}" "${TESSERA_NM}" "${TESSERA_CUDA_STUB_LIBCUDA}" "${_hook_table}"
          "${TESSERA_SHIM_GENERATED_DIR}"
  OUTPUT_VARIABLE _gen_out
  ERROR_VARIABLE _gen_err
  RESULT_VARIABLE _gen_rc)
if(NOT _gen_rc EQUAL 0)
  message(FATAL_ERROR "tessera: gen_symbols.sh failed (${_gen_rc}):\n${_gen_err}")
endif()
string(REGEX MATCH "TESSERA_EXPORTED_COUNT=([0-9]+)" _m "${_gen_out}")
set(_exported_count "${CMAKE_MATCH_1}")
string(REGEX MATCH "TESSERA_HOOKED_COUNT=([0-9]+)" _m "${_gen_out}")
set(_hooked_count "${CMAKE_MATCH_1}")
string(REGEX MATCH "TESSERA_HOOKED_NOT_IN_STUB=([^\n]*)" _m "${_gen_out}")
set(_hooked_missing "${CMAKE_MATCH_1}")
string(STRIP "${_hooked_missing}" _hooked_missing)
message(STATUS
  "tessera: shim forwards ${_exported_count} driver symbols (${_hooked_count} hooked) "
  "for ${TESSERA_CUDA_TARGET}")
if(_hooked_missing)
  message(WARNING
    "tessera: these hooked symbols are not exported by ${TESSERA_CUDA_STUB_LIBCUDA}, "
    "so their slots will never resolve on this architecture: ${_hooked_missing}")
endif()

if(CMAKE_LIBRARY_ARCHITECTURE)
  set(_multiarch "${CMAKE_LIBRARY_ARCHITECTURE}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(_multiarch "aarch64-linux-gnu")
else()
  set(_multiarch "x86_64-linux-gnu")
endif()

# --- the objects, compiled once ---------------------------------------------
#
# Exactly one trampoline source is compiled. Choosing it here rather than
# letting both files sort themselves out with #if is deliberate: an
# architecture we have no trampolines for must fail the configure, not produce
# a library that quietly forwards nothing.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(_trampolines src/trampoline_aarch64.S)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(_trampolines src/trampoline_x86_64.S)
else()
  message(FATAL_ERROR
    "tessera: the shim has no forwarding trampolines for CMAKE_SYSTEM_PROCESSOR="
    "'${CMAKE_SYSTEM_PROCESSOR}'. Add shim/src/trampoline_<arch>.S first.")
endif()

add_library(tessera_shim_objs OBJECT
  src/state.cc
  src/symbols.cc
  src/hooks.cc
  src/proc_address.cc
  src/native_dlsym.cc
  ${_trampolines})
target_include_directories(tessera_shim_objs
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${TESSERA_SHIM_GENERATED_DIR}")
target_link_libraries(tessera_shim_objs PUBLIC tessera::cuda_headers Threads::Threads)
target_link_libraries(tessera_shim_objs PRIVATE tessera::warnings)
target_compile_definitions(tessera_shim_objs PRIVATE "TESSERA_MULTIARCH_DIR=\"${_multiarch}\"")
# No exceptions may cross a cu* entry point (CLAUDE.md section 5). The entry
# points cannot be marked noexcept -- cuda.h declares them without it, and in
# C++17 that is part of the type -- so the guarantee is made at the compiler
# level instead, for the whole library.
target_compile_options(tessera_shim_objs PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions;-fno-rtti>")

# The dlsym hook and its bootstrap run before AddressSanitizer has finished
# initialising: the first caller of an interposed dlsym in a sanitized process
# is the sanitizer's own runtime, not the application. Instrumented code cannot
# run there -- it checks a shadow map that is not mapped yet -- and this was
# measured, not assumed: with these two files instrumented, every clang-asan
# run of the preload strategy died (first "CHECK failed: asan_posix.cpp:125
# ((tsd_key_inited)) != (0)", then a segfault inside __asan_init), while the
# masquerade strategy, which has no dlsym hook, passed. They are also the two
# files that must not call anything a sanitizer interposes on, which is why
# they read the loader's structures directly.
if(TESSERA_SANITIZE)
  set_source_files_properties(src/native_dlsym.cc src/dlsym_hook.cc
    PROPERTIES COMPILE_OPTIONS "-fno-sanitize=all")
endif()

# --- the two shared libraries ------------------------------------------------
function(_tessera_shim_lib target version_script)
  target_link_options(${target} PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/${version_script}"
    # Bind our own references to our own definitions. cuGetProcAddress answers
    # by pointer identity, so the addresses it hands out have to be THIS
    # object's wrappers, whatever else in the process defines the same names.
    "-Wl,-Bsymbolic-functions")
  set_property(TARGET ${target} APPEND PROPERTY
    LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${version_script}")
  target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(${target} PUBLIC tessera::cuda_headers Threads::Threads ${CMAKE_DL_LIBS})
endfunction()

# The preload copy. dlsym_hook.cc is the only preload-specific object.
add_library(tessera_shim SHARED
  $<TARGET_OBJECTS:tessera_shim_objs>
  src/dlsym_hook.cc)
add_library(tessera::shim ALIAS tessera_shim)
target_include_directories(tessera_shim
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${TESSERA_SHIM_GENERATED_DIR}")
target_link_libraries(tessera_shim PRIVATE tessera::warnings)
target_compile_options(tessera_shim PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions;-fno-rtti>")
set_target_properties(tessera_shim PROPERTIES
  OUTPUT_NAME tessera
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
_tessera_shim_lib(tessera_shim exports_preload.map)

# The masquerading copy: SONAME libcuda.so.1, in a directory of its own.
add_library(tessera_shim_masq SHARED $<TARGET_OBJECTS:tessera_shim_objs>)
add_library(tessera::shim_masq ALIAS tessera_shim_masq)
set_target_properties(tessera_shim_masq PROPERTIES
  OUTPUT_NAME cuda
  SOVERSION 1
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/masq")
_tessera_shim_lib(tessera_shim_masq exports.map)

# Header-only target for code that reads the shim's counters.
add_library(tessera_shim_api INTERFACE)
add_library(tessera::shim_api ALIAS tessera_shim_api)
target_include_directories(tessera_shim_api INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
