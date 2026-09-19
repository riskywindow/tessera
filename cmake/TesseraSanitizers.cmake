# Sanitizer configuration, driven by TESSERA_SANITIZE (a CMake list):
#   ""                  no sanitizers
#   "address;undefined" ASan + UBSan (the G4 gate configuration)
#   "thread"            TSan
#
# Applied globally so every target (shim, fake driver, test apps) is
# instrumented consistently. Also exposes TESSERA_SANITIZER_PRELOAD: the
# sanitizer runtime that must precede any LD_PRELOAD'd library when the
# runtime is a shared object (gcc's default). Tests that spawn processes with
# LD_PRELOAD must prepend it.

set(TESSERA_SANITIZER_PRELOAD "")

if(TESSERA_SANITIZE)
  list(JOIN TESSERA_SANITIZE "," _tessera_san)
  if("thread" IN_LIST TESSERA_SANITIZE AND "address" IN_LIST TESSERA_SANITIZE)
    message(FATAL_ERROR "TESSERA_SANITIZE: 'thread' cannot be combined with 'address'")
  endif()

  add_compile_options(
    "-fsanitize=${_tessera_san}"
    -fno-omit-frame-pointer
    -fno-sanitize-recover=all
    -g)
  add_link_options("-fsanitize=${_tessera_san}")
  add_compile_definitions(TESSERA_SANITIZED=1)

  # Locate the shared sanitizer runtime (gcc links it dynamically by default;
  # clang links a static runtime into executables, so nothing is needed there).
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if("address" IN_LIST TESSERA_SANITIZE)
      set(_rt libasan.so)
    elseif("thread" IN_LIST TESSERA_SANITIZE)
      set(_rt libtsan.so)
    endif()
    if(_rt)
      execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=${_rt}
        OUTPUT_VARIABLE _rt_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
      get_filename_component(TESSERA_SANITIZER_PRELOAD "${_rt_path}" REALPATH)
    endif()
  endif()
  message(STATUS "tessera: sanitizers=${_tessera_san} preload='${TESSERA_SANITIZER_PRELOAD}'")
endif()
