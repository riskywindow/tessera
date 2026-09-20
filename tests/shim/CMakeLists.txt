# Tests for libtessera.
#
# Two binaries:
#   shim_smoke_test    the shim's own smoke tests: in-process, and both
#                      interposition strategies with the tenant in a child
#   shim_smoke_tenant  the "application" those strategy tests run
#
# The acceptance matrix (tests/shim/acceptance_*) is written separately and
# adds its own targets here.

# The tenant links against the FAKE's masquerading libcuda.so.1 purely to get
# DT_NEEDED libcuda.so.1 recorded; which library that resolves to at run time
# is the whole point of the test, so it must carry no RPATH back to the build
# tree.
add_executable(shim_smoke_tenant smoke_tenant_main.cc)
target_link_libraries(shim_smoke_tenant PRIVATE tessera::warnings tessera::cuda_fake_masq
  ${CMAKE_DL_LIBS})
set_target_properties(shim_smoke_tenant PROPERTIES SKIP_BUILD_RPATH TRUE)

# The test binary, by contrast, keeps its RPATH: it loads the fake as its own
# libcuda.so.1 so that it can drive the simulation directly (tf_*), while the
# shim it dlopens is pointed at the very same file.
function(_tessera_shim_test name)
  cmake_parse_arguments(ARG "" "" "SOURCES;ENVIRONMENT" ${ARGN})
  tessera_add_gtest(${name}
    SOURCES ${ARG_SOURCES}
    LIBS tessera::cuda_fake_masq tessera::shim_api ${CMAKE_DL_LIBS}
    ENVIRONMENT ${ARG_ENVIRONMENT}
    TIMEOUT 180)
  add_dependencies(${name} shim_smoke_tenant tessera_shim tessera_shim_masq)
  target_include_directories(${name} PRIVATE "${TESSERA_SHIM_GENERATED_DIR}")
  target_compile_definitions(${name} PRIVATE
    "TESSERA_SHIM_PRELOAD_PATH=\"$<TARGET_FILE:tessera_shim>\""
    "TESSERA_SHIM_MASQ_DIR=\"$<TARGET_FILE_DIR:tessera_shim_masq>\""
    "TESSERA_FAKE_MASQ_DIR=\"$<TARGET_FILE_DIR:cuda_fake_masq>\""
    "TESSERA_FAKE_MASQ_LIB=\"$<TARGET_FILE:cuda_fake_masq>\""
    "TESSERA_SHIM_TENANT_PATH=\"$<TARGET_FILE:shim_smoke_tenant>\"")
endfunction()

_tessera_shim_test(shim_smoke_test SOURCES smoke_shim_test.cc smoke_strategies_test.cc)

# --- the acceptance matrix (WP-0.5, T1-T7 of docs/gates/M0.md) --------------
#
# The application every acceptance test runs. Like the smoke tenant it is an
# ordinary CUDA program, and it must carry no RPATH back to the build tree:
# which library its DT_NEEDED libcuda.so.1 resolves to is the thing under test.
#
# ENABLE_EXPORTS (-rdynamic) is load-bearing rather than cosmetic. T3 has to
# fork while the shim holds its one internal lock, and the only window in which
# that lock is held is the shim's dlopen of the real driver; an undefined
# symbol in a shared library binds to the executable's definition first, so the
# application's own dlopen is what the shim calls, and that is where the
# scenario stalls. Without -rdynamic the executable's definition is not in
# .dynsym and the shim would call libc's dlopen instead.
find_package(Threads REQUIRED)
add_executable(shim_acceptance_app acceptance_app_main.cc)
target_link_libraries(shim_acceptance_app PRIVATE tessera::warnings tessera::cuda_fake_masq
  tessera::shim_api Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(shim_acceptance_app PROPERTIES SKIP_BUILD_RPATH TRUE ENABLE_EXPORTS TRUE)

# The acceptance tests themselves link neither the shim nor the fake: every
# number they check comes out of a child process, through the three JSON files
# the application, the shim and the fake driver write.
function(_tessera_acceptance_test name)
  cmake_parse_arguments(ARG "" "" "SOURCES;ENVIRONMENT" ${ARGN})
  tessera_add_gtest(${name}
    SOURCES ${ARG_SOURCES}
    LIBS tessera::test_support tessera::shim_api ${CMAKE_DL_LIBS}
    ENVIRONMENT ${ARG_ENVIRONMENT}
    TIMEOUT 300)
  add_dependencies(${name} shim_acceptance_app tessera_shim tessera_shim_masq cuda_fake_masq
    tessera_null_driver)
  # The generated table of every driver export and whether the shim hooks it:
  # the matrix asserts different things about hooked and unhooked symbols, and
  # which is which must come from the shim's own build, not from a list here.
  target_include_directories(${name} PRIVATE "${TESSERA_SHIM_GENERATED_DIR}")
  target_compile_definitions(${name} PRIVATE
    "TESSERA_SHIM_PRELOAD_PATH=\"$<TARGET_FILE:tessera_shim>\""
    "TESSERA_SHIM_MASQ_DIR=\"$<TARGET_FILE_DIR:tessera_shim_masq>\""
    "TESSERA_SHIM_MASQ_LIB=\"$<TARGET_FILE:tessera_shim_masq>\""
    "TESSERA_FAKE_MASQ_DIR=\"$<TARGET_FILE_DIR:cuda_fake_masq>\""
    "TESSERA_FAKE_MASQ_LIB=\"$<TARGET_FILE:cuda_fake_masq>\""
    "TESSERA_NULL_DRIVER_DIR=\"$<TARGET_FILE_DIR:tessera_null_driver>\""
    "TESSERA_NULL_DRIVER_LIB=\"$<TARGET_FILE:tessera_null_driver>\""
    "TESSERA_ACCEPTANCE_APP=\"$<TARGET_FILE:shim_acceptance_app>\"")
endfunction()

_tessera_acceptance_test(shim_acceptance_test SOURCES acceptance_matrix_test.cc)
_tessera_acceptance_test(shim_acceptance_ptsz_test SOURCES acceptance_ptsz_test.cc)
_tessera_acceptance_test(shim_acceptance_hotpath_test SOURCES acceptance_hotpath_test.cc)

# T3 forks a process whose other thread is inside the shim's initialisation.
# ThreadSanitizer kills a child that starts threads after a multi-threaded
# fork unless die_after_fork is off, and recreating its worker thread in the
# child is exactly what the fake driver does on purpose. The setting reaches
# the application through the environment it inherits from this test.
_tessera_acceptance_test(shim_acceptance_fork_test
  SOURCES acceptance_fork_test.cc
  ENVIRONMENT "TSAN_OPTIONS=die_after_fork=0")

# The fork test is a binary of its own: ThreadSanitizer kills a child that
# starts threads after a multi-threaded fork unless die_after_fork is off, and
# recreating its worker thread in the child is exactly what the fake driver
# does on purpose. The setting is ignored by every other configuration.
_tessera_shim_test(shim_smoke_fork_test
  SOURCES smoke_fork_test.cc
  ENVIRONMENT "TSAN_OPTIONS=die_after_fork=0")
