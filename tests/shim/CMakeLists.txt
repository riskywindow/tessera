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

# The fork test is a binary of its own: ThreadSanitizer kills a child that
# starts threads after a multi-threaded fork unless die_after_fork is off, and
# recreating its worker thread in the child is exactly what the fake driver
# does on purpose. The setting is ignored by every other configuration.
_tessera_shim_test(shim_smoke_fork_test
  SOURCES smoke_fork_test.cc
  ENVIRONMENT "TSAN_OPTIONS=die_after_fork=0")
