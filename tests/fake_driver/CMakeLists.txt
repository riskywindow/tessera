# Tests for libcuda_fake itself.
#
# Three binaries:
#   fake_driver_test        the bulk of the suite, in one process per test case
#   fake_driver_fork_test   the fork tests, which need TSAN_OPTIONS of their own
#   fake_driver_stats_child the "application" the stats round-trip spawns

set(_masq_dir "$<TARGET_FILE_DIR:cuda_fake_masq>")

add_executable(fake_driver_stats_child stats_child_main.cc)
target_link_libraries(fake_driver_stats_child PRIVATE tessera::warnings tessera::cuda_fake)

tessera_add_gtest(fake_driver_test
  SOURCES
    stream_test.cc
    time_test.cc
    event_test.cc
    memory_test.cc
    graph_test.cc
    proc_address_test.cc
    ptsz_test.cc
    control_test.cc
    exports_test.cc
  LIBS tessera::cuda_fake ${CMAKE_DL_LIBS}
  TIMEOUT 180)
add_dependencies(fake_driver_test fake_driver_stats_child)
target_compile_definitions(fake_driver_test PRIVATE
  "TESSERA_STATS_CHILD_PATH=\"$<TARGET_FILE:fake_driver_stats_child>\""
  "TESSERA_FAKE_MASQ_DIR=\"${_masq_dir}\"")

# ThreadSanitizer kills a child that starts threads after a multi-threaded
# fork unless die_after_fork is off. Recreating the worker thread in the child
# is exactly what the fake must do, so these tests opt out. The setting is
# ignored by every other configuration.
tessera_add_gtest(fake_driver_fork_test
  SOURCES fork_test.cc
  LIBS tessera::cuda_fake ${CMAKE_DL_LIBS}
  ENVIRONMENT "TSAN_OPTIONS=die_after_fork=0"
  TIMEOUT 180)
