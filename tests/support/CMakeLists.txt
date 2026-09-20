# Shared scaffolding for the test suites.
#
#   tessera::test_support   header-only: spawn.h (launching a child process the
#                           same way on every configuration) and counters.h
#                           (reading the three counter files the acceptance
#                           matrix compares). Headers are included as
#                           "support/spawn.h", so the include directory is
#                           tests/ itself.
#
#   tessera_null_driver     a CUDA driver that does nothing, used by the I-4
#                           measurement (T7). See null_driver.c for why the
#                           measurement cannot use libcuda_fake.

add_library(tessera_test_support INTERFACE)
add_library(tessera::test_support ALIAS tessera_test_support)
target_include_directories(tessera_test_support INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/..")

# SONAME libcuda.so.1, alone in a directory of its own, so that one directory
# on LD_LIBRARY_PATH makes this the application's driver and nothing else
# comes with it -- the same shape as fake_driver/fake_masq and shim/masq.
add_library(tessera_null_driver SHARED null_driver.c)
add_library(tessera::null_driver ALIAS tessera_null_driver)
target_link_libraries(tessera_null_driver PUBLIC tessera::cuda_headers)
target_link_libraries(tessera_null_driver PRIVATE tessera::warnings)
set_target_properties(tessera_null_driver PROPERTIES
  OUTPUT_NAME cuda
  SOVERSION 1
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/null_masq")
