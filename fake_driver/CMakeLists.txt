# libcuda_fake: a CPU-only simulation of the CUDA driver.
#
# Two shared libraries are built from one set of objects:
#
#   <build>/fake_driver/libcuda_fake.so           SONAME libcuda_fake.so
#   <build>/fake_driver/fake_masq/libcuda.so.1    SONAME libcuda.so.1
#   <build>/fake_driver/fake_masq/libcuda.so   -> libcuda.so.1
#
# fake_masq holds nothing else, so a test can put that one directory on
# LD_LIBRARY_PATH and have dlopen("libcuda.so.1") find the fake and nothing
# else. The two copies are separate .so files with separate state: a process
# should load one or the other, never both.

find_package(Threads REQUIRED)

# --- linker version script, generated from the entry-point table ------------
set(_tf_table "${CMAKE_CURRENT_SOURCE_DIR}/src/table.inc")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_tf_table}")
file(STRINGS "${_tf_table}" _tf_rows REGEX "^TF_ROW\\(")
set(TF_EXPORT_LIST "")
foreach(_row IN LISTS _tf_rows)
  string(REGEX REPLACE "^TF_ROW\\(([A-Za-z0-9_]+),.*$" "\\1" _name "${_row}")
  string(APPEND TF_EXPORT_LIST "    ${_name};\n")
endforeach()
list(LENGTH _tf_rows _tf_row_count)
if(_tf_row_count LESS 100)
  message(FATAL_ERROR "fake_driver: only ${_tf_row_count} rows parsed from src/table.inc")
endif()
set(_tf_map "${CMAKE_CURRENT_BINARY_DIR}/cuda_fake_exports.map")
configure_file(exports.map.in "${_tf_map}" @ONLY)
message(STATUS "tessera: libcuda_fake exports ${_tf_row_count} cu* entry points")

# --- the objects, compiled once ---------------------------------------------
add_library(cuda_fake_objs OBJECT
  src/sim.cc
  src/entry_points.cc
  src/control.cc)
target_include_directories(cuda_fake_objs PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(cuda_fake_objs PUBLIC tessera::cuda_headers Threads::Threads)
target_link_libraries(cuda_fake_objs PRIVATE tessera::warnings)

# --- the two shared libraries -----------------------------------------------
function(_tessera_fake_lib target)
  add_library(${target} SHARED $<TARGET_OBJECTS:cuda_fake_objs>)
  target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(${target} PUBLIC tessera::cuda_headers Threads::Threads)
  target_link_options(${target} PRIVATE "-Wl,--version-script=${_tf_map}")
  set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${_tf_map}")
endfunction()

_tessera_fake_lib(cuda_fake)
add_library(tessera::cuda_fake ALIAS cuda_fake)
set_target_properties(cuda_fake PROPERTIES
  OUTPUT_NAME cuda_fake
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

_tessera_fake_lib(cuda_fake_masq)
add_library(tessera::cuda_fake_masq ALIAS cuda_fake_masq)
set_target_properties(cuda_fake_masq PROPERTIES
  OUTPUT_NAME cuda
  SOVERSION 1
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fake_masq")
