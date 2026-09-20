add_subdirectory(common)

# Scaffolding the suites share (spawn.h, counters.h, the null driver). It has
# to be configured before the suites that link it.
add_subdirectory(support)

foreach(dir IN ITEMS fake_driver shim)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
    add_subdirectory(${dir})
  endif()
endforeach()
