add_subdirectory(common)

foreach(dir IN ITEMS fake_driver shim)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
    add_subdirectory(${dir})
  endif()
endforeach()
