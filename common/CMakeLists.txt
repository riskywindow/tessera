# Shared contracts between shim, daemon, and CLI: the shm page layout, and
# later the wire protocol, config schema, clock, and logging.
add_library(tessera_common INTERFACE)
add_library(tessera::common ALIAS tessera_common)
target_include_directories(tessera_common INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(tessera_common INTERFACE tessera::warnings)
