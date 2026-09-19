# Warning policy: -Wall -Wextra -Werror for all first-party C/C++ code.
# Third-party and NVIDIA headers are included as SYSTEM so they are exempt.
#
# Usage: target_link_libraries(<tgt> PRIVATE tessera::warnings)

add_library(tessera_warnings INTERFACE)
add_library(tessera::warnings ALIAS tessera_warnings)

set(_tessera_warn_common
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
  -Wformat=2
  -Wimplicit-fallthrough
  -Wundef
  -Wcast-qual
  -Wmissing-declarations)

set(_tessera_warn_cxx
  -Wnon-virtual-dtor
  -Woverloaded-virtual
  -Wold-style-cast
  -Wextra-semi)

if(TESSERA_WERROR)
  list(APPEND _tessera_warn_common -Werror)
endif()

target_compile_options(tessera_warnings INTERFACE
  "$<$<COMPILE_LANGUAGE:C,CXX>:${_tessera_warn_common}>"
  "$<$<COMPILE_LANGUAGE:CXX>:${_tessera_warn_cxx}>")
