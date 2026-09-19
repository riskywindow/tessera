# Cross-compile to arm64 with clang, using the aarch64-linux-gnu gcc
# installation for the sysroot and runtime libraries. Tests run under qemu.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_ASM_COMPILER clang)
set(CMAKE_C_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET aarch64-linux-gnu)
set(CMAKE_ASM_COMPILER_TARGET aarch64-linux-gnu)

# lld avoids depending on the cross binutils default linker selection.
foreach(_t EXE SHARED MODULE)
  set(CMAKE_${_t}_LINKER_FLAGS_INIT "-fuse-ld=lld")
endforeach()

set(TESSERA_AARCH64_SYSROOT "/usr/aarch64-linux-gnu" CACHE PATH "arm64 sysroot")
set(CMAKE_FIND_ROOT_PATH "${TESSERA_AARCH64_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

find_program(TESSERA_QEMU_AARCH64 qemu-aarch64 REQUIRED)
set(CMAKE_CROSSCOMPILING_EMULATOR "${TESSERA_QEMU_AARCH64}" -L "${TESSERA_AARCH64_SYSROOT}")
