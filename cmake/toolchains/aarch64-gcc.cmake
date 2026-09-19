# Cross-compile to arm64 with the Ubuntu aarch64-linux-gnu gcc, and run tests
# under qemu user-mode emulation. The dev host is x86-64, so this is how the
# I-6 arm64 requirement is met locally; CI additionally uses native arm64
# runners.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)

set(TESSERA_AARCH64_SYSROOT "/usr/aarch64-linux-gnu" CACHE PATH "arm64 sysroot")
set(CMAKE_FIND_ROOT_PATH "${TESSERA_AARCH64_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

find_program(TESSERA_QEMU_AARCH64 qemu-aarch64 REQUIRED)
set(CMAKE_CROSSCOMPILING_EMULATOR "${TESSERA_QEMU_AARCH64}" -L "${TESSERA_AARCH64_SYSROOT}")
