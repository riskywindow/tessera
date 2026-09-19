// <cuda.h>, configured the way the driver's own sources see it.
//
// Every translation unit in the shim that needs the driver API includes this
// header instead of <cuda.h> directly, so all of them see the same
// declarations and the same enumerations.
//
//   __CUDA_API_VERSION_INTERNAL        declares every variant under its real
//                                      exported name (cuMemAlloc AND
//                                      cuMemAlloc_v2, cuLaunchKernel AND
//                                      cuLaunchKernel_ptsz) and turns off the
//                                      renaming macros that would otherwise
//                                      rewrite our definitions.
//   __CUDA_API_VERSION_INTERNAL_ODR    keeps the enumerations identical to the
//                                      ones a normal translation unit sees.
//   __CUDA_API_PUSH_VISIBILITY_DEFAULT wraps the declarations in
//                                      `#pragma GCC visibility push(default)`,
//                                      so the entry points are exported even
//                                      though the build compiles with
//                                      -fvisibility=hidden.

#ifndef TESSERA_SHIM_SRC_CUDA_API_H
#define TESSERA_SHIM_SRC_CUDA_API_H

// clang-format off
// These are cuda.h's own configuration macros, spelled the way NVIDIA spells
// them; they have to be defined before the header is read.
// NOLINTBEGIN(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
#define __CUDA_API_VERSION_INTERNAL 1
#define __CUDA_API_VERSION_INTERNAL_ODR 1
#define __CUDA_API_PUSH_VISIBILITY_DEFAULT 1
// NOLINTEND(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
#include <cuda.h>
// clang-format on

#endif  // TESSERA_SHIM_SRC_CUDA_API_H
