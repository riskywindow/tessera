/* The hooked entry points, with their signatures: the one source of truth for
 * everything the shim knows about a symbol it intercepts.
 *
 * TESSERA_WRAP(exported, base, kind, (params), (args))
 *
 *   exported   the exported symbol name, e.g. cuMemAlloc_v2. It is also the
 *              name of the wrapper this file defines, and the name
 *              gen_symbols.sh removes from the trampoline list.
 *   base       the name an application passes to cuGetProcAddress, e.g.
 *              cuMemAlloc. Several exported symbols share one base; which one
 *              the caller gets is the driver's choice (cudaVersion and the
 *              CU_GET_PROC_ADDRESS_* flags), and the shim discovers that
 *              choice by pointer identity rather than by guessing (ADR-001).
 *   kind       kPlain, kLaunch or kGraphLaunch: which aggregate counter this
 *              call bumps on top of its own per-symbol counter.
 *   params     the parameter list, exactly as cuda.h declares it with
 *              __CUDA_API_VERSION_INTERNAL defined. The compiler checks it
 *              against that declaration, so a wrong signature is a build
 *              error rather than a silent ABI mismatch.
 *   args       the same parameters, forwarded by name.
 *
 * TESSERA_HOOK_MANUAL(exported, base) names an entry point whose wrapper is
 * written by hand (cuGetProcAddress and cuGetProcAddress_v2 do more than
 * forward) but which still needs an index, a counter and a row in the
 * variant table.
 *
 * This file is #included several times with the macros defined differently;
 * it has no include guard on purpose.
 *
 * Every name here is exported by the CUDA 12.6.77 driver stub on BOTH
 * x86-64 and sbsa (checked with nm against lib/stubs/libcuda.so).
 */

/* ---- launches --------------------------------------------------------------
 * Launch entry points. `launches` counts every one of these; a graph
 * launch is counted separately in `graph_launches` because one graph launch
 * is one API call but many kernels (M0 gate G9).
 */
TESSERA_WRAP(cuLaunchKernel, cuLaunchKernel, kLaunch,
             (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra),
             (f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra))
TESSERA_WRAP(cuLaunchKernel_ptsz, cuLaunchKernel, kLaunch,
             (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra),
             (f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra))
TESSERA_WRAP(cuLaunchKernelEx, cuLaunchKernelEx, kLaunch,
             (const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra),
             (config, f, kernelParams, extra))
TESSERA_WRAP(cuLaunchKernelEx_ptsz, cuLaunchKernelEx, kLaunch,
             (const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra),
             (config, f, kernelParams, extra))
TESSERA_WRAP(cuLaunchCooperativeKernel, cuLaunchCooperativeKernel, kLaunch,
             (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams),
             (f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams))
TESSERA_WRAP(cuLaunchCooperativeKernel_ptsz, cuLaunchCooperativeKernel, kLaunch,
             (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams),
             (f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams))
TESSERA_WRAP(cuLaunch, cuLaunch, kLaunch,
             (CUfunction f),
             (f))
TESSERA_WRAP(cuLaunchGrid, cuLaunchGrid, kLaunch,
             (CUfunction f, int grid_width, int grid_height),
             (f, grid_width, grid_height))
TESSERA_WRAP(cuLaunchGridAsync, cuLaunchGridAsync, kLaunch,
             (CUfunction f, int grid_width, int grid_height, CUstream hStream),
             (f, grid_width, grid_height, hStream))

/* ---- graph launches --------------------------------------------------------
 */
TESSERA_WRAP(cuGraphLaunch, cuGraphLaunch, kGraphLaunch,
             (CUgraphExec hGraph, CUstream hStream),
             (hGraph, hStream))
TESSERA_WRAP(cuGraphLaunch_ptsz, cuGraphLaunch, kGraphLaunch,
             (CUgraphExec hGraphExec, CUstream hStream),
             (hGraphExec, hStream))

/* ---- memory ----------------------------------------------------------------
 * The v1 forms are hooked next to the _v2 forms on purpose: a quota
 * that only covers _v2 is a quota with a hole (M2 depends on this).
 */
TESSERA_WRAP(cuMemAlloc, cuMemAlloc, kPlain,
             (CUdeviceptr_v1* dptr, unsigned int bytesize),
             (dptr, bytesize))
TESSERA_WRAP(cuMemAlloc_v2, cuMemAlloc, kPlain,
             (CUdeviceptr* dptr, size_t bytesize),
             (dptr, bytesize))
TESSERA_WRAP(cuMemAllocPitch, cuMemAllocPitch, kPlain,
             (CUdeviceptr_v1* dptr, unsigned int* pPitch, unsigned int WidthInBytes, unsigned int Height, unsigned int ElementSizeBytes),
             (dptr, pPitch, WidthInBytes, Height, ElementSizeBytes))
TESSERA_WRAP(cuMemAllocPitch_v2, cuMemAllocPitch, kPlain,
             (CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes),
             (dptr, pPitch, WidthInBytes, Height, ElementSizeBytes))
TESSERA_WRAP(cuMemAllocManaged, cuMemAllocManaged, kPlain,
             (CUdeviceptr* dptr, size_t bytesize, unsigned int flags),
             (dptr, bytesize, flags))
TESSERA_WRAP(cuMemAllocAsync, cuMemAllocAsync, kPlain,
             (CUdeviceptr* dptr, size_t bytesize, CUstream hStream),
             (dptr, bytesize, hStream))
TESSERA_WRAP(cuMemAllocAsync_ptsz, cuMemAllocAsync, kPlain,
             (CUdeviceptr* dptr, size_t bytesize, CUstream hStream),
             (dptr, bytesize, hStream))
TESSERA_WRAP(cuMemAllocFromPoolAsync, cuMemAllocFromPoolAsync, kPlain,
             (CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream),
             (dptr, bytesize, pool, hStream))
TESSERA_WRAP(cuMemAllocFromPoolAsync_ptsz, cuMemAllocFromPoolAsync, kPlain,
             (CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream),
             (dptr, bytesize, pool, hStream))
TESSERA_WRAP(cuMemFree, cuMemFree, kPlain,
             (CUdeviceptr_v1 dptr),
             (dptr))
TESSERA_WRAP(cuMemFree_v2, cuMemFree, kPlain,
             (CUdeviceptr dptr),
             (dptr))
TESSERA_WRAP(cuMemFreeAsync, cuMemFreeAsync, kPlain,
             (CUdeviceptr dptr, CUstream hStream),
             (dptr, hStream))
TESSERA_WRAP(cuMemFreeAsync_ptsz, cuMemFreeAsync, kPlain,
             (CUdeviceptr dptr, CUstream hStream),
             (dptr, hStream))
TESSERA_WRAP(cuMemCreate, cuMemCreate, kPlain,
             (CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags),
             (handle, size, prop, flags))
TESSERA_WRAP(cuMemRelease, cuMemRelease, kPlain,
             (CUmemGenericAllocationHandle handle),
             (handle))
TESSERA_WRAP(cuMemMap, cuMemMap, kPlain,
             (CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags),
             (ptr, size, offset, handle, flags))
TESSERA_WRAP(cuMemUnmap, cuMemUnmap, kPlain,
             (CUdeviceptr ptr, size_t size),
             (ptr, size))
TESSERA_WRAP(cuMemAddressReserve, cuMemAddressReserve, kPlain,
             (CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags),
             (ptr, size, alignment, addr, flags))
TESSERA_WRAP(cuMemSetAccess, cuMemSetAccess, kPlain,
             (CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count),
             (ptr, size, desc, count))
TESSERA_WRAP(cuMemGetInfo, cuMemGetInfo, kPlain,
             (unsigned int* free, unsigned int* total),
             (free, total))
TESSERA_WRAP(cuMemGetInfo_v2, cuMemGetInfo, kPlain,
             (size_t* free, size_t* total),
             (free, total))
TESSERA_WRAP(cuDeviceTotalMem, cuDeviceTotalMem, kPlain,
             (unsigned int* bytes, CUdevice dev),
             (bytes, dev))
TESSERA_WRAP(cuDeviceTotalMem_v2, cuDeviceTotalMem, kPlain,
             (size_t* bytes, CUdevice dev),
             (bytes, dev))

/* ---- streams and contexts --------------------------------------------------
 */
TESSERA_WRAP(cuStreamCreate, cuStreamCreate, kPlain,
             (CUstream* phStream, unsigned int Flags),
             (phStream, Flags))
TESSERA_WRAP(cuStreamCreateWithPriority, cuStreamCreateWithPriority, kPlain,
             (CUstream* phStream, unsigned int flags, int priority),
             (phStream, flags, priority))
TESSERA_WRAP(cuStreamDestroy, cuStreamDestroy, kPlain,
             (CUstream hStream),
             (hStream))
TESSERA_WRAP(cuStreamDestroy_v2, cuStreamDestroy, kPlain,
             (CUstream hStream),
             (hStream))
TESSERA_WRAP(cuStreamBeginCapture, cuStreamBeginCapture, kPlain,
             (CUstream hStream),
             (hStream))
TESSERA_WRAP(cuStreamBeginCapture_ptsz, cuStreamBeginCapture, kPlain,
             (CUstream hStream),
             (hStream))
TESSERA_WRAP(cuStreamBeginCapture_v2, cuStreamBeginCapture, kPlain,
             (CUstream hStream, CUstreamCaptureMode mode),
             (hStream, mode))
TESSERA_WRAP(cuStreamBeginCapture_v2_ptsz, cuStreamBeginCapture, kPlain,
             (CUstream hStream, CUstreamCaptureMode mode),
             (hStream, mode))
TESSERA_WRAP(cuStreamEndCapture, cuStreamEndCapture, kPlain,
             (CUstream hStream, CUgraph* phGraph),
             (hStream, phGraph))
TESSERA_WRAP(cuStreamEndCapture_ptsz, cuStreamEndCapture, kPlain,
             (CUstream hStream, CUgraph* phGraph),
             (hStream, phGraph))
TESSERA_WRAP(cuStreamIsCapturing, cuStreamIsCapturing, kPlain,
             (CUstream hStream, CUstreamCaptureStatus* captureStatus),
             (hStream, captureStatus))
TESSERA_WRAP(cuStreamIsCapturing_ptsz, cuStreamIsCapturing, kPlain,
             (CUstream hStream, CUstreamCaptureStatus* captureStatus),
             (hStream, captureStatus))
TESSERA_WRAP(cuCtxCreate, cuCtxCreate, kPlain,
             (CUcontext* pctx, unsigned int flags, CUdevice dev),
             (pctx, flags, dev))
TESSERA_WRAP(cuCtxCreate_v2, cuCtxCreate, kPlain,
             (CUcontext* pctx, unsigned int flags, CUdevice dev),
             (pctx, flags, dev))
TESSERA_WRAP(cuCtxCreate_v3, cuCtxCreate, kPlain,
             (CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev),
             (pctx, paramsArray, numParams, flags, dev))
TESSERA_WRAP(cuCtxCreate_v4, cuCtxCreate, kPlain,
             (CUcontext* pctx, CUctxCreateParams* ctxCreateParams, unsigned int flags, CUdevice dev),
             (pctx, ctxCreateParams, flags, dev))
TESSERA_WRAP(cuDevicePrimaryCtxRetain, cuDevicePrimaryCtxRetain, kPlain,
             (CUcontext* pctx, CUdevice dev),
             (pctx, dev))

/* ---- events ----------------------------------------------------------------
 */
TESSERA_WRAP(cuEventRecord, cuEventRecord, kPlain,
             (CUevent hEvent, CUstream hStream),
             (hEvent, hStream))
TESSERA_WRAP(cuEventRecord_ptsz, cuEventRecord, kPlain,
             (CUevent hEvent, CUstream hStream),
             (hEvent, hStream))
TESSERA_WRAP(cuEventRecordWithFlags, cuEventRecordWithFlags, kPlain,
             (CUevent hEvent, CUstream hStream, unsigned int flags),
             (hEvent, hStream, flags))
TESSERA_WRAP(cuEventRecordWithFlags_ptsz, cuEventRecordWithFlags, kPlain,
             (CUevent hEvent, CUstream hStream, unsigned int flags),
             (hEvent, hStream, flags))

/* ---- asynchronous copies (accounting only in v0) ---------------------------
 */
TESSERA_WRAP(cuMemcpyAsync, cuMemcpyAsync, kPlain,
             (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream),
             (dst, src, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyAsync_ptsz, cuMemcpyAsync, kPlain,
             (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream),
             (dst, src, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyHtoDAsync, cuMemcpyHtoDAsync, kPlain,
             (CUdeviceptr_v1 dstDevice, const void* srcHost, unsigned int ByteCount, CUstream hStream),
             (dstDevice, srcHost, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyHtoDAsync_v2, cuMemcpyHtoDAsync, kPlain,
             (CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream),
             (dstDevice, srcHost, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyHtoDAsync_v2_ptsz, cuMemcpyHtoDAsync, kPlain,
             (CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream),
             (dstDevice, srcHost, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyDtoHAsync, cuMemcpyDtoHAsync, kPlain,
             (void* dstHost, CUdeviceptr_v1 srcDevice, unsigned int ByteCount, CUstream hStream),
             (dstHost, srcDevice, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyDtoHAsync_v2, cuMemcpyDtoHAsync, kPlain,
             (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream),
             (dstHost, srcDevice, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyDtoHAsync_v2_ptsz, cuMemcpyDtoHAsync, kPlain,
             (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream),
             (dstHost, srcDevice, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyDtoDAsync, cuMemcpyDtoDAsync, kPlain,
             (CUdeviceptr_v1 dstDevice, CUdeviceptr_v1 srcDevice, unsigned int ByteCount, CUstream hStream),
             (dstDevice, srcDevice, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyDtoDAsync_v2, cuMemcpyDtoDAsync, kPlain,
             (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream),
             (dstDevice, srcDevice, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyDtoDAsync_v2_ptsz, cuMemcpyDtoDAsync, kPlain,
             (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream),
             (dstDevice, srcDevice, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyPeerAsync, cuMemcpyPeerAsync, kPlain,
             (CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream),
             (dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyPeerAsync_ptsz, cuMemcpyPeerAsync, kPlain,
             (CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream),
             (dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream))
TESSERA_WRAP(cuMemcpy2DAsync, cuMemcpy2DAsync, kPlain,
             (const CUDA_MEMCPY2D_v1* pCopy, CUstream hStream),
             (pCopy, hStream))
TESSERA_WRAP(cuMemcpy2DAsync_v2, cuMemcpy2DAsync, kPlain,
             (const CUDA_MEMCPY2D* pCopy, CUstream hStream),
             (pCopy, hStream))
TESSERA_WRAP(cuMemcpy2DAsync_v2_ptsz, cuMemcpy2DAsync, kPlain,
             (const CUDA_MEMCPY2D* pCopy, CUstream hStream),
             (pCopy, hStream))
TESSERA_WRAP(cuMemcpy3DAsync, cuMemcpy3DAsync, kPlain,
             (const CUDA_MEMCPY3D_v1* pCopy, CUstream hStream),
             (pCopy, hStream))
TESSERA_WRAP(cuMemcpy3DAsync_v2, cuMemcpy3DAsync, kPlain,
             (const CUDA_MEMCPY3D* pCopy, CUstream hStream),
             (pCopy, hStream))
TESSERA_WRAP(cuMemcpy3DAsync_v2_ptsz, cuMemcpy3DAsync, kPlain,
             (const CUDA_MEMCPY3D* pCopy, CUstream hStream),
             (pCopy, hStream))
TESSERA_WRAP(cuMemcpy3DPeerAsync, cuMemcpy3DPeerAsync, kPlain,
             (const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream),
             (pCopy, hStream))
TESSERA_WRAP(cuMemcpy3DPeerAsync_ptsz, cuMemcpy3DPeerAsync, kPlain,
             (const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream),
             (pCopy, hStream))
TESSERA_WRAP(cuMemcpyHtoAAsync, cuMemcpyHtoAAsync, kPlain,
             (CUarray dstArray, unsigned int dstOffset, const void* srcHost, unsigned int ByteCount, CUstream hStream),
             (dstArray, dstOffset, srcHost, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyHtoAAsync_v2, cuMemcpyHtoAAsync, kPlain,
             (CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream),
             (dstArray, dstOffset, srcHost, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyHtoAAsync_v2_ptsz, cuMemcpyHtoAAsync, kPlain,
             (CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream),
             (dstArray, dstOffset, srcHost, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyAtoHAsync, cuMemcpyAtoHAsync, kPlain,
             (void* dstHost, CUarray srcArray, unsigned int srcOffset, unsigned int ByteCount, CUstream hStream),
             (dstHost, srcArray, srcOffset, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyAtoHAsync_v2, cuMemcpyAtoHAsync, kPlain,
             (void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream),
             (dstHost, srcArray, srcOffset, ByteCount, hStream))
TESSERA_WRAP(cuMemcpyAtoHAsync_v2_ptsz, cuMemcpyAtoHAsync, kPlain,
             (void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream),
             (dstHost, srcArray, srcOffset, ByteCount, hStream))
/* ---- the lookup itself ------------------------------------------------------
 * cudart resolves every entry point through cuGetProcAddress, including
 * cuGetProcAddress itself, so this is the hook the rest depend on. Its
 * wrappers live in proc_address.cc.
 */
TESSERA_HOOK_MANUAL(cuGetProcAddress, cuGetProcAddress)
TESSERA_HOOK_MANUAL(cuGetProcAddress_v2, cuGetProcAddress)
