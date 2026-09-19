/* Direct access to libcuda_fake's versioned and per-thread-default-stream
 * entry points.
 *
 * cuda.h reaches most driver entry points through macros: `cuMemAlloc` is a
 * macro for `cuMemAlloc_v2`, `cuLaunchKernel` is a macro for
 * `cuLaunchKernel_ptsz` when the translation unit is compiled with
 * CUDA_API_PER_THREAD_DEFAULT_STREAM, and so on. That is fine for an
 * application, which wants one entry point per operation, but a test of the
 * fake driver has to call a NAMED variant and prove which one ran.
 *
 * This header declares those variants under their real exported names. It
 * defines no macros, so it is safe to include after <cuda.h> and it does not
 * change the meaning of any cuda.h name.
 *
 * The version-1 entry points cannot be declared this way, because their names
 * (cuMemAlloc, cuCtxCreate, ...) are cuda.h macros for the versioned ones.
 * Reach those through tf_symbol_address("cuMemAlloc") and one of the TF_PFN_*
 * typedefs below, or through cuGetProcAddress with a low cudaVersion, which is
 * what an old application does.
 */
#ifndef TESSERA_FAKE_CUDA_VARIANTS_H
#define TESSERA_FAKE_CUDA_VARIANTS_H

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * Per-thread-default-stream entry points.
 *
 * A NULL stream argument to one of these resolves to the calling thread's
 * per-thread default stream. The identically shaped entry point without the
 * suffix resolves NULL to the context's legacy default stream.
 * ========================================================================== */

CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                     unsigned int gridDimZ, unsigned int blockDimX,
                                     unsigned int blockDimY, unsigned int blockDimZ,
                                     unsigned int sharedMemBytes, CUstream hStream,
                                     void** kernelParams, void** extra);
CUresult CUDAAPI cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction f,
                                       void** kernelParams, void** extra);
CUresult CUDAAPI cuLaunchCooperativeKernel_ptsz(CUfunction f, unsigned int gridDimX,
                                                unsigned int gridDimY, unsigned int gridDimZ,
                                                unsigned int blockDimX, unsigned int blockDimY,
                                                unsigned int blockDimZ, unsigned int sharedMemBytes,
                                                CUstream hStream, void** kernelParams);
CUresult CUDAAPI cuGraphLaunch_ptsz(CUgraphExec hGraphExec, CUstream hStream);

CUresult CUDAAPI cuMemAllocAsync_ptsz(CUdeviceptr* dptr, size_t bytesize, CUstream hStream);
CUresult CUDAAPI cuMemAllocFromPoolAsync_ptsz(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool,
                                              CUstream hStream);
CUresult CUDAAPI cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream);

CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream hStream);
CUresult CUDAAPI cuStreamQuery_ptsz(CUstream hStream);
CUresult CUDAAPI cuStreamGetPriority_ptsz(CUstream hStream, int* priority);
CUresult CUDAAPI cuStreamGetFlags_ptsz(CUstream hStream, unsigned int* flags);
CUresult CUDAAPI cuStreamWaitEvent_ptsz(CUstream hStream, CUevent hEvent, unsigned int Flags);

CUresult CUDAAPI cuStreamBeginCapture_ptsz(CUstream hStream);
CUresult CUDAAPI cuStreamBeginCapture_v2_ptsz(CUstream hStream, CUstreamCaptureMode mode);
CUresult CUDAAPI cuStreamEndCapture_ptsz(CUstream hStream, CUgraph* phGraph);
CUresult CUDAAPI cuStreamIsCapturing_ptsz(CUstream hStream, CUstreamCaptureStatus* captureStatus);

CUresult CUDAAPI cuEventRecord_ptsz(CUevent hEvent, CUstream hStream);
CUresult CUDAAPI cuEventRecordWithFlags_ptsz(CUevent hEvent, CUstream hStream, unsigned int flags);

CUresult CUDAAPI cuMemcpyHtoD_v2_ptds(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount);
CUresult CUDAAPI cuMemcpyDtoH_v2_ptds(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult CUDAAPI cuMemcpyDtoD_v2_ptds(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                      size_t ByteCount);
CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dstDevice, const void* srcHost,
                                           size_t ByteCount, CUstream hStream);
CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                                           CUstream hStream);
CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount,
                                    CUstream hStream);

CUresult CUDAAPI cuMemsetD8_v2_ptds(CUdeviceptr dstDevice, unsigned char uc, size_t N);
CUresult CUDAAPI cuMemsetD32_v2_ptds(CUdeviceptr dstDevice, unsigned int ui, size_t N);
CUresult CUDAAPI cuMemsetD8Async_ptsz(CUdeviceptr dstDevice, unsigned char uc, size_t N,
                                      CUstream hStream);
CUresult CUDAAPI cuMemsetD32Async_ptsz(CUdeviceptr dstDevice, unsigned int ui, size_t N,
                                       CUstream hStream);

/* ==========================================================================
 * Function-pointer types for the entry points whose plain name is a cuda.h
 * macro. cudaTypedefs.h only declares these behind __CUDA_API_VERSION_INTERNAL,
 * which would rewrite every other name in the translation unit, so they are
 * repeated here under a TF_PFN_ prefix. The signatures are the ones cuda.h
 * 12.6 declares for the same entry points.
 * ========================================================================== */

/* Version 1: sizes and device pointers are 32 bits wide. */
typedef CUresult(CUDAAPI* TF_PFN_cuDeviceTotalMem_v1)(unsigned int* bytes, CUdevice dev);
typedef CUresult(CUDAAPI* TF_PFN_cuCtxCreate_v1)(CUcontext* pctx, unsigned int flags, CUdevice dev);
typedef CUresult(CUDAAPI* TF_PFN_cuCtxDestroy_v1)(CUcontext ctx);
typedef CUresult(CUDAAPI* TF_PFN_cuCtxPushCurrent_v1)(CUcontext ctx);
typedef CUresult(CUDAAPI* TF_PFN_cuCtxPopCurrent_v1)(CUcontext* pctx);
typedef CUresult(CUDAAPI* TF_PFN_cuDevicePrimaryCtxRelease_v1)(CUdevice dev);
typedef CUresult(CUDAAPI* TF_PFN_cuMemGetInfo_v1)(unsigned int* free, unsigned int* total);
typedef CUresult(CUDAAPI* TF_PFN_cuMemAlloc_v1)(unsigned int* dptr, unsigned int bytesize);
typedef CUresult(CUDAAPI* TF_PFN_cuMemAllocPitch_v1)(unsigned int* dptr, unsigned int* pPitch,
                                                     unsigned int WidthInBytes, unsigned int Height,
                                                     unsigned int ElementSizeBytes);
typedef CUresult(CUDAAPI* TF_PFN_cuMemFree_v1)(unsigned int dptr);
typedef CUresult(CUDAAPI* TF_PFN_cuMemcpyHtoD_v1)(unsigned int dstDevice, const void* srcHost,
                                                  unsigned int ByteCount);
typedef CUresult(CUDAAPI* TF_PFN_cuMemcpyDtoH_v1)(void* dstHost, unsigned int srcDevice,
                                                  unsigned int ByteCount);
typedef CUresult(CUDAAPI* TF_PFN_cuMemcpyDtoD_v1)(unsigned int dstDevice, unsigned int srcDevice,
                                                  unsigned int ByteCount);
typedef CUresult(CUDAAPI* TF_PFN_cuMemcpyHtoDAsync_v1)(unsigned int dstDevice, const void* srcHost,
                                                       unsigned int ByteCount, CUstream hStream);
typedef CUresult(CUDAAPI* TF_PFN_cuMemcpyDtoHAsync_v1)(void* dstHost, unsigned int srcDevice,
                                                       unsigned int ByteCount, CUstream hStream);
typedef CUresult(CUDAAPI* TF_PFN_cuMemsetD8_v1)(unsigned int dstDevice, unsigned char uc,
                                                unsigned int N);
typedef CUresult(CUDAAPI* TF_PFN_cuMemsetD32_v1)(unsigned int dstDevice, unsigned int ui,
                                                 unsigned int N);
typedef CUresult(CUDAAPI* TF_PFN_cuStreamDestroy_v1)(CUstream hStream);
typedef CUresult(CUDAAPI* TF_PFN_cuEventDestroy_v1)(CUevent hEvent);
typedef CUresult(CUDAAPI* TF_PFN_cuStreamBeginCapture_v1)(CUstream hStream);
typedef CUresult(CUDAAPI* TF_PFN_cuGraphInstantiate_v1)(CUgraphExec* phGraphExec, CUgraph hGraph,
                                                        CUgraphNode* phErrorNode, char* logBuffer,
                                                        size_t bufferSize);

/* Version 2 and later, for the names cuda.h hides behind a macro. */
typedef CUresult(CUDAAPI* TF_PFN_cuMemAlloc_v2)(CUdeviceptr* dptr, size_t bytesize);
typedef CUresult(CUDAAPI* TF_PFN_cuMemFree_v2)(CUdeviceptr dptr);
typedef CUresult(CUDAAPI* TF_PFN_cuDeviceTotalMem_v2)(size_t* bytes, CUdevice dev);
typedef CUresult(CUDAAPI* TF_PFN_cuCtxCreate_v2)(CUcontext* pctx, unsigned int flags, CUdevice dev);
typedef CUresult(CUDAAPI* TF_PFN_cuCtxCreate_v3)(CUcontext* pctx, CUexecAffinityParam* paramsArray,
                                                 int numParams, unsigned int flags, CUdevice dev);
typedef CUresult(CUDAAPI* TF_PFN_cuCtxCreate_v4)(CUcontext* pctx,
                                                 CUctxCreateParams* ctxCreateParams,
                                                 unsigned int flags, CUdevice dev);
typedef CUresult(CUDAAPI* TF_PFN_cuStreamBeginCapture_v2)(CUstream hStream,
                                                          CUstreamCaptureMode mode);
typedef CUresult(CUDAAPI* TF_PFN_cuGraphInstantiate_v2)(CUgraphExec* phGraphExec, CUgraph hGraph,
                                                        CUgraphNode* phErrorNode, char* logBuffer,
                                                        size_t bufferSize);
typedef CUresult(CUDAAPI* TF_PFN_cuGraphInstantiateWithFlags)(CUgraphExec* phGraphExec,
                                                              CUgraph hGraph,
                                                              unsigned long long flags);

/* Both forms of the lookup function. */
typedef CUresult(CUDAAPI* TF_PFN_cuGetProcAddress_v1)(const char* symbol, void** pfn,
                                                      int cudaVersion, cuuint64_t flags);
typedef CUresult(CUDAAPI* TF_PFN_cuGetProcAddress_v2)(const char* symbol, void** pfn,
                                                      int cudaVersion, cuuint64_t flags,
                                                      CUdriverProcAddressQueryResult* symbolStatus);

/* Entry points that take a stream, for tests that resolve them by name. */
typedef CUresult(CUDAAPI* TF_PFN_cuLaunchKernel)(CUfunction f, unsigned int gridDimX,
                                                 unsigned int gridDimY, unsigned int gridDimZ,
                                                 unsigned int blockDimX, unsigned int blockDimY,
                                                 unsigned int blockDimZ,
                                                 unsigned int sharedMemBytes, CUstream hStream,
                                                 void** kernelParams, void** extra);
typedef CUresult(CUDAAPI* TF_PFN_cuStreamSynchronize)(CUstream hStream);
typedef CUresult(CUDAAPI* TF_PFN_cuEventRecord)(CUevent hEvent, CUstream hStream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TESSERA_FAKE_CUDA_VARIANTS_H */
