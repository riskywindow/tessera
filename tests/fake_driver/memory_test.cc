// Device memory is real host memory, so a simulated kernel computes real
// results and a test can check them bit for bit. That is what makes the
// shim's I-3 "outputs are identical with and without Tessera" claim testable
// on a machine with no GPU.

#include <cstdint>
#include <cstring>
#include <vector>

#include "fake_fixture.h"

namespace tessera_test {
namespace {

// c[i] = a[i] + b[i], over n elements. The parameter array is the same one the
// application passed to cuLaunchKernel: pointers to the arguments.
float* as_float_ptr(const void* param) {
  const CUdeviceptr p = *static_cast<const CUdeviceptr*>(param);
  return reinterpret_cast<float*>(static_cast<uintptr_t>(p));
}

void vector_add_body(void** params) {
  const float* a = as_float_ptr(params[0]);
  const float* b = as_float_ptr(params[1]);
  float* c = as_float_ptr(params[2]);
  const uint32_t n = *static_cast<const uint32_t*>(params[3]);
  for (uint32_t i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

class MemoryTest : public FakeDriverTest {};

TEST_F(MemoryTest, CopyToDeviceAndBackIsBitwiseIdentical) {
  constexpr size_t kN = 1024;
  std::vector<uint32_t> src(kN);
  for (size_t i = 0; i < kN; ++i) {
    src[i] = static_cast<uint32_t>(i * 2654435761u);
  }
  std::vector<uint32_t> dst(kN, 0);

  CUdeviceptr d = 0;
  ASSERT_CU(cuMemAlloc(&d, kN * sizeof(uint32_t)));
  ASSERT_CU(cuMemcpyHtoD(d, src.data(), kN * sizeof(uint32_t)));
  ASSERT_CU(cuMemcpyDtoH(dst.data(), d, kN * sizeof(uint32_t)));
  EXPECT_EQ(0, std::memcmp(src.data(), dst.data(), kN * sizeof(uint32_t)));
  ASSERT_CU(cuMemFree(d));
}

TEST_F(MemoryTest, AKernelComputesIntoDeviceMemory) {
  constexpr uint32_t kN = 256;
  std::vector<float> a(kN);
  std::vector<float> b(kN);
  std::vector<float> out(kN, 0.0F);
  for (uint32_t i = 0; i < kN; ++i) {
    a[i] = static_cast<float>(i) * 0.5F;
    b[i] = static_cast<float>(kN - i) * 0.25F;
  }

  CUdeviceptr da = 0;
  CUdeviceptr db = 0;
  CUdeviceptr dc = 0;
  const size_t bytes = kN * sizeof(float);
  ASSERT_CU(cuMemAlloc(&da, bytes));
  ASSERT_CU(cuMemAlloc(&db, bytes));
  ASSERT_CU(cuMemAlloc(&dc, bytes));
  ASSERT_CU(cuMemcpyHtoD(da, a.data(), bytes));
  ASSERT_CU(cuMemcpyHtoD(db, b.data(), bytes));

  CUfunction f = tf_register_kernel("vector_add", vector_add_body, 100);
  ASSERT_NE(nullptr, f);
  tf_set_kernel_param_count(f, 4);

  uint32_t n = kN;
  void* params[] = {&da, &db, &dc, &n};
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, s, params, nullptr));
  ASSERT_CU(cuStreamSynchronize(s));
  ASSERT_CU(cuMemcpyDtoH(out.data(), dc, bytes));

  std::vector<float> expected(kN);
  for (uint32_t i = 0; i < kN; ++i) {
    expected[i] = a[i] + b[i];
  }
  EXPECT_EQ(0, std::memcmp(expected.data(), out.data(), bytes));

  ASSERT_CU(cuStreamDestroy(s));
  ASSERT_CU(cuMemFree(da));
  ASSERT_CU(cuMemFree(db));
  ASSERT_CU(cuMemFree(dc));
}

TEST_F(MemoryTest, AsyncCopiesAndLaunchesAreOrderedOnTheStream) {
  constexpr uint32_t kN = 64;
  std::vector<float> a(kN, 1.5F);
  std::vector<float> b(kN, 2.25F);
  std::vector<float> out(kN, 0.0F);

  CUdeviceptr da = 0;
  CUdeviceptr db = 0;
  CUdeviceptr dc = 0;
  const size_t bytes = kN * sizeof(float);
  ASSERT_CU(cuMemAlloc(&da, bytes));
  ASSERT_CU(cuMemAlloc(&db, bytes));
  ASSERT_CU(cuMemAlloc(&dc, bytes));

  CUfunction f = tf_register_kernel("vector_add", vector_add_body, 10);
  tf_set_kernel_param_count(f, 4);
  uint32_t n = kN;
  void* params[] = {&da, &db, &dc, &n};

  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  // Everything is queued before anything runs; the copies must still land in
  // order around the launch.
  ASSERT_CU(cuMemcpyHtoDAsync(da, a.data(), bytes, s));
  ASSERT_CU(cuMemcpyHtoDAsync(db, b.data(), bytes, s));
  ASSERT_CU(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, s, params, nullptr));
  ASSERT_CU(cuMemcpyDtoHAsync(out.data(), dc, bytes, s));
  EXPECT_EQ(0.0F, out[0]);
  ASSERT_CU(cuStreamSynchronize(s));

  for (uint32_t i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(3.75F, out[i]);
  }

  ASSERT_CU(cuStreamDestroy(s));
  ASSERT_CU(cuMemFree(da));
  ASSERT_CU(cuMemFree(db));
  ASSERT_CU(cuMemFree(dc));
}

TEST_F(MemoryTest, MemsetFillsWithTheRequestedPattern) {
  constexpr size_t kWords = 128;
  CUdeviceptr d = 0;
  ASSERT_CU(cuMemAlloc(&d, kWords * sizeof(uint32_t)));

  ASSERT_CU(cuMemsetD8(d, 0xAB, kWords * sizeof(uint32_t)));
  std::vector<uint8_t> bytes(kWords * sizeof(uint32_t), 0);
  ASSERT_CU(cuMemcpyDtoH(bytes.data(), d, bytes.size()));
  for (uint8_t v : bytes) {
    EXPECT_EQ(0xAB, v);
  }

  ASSERT_CU(cuMemsetD32(d, 0xDEADBEEF, kWords));
  std::vector<uint32_t> words(kWords, 0);
  ASSERT_CU(cuMemcpyDtoH(words.data(), d, kWords * sizeof(uint32_t)));
  for (uint32_t v : words) {
    EXPECT_EQ(0xDEADBEEFu, v);
  }

  ASSERT_CU(cuMemFree(d));
}

TEST_F(MemoryTest, AsyncMemsetIsOrderedOnTheStream) {
  constexpr size_t kWords = 32;
  CUdeviceptr d = 0;
  ASSERT_CU(cuMemAlloc(&d, kWords * sizeof(uint32_t)));
  std::vector<uint32_t> words(kWords, 0);

  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(cuMemsetD32Async(d, 7, kWords, s));
  ASSERT_CU(cuMemcpyDtoHAsync(words.data(), d, kWords * sizeof(uint32_t), s));
  ASSERT_CU(cuStreamSynchronize(s));
  for (uint32_t v : words) {
    EXPECT_EQ(7u, v);
  }

  ASSERT_CU(cuMemsetD8Async(d, 0, kWords * sizeof(uint32_t), s));
  ASSERT_CU(cuMemcpyDtoHAsync(words.data(), d, kWords * sizeof(uint32_t), s));
  ASSERT_CU(cuStreamSynchronize(s));
  for (uint32_t v : words) {
    EXPECT_EQ(0u, v);
  }

  ASSERT_CU(cuStreamDestroy(s));
  ASSERT_CU(cuMemFree(d));
}

TEST_F(MemoryTest, DeviceToDeviceCopy) {
  constexpr size_t kN = 100;
  std::vector<uint32_t> src(kN);
  for (size_t i = 0; i < kN; ++i) {
    src[i] = static_cast<uint32_t>(i + 1);
  }
  std::vector<uint32_t> back(kN, 0);

  CUdeviceptr a = 0;
  CUdeviceptr b = 0;
  ASSERT_CU(cuMemAlloc(&a, kN * sizeof(uint32_t)));
  ASSERT_CU(cuMemAlloc(&b, kN * sizeof(uint32_t)));
  ASSERT_CU(cuMemcpyHtoD(a, src.data(), kN * sizeof(uint32_t)));
  ASSERT_CU(cuMemcpyDtoD(b, a, kN * sizeof(uint32_t)));
  ASSERT_CU(cuMemcpyDtoH(back.data(), b, kN * sizeof(uint32_t)));
  EXPECT_EQ(src, back);

  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  ASSERT_CU(cuMemsetD32(b, 0, kN));
  ASSERT_CU(cuMemcpyAsync(b, a, kN * sizeof(uint32_t), s));
  ASSERT_CU(cuStreamSynchronize(s));
  ASSERT_CU(cuMemcpyDtoH(back.data(), b, kN * sizeof(uint32_t)));
  EXPECT_EQ(src, back);

  ASSERT_CU(cuStreamDestroy(s));
  ASSERT_CU(cuMemFree(a));
  ASSERT_CU(cuMemFree(b));
}

TEST_F(MemoryTest, MemGetInfoTracksAllocations) {
  size_t free_before = 0;
  size_t total = 0;
  ASSERT_CU(cuMemGetInfo(&free_before, &total));
  EXPECT_GT(total, 0u);
  EXPECT_LE(free_before, total);

  CUdeviceptr d = 0;
  constexpr size_t kBytes = 1 << 20;
  ASSERT_CU(cuMemAlloc(&d, kBytes));
  size_t free_after = 0;
  ASSERT_CU(cuMemGetInfo(&free_after, &total));
  EXPECT_EQ(free_before - kBytes, free_after);

  ASSERT_CU(cuMemFree(d));
  ASSERT_CU(cuMemGetInfo(&free_after, &total));
  EXPECT_EQ(free_before, free_after);
}

TEST_F(MemoryTest, AllocationPastTheDeviceBudgetFails) {
  tf_set_device_total_mem(1 << 20);
  CUdeviceptr d = 0;
  EXPECT_EQ(CUDA_ERROR_OUT_OF_MEMORY, cuMemAlloc(&d, 4u << 20));
  ASSERT_CU(cuMemAlloc(&d, 512u << 10));
  CUdeviceptr e = 0;
  EXPECT_EQ(CUDA_ERROR_OUT_OF_MEMORY, cuMemAlloc(&e, 1u << 20));
  ASSERT_CU(cuMemFree(d));
  ASSERT_CU(cuMemAlloc(&e, 1u << 20));
  ASSERT_CU(cuMemFree(e));
}

TEST_F(MemoryTest, FreeingSomethingThatIsNotAnAllocationFails) {
  CUdeviceptr d = 0;
  ASSERT_CU(cuMemAlloc(&d, 4096));
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuMemFree(d + 64));
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuMemFree(0));
  ASSERT_CU(cuMemFree(d));
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuMemFree(d));
}

TEST_F(MemoryTest, CopyOutsideAnAllocationIsRejected) {
  CUdeviceptr d = 0;
  ASSERT_CU(cuMemAlloc(&d, 1024));
  std::vector<uint8_t> host(4096, 0);
  // The allocation is rounded up to 1024 bytes; 4096 runs off the end.
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuMemcpyHtoD(d, host.data(), 4096));
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuMemcpyDtoH(host.data(), d, 4096));
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuMemsetD8(d, 0, 4096));
  ASSERT_CU(cuMemFree(d));
}

TEST_F(MemoryTest, PitchedAllocationsArePadded) {
  CUdeviceptr d = 0;
  size_t pitch = 0;
  ASSERT_CU(cuMemAllocPitch(&d, &pitch, 300, 16, 4));
  EXPECT_GE(pitch, 300u);
  EXPECT_EQ(0u, pitch % 256u);
  ASSERT_CU(cuMemsetD8(d, 0x11, pitch * 16));
  ASSERT_CU(cuMemFree(d));
}

TEST_F(MemoryTest, AsyncAllocationAndOrderedFree) {
  CUstream s = nullptr;
  ASSERT_CU(cuStreamCreate(&s, 0));
  CUdeviceptr d = 0;
  ASSERT_CU(cuMemAllocAsync(&d, 4096, s));
  ASSERT_NE(0u, d);

  std::vector<uint8_t> host(4096, 0x5A);
  ASSERT_CU(cuMemcpyHtoDAsync(d, host.data(), host.size(), s));
  ASSERT_CU(cuMemFreeAsync(d, s));
  ASSERT_CU(cuStreamSynchronize(s));
  // After the ordered free the pointer is gone.
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuMemFree(d));

  CUmemoryPool pool = nullptr;
  ASSERT_CU(cuDeviceGetDefaultMemPool(&pool, dev_));
  CUdeviceptr p = 0;
  ASSERT_CU(cuMemAllocFromPoolAsync(&p, 256, pool, s));
  ASSERT_CU(cuMemFreeAsync(p, s));
  ASSERT_CU(cuStreamSynchronize(s));
  ASSERT_CU(cuStreamDestroy(s));
}

TEST_F(MemoryTest, ManagedAllocationBehavesLikeDeviceMemory) {
  CUdeviceptr d = 0;
  ASSERT_CU(cuMemAllocManaged(&d, 256, CU_MEM_ATTACH_GLOBAL));
  std::vector<uint8_t> host(256, 0x7E);
  ASSERT_CU(cuMemcpyHtoD(d, host.data(), host.size()));
  std::vector<uint8_t> back(256, 0);
  ASSERT_CU(cuMemcpyDtoH(back.data(), d, back.size()));
  EXPECT_EQ(host, back);
  ASSERT_CU(cuMemFree(d));
}

TEST_F(MemoryTest, VirtualMemoryReserveCreateMapUnmap) {
  const size_t page = 64u << 10;
  CUdeviceptr base = 0;
  ASSERT_CU(cuMemAddressReserve(&base, page, 0, 0, 0));
  ASSERT_NE(0u, base);

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev_;
  CUmemGenericAllocationHandle handle = 0;
  ASSERT_CU(cuMemCreate(&handle, page, &prop, 0));

  ASSERT_CU(cuMemMap(base, page, 0, handle, 0));
  CUmemAccessDesc access{};
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ASSERT_CU(cuMemSetAccess(base, page, &access, 1));

  std::vector<uint8_t> host(page, 0x3C);
  ASSERT_CU(cuMemcpyHtoD(base, host.data(), page));
  std::vector<uint8_t> back(page, 0);
  ASSERT_CU(cuMemcpyDtoH(back.data(), base, page));
  EXPECT_EQ(host, back);

  // Mapped memory is not an ordinary allocation.
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuMemFree(base));

  ASSERT_CU(cuMemUnmap(base, page));
  // Once unmapped the range is no longer usable as device memory.
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuMemcpyDtoH(back.data(), base, page));
  ASSERT_CU(cuMemRelease(handle));
  ASSERT_CU(cuMemAddressFree(base, page));
}

TEST_F(MemoryTest, VersionOneEntryPointsHaveTheirOwnPointerSpace) {
  auto alloc_v1 = reinterpret_cast<TF_PFN_cuMemAlloc_v1>(tf_symbol_address("cuMemAlloc"));
  auto free_v1 = reinterpret_cast<TF_PFN_cuMemFree_v1>(tf_symbol_address("cuMemFree"));
  auto htod_v1 = reinterpret_cast<TF_PFN_cuMemcpyHtoD_v1>(tf_symbol_address("cuMemcpyHtoD"));
  auto dtoh_v1 = reinterpret_cast<TF_PFN_cuMemcpyDtoH_v1>(tf_symbol_address("cuMemcpyDtoH"));
  ASSERT_NE(nullptr, alloc_v1);
  ASSERT_NE(nullptr, free_v1);

  unsigned int d = 0;
  ASSERT_CU(alloc_v1(&d, 512));
  ASSERT_NE(0u, d);

  std::vector<uint8_t> host(512);
  for (size_t i = 0; i < host.size(); ++i) {
    host[i] = static_cast<uint8_t>(i);
  }
  ASSERT_CU(htod_v1(d, host.data(), 512));
  std::vector<uint8_t> back(512, 0);
  ASSERT_CU(dtoh_v1(back.data(), d, 512));
  EXPECT_EQ(host, back);

  // A version-1 token means nothing to the version-2 entry points.
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, cuMemFree(static_cast<CUdeviceptr>(d)));
  ASSERT_CU(free_v1(d));
  EXPECT_EQ(CUDA_ERROR_INVALID_VALUE, free_v1(d));
}

TEST_F(MemoryTest, DeviceReportsPlausibleProperties) {
  char name[128] = {};
  ASSERT_CU(cuDeviceGetName(name, sizeof(name), dev_));
  EXPECT_GT(std::strlen(name), 0u);

  size_t total = 0;
  ASSERT_CU(cuDeviceTotalMem(&total, dev_));
  EXPECT_EQ(UINT64_C(16) << 30, total);

  int sms = 0;
  ASSERT_CU(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_));
  EXPECT_EQ(58, sms);
  tf_set_device_sm_count(132);
  ASSERT_CU(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev_));
  EXPECT_EQ(132, sms);

  int major = 0;
  int minor = 0;
  ASSERT_CU(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev_));
  ASSERT_CU(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev_));
  EXPECT_GE(major, 3);
  EXPECT_GE(minor, 0);

  int count = 0;
  ASSERT_CU(cuDeviceGetCount(&count));
  EXPECT_EQ(1, count);
  CUdevice other = 0;
  EXPECT_EQ(CUDA_ERROR_INVALID_DEVICE, cuDeviceGet(&other, 1));
}

}  // namespace
}  // namespace tessera_test
