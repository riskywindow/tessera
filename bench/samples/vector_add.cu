// vectorAdd-class sample, driven through the CUDA *runtime* API.
//
// Why the runtime API and not the driver API: cudart does not call the exported
// cu* symbols. It resolves every driver entry point through cuGetProcAddress /
// cuGetProcAddress_v2, including the lookup function itself. That is exactly
// the path the shim has to intercept (CLAUDE.md, "Shim traps"), so the G8/G9
// sample must go through cudaMalloc/cudaMemcpy/<<<>>> rather than cuMemAlloc
// and cuLaunchKernel.
//
// Properties this sample is built to have:
//   * Deterministic inputs from a fixed seed, with no host RNG library
//     involved: a splitmix64 stream quantised onto multiples of 1/8. Every
//     input and every sum is exactly representable in binary32, so a + b is
//     exact and the output bytes do not depend on the GPU, the compiler, or
//     fast-math settings. The SHA-256 is therefore comparable across machines
//     and, more to the point, with and without the shim (G8).
//   * A kernel-launch count the program itself knows: the buffer is processed
//     in kChunks chunks, one launch each, and a counter is incremented at the
//     single launch site. Nothing else in the program launches a kernel, so
//     this count is the ground truth the shim's counter must match (G9).
//   * One machine-readable line on stdout, prefixed TESSERA_JSON, so
//     infra/modal_gpu.py can parse it without screen-scraping.
//
// Exit codes: 0 on success, non-zero on any CUDA error or verification
// mismatch. Nothing is masked.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// --- Configuration. Fixed, so the run is reproducible. ---------------------
constexpr int kElements = 1 << 20;  // 1048576
constexpr int kChunks = 16;         // one kernel launch per chunk
constexpr int kThreadsPerBlock = 256;
constexpr std::uint64_t kSeed = 20260919ULL;

// --- Deterministic host RNG (splitmix64). ----------------------------------
// Values are quantised to multiples of 1/8 below 128, so a, b and a + b are all
// exact in binary32 and the output hash is bit-stable everywhere.
std::uint64_t Splitmix64(std::uint64_t* state) {
  std::uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

float QuantisedFloat(std::uint64_t* state) {
  const std::uint64_t r = Splitmix64(state) % 1024ULL;  // 0 .. 1023
  return static_cast<float>(r) * 0.125f;                // 0 .. 127.875, exact
}

// --- SHA-256, host side, self-contained (no OpenSSL in the image). ---------
struct Sha256 {
  std::uint32_t h[8];
  std::uint64_t total_bits;
  std::uint8_t buf[64];
  std::size_t buf_len;
};

std::uint32_t Rotr32(std::uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}

void Sha256Init(Sha256* s) {
  static const std::uint32_t kInit[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                         0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::memcpy(s->h, kInit, sizeof(kInit));
  s->total_bits = 0;
  s->buf_len = 0;
}

void Sha256Block(Sha256* s, const std::uint8_t* p) {
  static const std::uint32_t k[64] = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
      0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
      0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
      0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
      0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
      0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
      0xc67178f2U};
  std::uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<std::uint32_t>(p[4 * i]) << 24) |
           (static_cast<std::uint32_t>(p[4 * i + 1]) << 16) |
           (static_cast<std::uint32_t>(p[4 * i + 2]) << 8) |
           static_cast<std::uint32_t>(p[4 * i + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    const std::uint32_t s0 = Rotr32(w[i - 15], 7) ^ Rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const std::uint32_t s1 = Rotr32(w[i - 2], 17) ^ Rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  std::uint32_t a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3];
  std::uint32_t e = s->h[4], f = s->h[5], g = s->h[6], hh = s->h[7];
  for (int i = 0; i < 64; ++i) {
    const std::uint32_t S1 = Rotr32(e, 6) ^ Rotr32(e, 11) ^ Rotr32(e, 25);
    const std::uint32_t ch = (e & f) ^ ((~e) & g);
    const std::uint32_t t1 = hh + S1 + ch + k[i] + w[i];
    const std::uint32_t S0 = Rotr32(a, 2) ^ Rotr32(a, 13) ^ Rotr32(a, 22);
    const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t t2 = S0 + maj;
    hh = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  s->h[0] += a;
  s->h[1] += b;
  s->h[2] += c;
  s->h[3] += d;
  s->h[4] += e;
  s->h[5] += f;
  s->h[6] += g;
  s->h[7] += hh;
}

void Sha256Update(Sha256* s, const void* data, std::size_t len) {
  const std::uint8_t* p = static_cast<const std::uint8_t*>(data);
  s->total_bits += static_cast<std::uint64_t>(len) * 8ULL;
  while (len > 0) {
    const std::size_t take = (64 - s->buf_len) < len ? (64 - s->buf_len) : len;
    std::memcpy(s->buf + s->buf_len, p, take);
    s->buf_len += take;
    p += take;
    len -= take;
    if (s->buf_len == 64) {
      Sha256Block(s, s->buf);
      s->buf_len = 0;
    }
  }
}

void Sha256Final(Sha256* s, char out_hex[65]) {
  const std::uint64_t bits = s->total_bits;
  const std::uint8_t pad = 0x80U;
  Sha256Update(s, &pad, 1);
  s->total_bits = bits;  // padding is not message length
  const std::uint8_t zero = 0U;
  while (s->buf_len != 56) {
    Sha256Update(s, &zero, 1);
    s->total_bits = bits;
  }
  std::uint8_t len_be[8];
  for (int i = 0; i < 8; ++i) {
    len_be[i] = static_cast<std::uint8_t>((bits >> (56 - 8 * i)) & 0xFFU);
  }
  Sha256Update(s, len_be, 8);
  static const char* kHex = "0123456789abcdef";
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      const std::uint8_t byte = static_cast<std::uint8_t>((s->h[i] >> (24 - 8 * j)) & 0xFFU);
      out_hex[i * 8 + j * 2] = kHex[byte >> 4];
      out_hex[i * 8 + j * 2 + 1] = kHex[byte & 0x0FU];
    }
  }
  out_hex[64] = '\0';
}

}  // namespace

// --- Device code. One kernel, one launch site. -----------------------------
__global__ void VectorAddChunk(const float* a, const float* b, float* c, int offset, int count) {
  const int i = offset + static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < offset + count) {
    c[i] = a[i] + b[i];
  }
}

#define TESSERA_CUDA_CHECK(expr)                                                                   \
  do {                                                                                             \
    const cudaError_t err__ = (expr);                                                              \
    if (err__ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "vector_add: %s failed at %s:%d: %s (%d)\n", #expr, __FILE__, __LINE__, \
                   cudaGetErrorString(err__), static_cast<int>(err__));                            \
      return 2;                                                                                    \
    }                                                                                              \
  } while (0)

int main(int argc, char** argv) {
  int device = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      device = std::atoi(argv[i + 1]);
      ++i;
    } else {
      std::fprintf(stderr, "vector_add: unknown argument '%s'\n", argv[i]);
      return 64;
    }
  }

  TESSERA_CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp prop;
  TESSERA_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int runtime_version = 0;
  int driver_version = 0;
  TESSERA_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  TESSERA_CUDA_CHECK(cudaDriverGetVersion(&driver_version));

  std::vector<float> host_a(kElements);
  std::vector<float> host_b(kElements);
  std::vector<float> host_c(kElements, 0.0f);
  std::uint64_t state = kSeed;
  for (int i = 0; i < kElements; ++i) {
    host_a[i] = QuantisedFloat(&state);
    host_b[i] = QuantisedFloat(&state);
  }

  const std::size_t bytes = static_cast<std::size_t>(kElements) * sizeof(float);
  float* dev_a = nullptr;
  float* dev_b = nullptr;
  float* dev_c = nullptr;
  TESSERA_CUDA_CHECK(cudaMalloc(&dev_a, bytes));
  TESSERA_CUDA_CHECK(cudaMalloc(&dev_b, bytes));
  TESSERA_CUDA_CHECK(cudaMalloc(&dev_c, bytes));
  TESSERA_CUDA_CHECK(cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  TESSERA_CUDA_CHECK(cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice));
  TESSERA_CUDA_CHECK(cudaMemset(dev_c, 0, bytes));

  // The counted window. kChunks launches, and no other kernel in the program.
  long long launches = 0;
  static_assert(kElements % kChunks == 0, "chunks must divide the buffer");
  constexpr int kChunkElems = kElements / kChunks;
  constexpr int kBlocks = (kChunkElems + kThreadsPerBlock - 1) / kThreadsPerBlock;
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    VectorAddChunk<<<kBlocks, kThreadsPerBlock>>>(dev_a, dev_b, dev_c, chunk * kChunkElems,
                                                  kChunkElems);
    ++launches;
    TESSERA_CUDA_CHECK(cudaGetLastError());
  }
  TESSERA_CUDA_CHECK(cudaDeviceSynchronize());

  TESSERA_CUDA_CHECK(cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost));
  TESSERA_CUDA_CHECK(cudaFree(dev_a));
  TESSERA_CUDA_CHECK(cudaFree(dev_b));
  TESSERA_CUDA_CHECK(cudaFree(dev_c));

  // Verify against the host reference. The inputs were chosen so this is an
  // exact equality, not a tolerance.
  long long mismatches = 0;
  for (int i = 0; i < kElements; ++i) {
    if (host_c[i] != host_a[i] + host_b[i]) {
      ++mismatches;
    }
  }

  Sha256 sha;
  Sha256Init(&sha);
  Sha256Update(&sha, host_c.data(), bytes);
  char hex[65];
  Sha256Final(&sha, hex);

  std::printf(
      "TESSERA_JSON {\"sample\":\"vector_add\",\"elements\":%d,\"chunks\":%d,"
      "\"threads_per_block\":%d,\"blocks_per_chunk\":%d,\"seed\":%llu,"
      "\"kernel_launches\":%lld,\"sha256\":\"%s\",\"mismatches\":%lld,"
      "\"device_name\":\"%s\",\"compute_capability\":\"%d.%d\","
      "\"multiprocessor_count\":%d,\"cuda_runtime_version\":%d,"
      "\"cuda_driver_version\":%d}\n",
      kElements, kChunks, kThreadsPerBlock, kBlocks, static_cast<unsigned long long>(kSeed),
      launches, hex, mismatches, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
      runtime_version, driver_version);
  std::fflush(stdout);

  if (mismatches != 0) {
    std::fprintf(stderr, "vector_add: %lld mismatches against host reference\n", mismatches);
    return 1;
  }
  return 0;
}
