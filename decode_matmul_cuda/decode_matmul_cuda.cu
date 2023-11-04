#include <iostream>
#include <cuda.h>
#include <mma.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>

#include <torch/types.h>
#include <torch/extension.h>

using namespace nvcuda;

#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) 	        do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(false)
#define gpuErrchk(ans)          do { gpuAssert((ans), __FILE__, __LINE__); } while (false)

#define WMMA_M                  8
#define WMMA_N                  32
#define WMMA_K                  16

#define BLOCK_SIZE              1024
#define WARP_SIZE               32

#define SMEM_SIZE_MAX           (48 * 1024)


__host__ static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}


__device__ static inline uint64_t decode8weights(
        uint16_t weight_compressed,
        const int64_t *__restrict__ codebook_abs) {

    uint8_t bits_abs = weight_compressed & ((1 << 8) - 1);
    uint8_t bits_sign = (weight_compressed >> 8) & ((1 << 7) - 1);
    bool bit_shift = (weight_compressed >> 15) & ((1 << 1) - 1);

    uint64_t decoded_sign = bits_sign | ((__popc(bits_sign) & 1) << 7);
    decoded_sign |= (decoded_sign << (32-4));
    decoded_sign |= (decoded_sign << (16-2));
    decoded_sign |= (decoded_sign << (8-1));
    decoded_sign &= 0x0101010101010101;
    decoded_sign *= 255 - 3;

    int64_t packed = codebook_abs[bits_abs] ^ decoded_sign;
    packed -= bit_shift * 0x0202020202020202;
    packed |= 0x0101010101010101;

    return packed;
}


__global__ static void
__launch_bounds__(BLOCK_SIZE)
decode_matmul_kernel(
        int32_t *__restrict__ output,
        const int8_t *__restrict__ x,
        const int16_t *__restrict__ weights_compressed,
        const int64_t *__restrict__ codebook_abs,
        int64_t GLOBAL_M,
        int64_t GLOBAL_N,
        int64_t GLOBAL_K) {

    int64_t warpId = threadIdx.x / WARP_SIZE;
    int64_t laneId = threadIdx.x % WARP_SIZE;

    constexpr size_t SMEM_SIZE_PER_WARP = std::max(WMMA_N*WMMA_K / 8, WMMA_M*WMMA_N / 2);
    __shared__ uint64_t shared_scratch[BLOCK_SIZE/WARP_SIZE * SMEM_SIZE_PER_WARP];
    uint64_t* local_scratch = shared_scratch + warpId * SMEM_SIZE_PER_WARP;

    int64_t TILES_M = GLOBAL_M / WMMA_M;
    int64_t TILES_N = GLOBAL_N / WMMA_N;
    int64_t TILES_K = GLOBAL_K / WMMA_K;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag;

    for (int64_t blockPos = blockIdx.x; blockPos < TILES_M * TILES_N; blockPos += gridDim.x) {
        int64_t M_TILE = blockPos / TILES_N;
        int64_t N_TILE = blockPos % TILES_N;

        wmma::fill_fragment(acc_frag, 0);

        for (int64_t K_TILE = warpId; K_TILE < TILES_K; K_TILE += BLOCK_SIZE/WARP_SIZE) {
            int64_t aRow = M_TILE * WMMA_M;
            int64_t aCol = K_TILE * WMMA_K;
            wmma::load_matrix_sync(a_frag, x + aRow*GLOBAL_K + aCol, GLOBAL_K);

            int64_t bRow = N_TILE * WMMA_N;
            int64_t bCol = K_TILE * WMMA_K/8;
            for (int64_t i = laneId; i < WMMA_N * WMMA_K/8; i += WARP_SIZE) {
                int64_t THREAD_N = i / (WMMA_K/8);
                int64_t THREAD_K = i % (WMMA_K/8);
                uint16_t weight_compressed = weights_compressed[(bRow+THREAD_N) * (GLOBAL_K/8) + (bCol+THREAD_K)];
                local_scratch[i] = decode8weights(weight_compressed, codebook_abs);
            }
            wmma::load_matrix_sync(b_frag, reinterpret_cast<int8_t *>(local_scratch), WMMA_K);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        wmma::store_matrix_sync(reinterpret_cast<int32_t *>(local_scratch), acc_frag, WMMA_N, wmma::mem_row_major);

        __syncthreads();

        for (int64_t i = threadIdx.x; i < WMMA_M * WMMA_N; i += BLOCK_SIZE) {
            int32_t acc = 0;
            for (int64_t j = 0; j < BLOCK_SIZE/WARP_SIZE; j += 1) {
                acc += reinterpret_cast<int32_t *>(shared_scratch)[j * WMMA_M * WMMA_N + i];
            }

            int64_t cRow = M_TILE * WMMA_M + i / WMMA_N;
            int64_t cCol = N_TILE * WMMA_N + i % WMMA_N;
            output[cRow*GLOBAL_N + cCol] = acc;
        }
    }
}


__host__ torch::Tensor decode_matmul(
        torch::Tensor x,
        torch::Tensor weights_compressed,
        torch::Tensor codebook_abs) {
    CHECK_INPUT(x);
    CHECK_INPUT(weights_compressed);
    CHECK_INPUT(codebook_abs);

    TORCH_CHECK(x.scalar_type() == torch::kInt8);
    TORCH_CHECK(weights_compressed.scalar_type() == torch::kInt16);
    TORCH_CHECK(codebook_abs.scalar_type() == torch::kInt64);
    TORCH_CHECK(x.size(-1) == weights_compressed.size(-1) << 3);

    int64_t GLOBAL_M = x.size(-2);
    int64_t GLOBAL_N = weights_compressed.size(-2);
    int64_t GLOBAL_K = x.size(-1);

    TORCH_CHECK(GLOBAL_M % WMMA_M == 0, "GLOBAL_M is not divisible by WMMA_M");
    TORCH_CHECK(GLOBAL_N % WMMA_N == 0, "GLOBAL_N is not divisible by WMMA_N");
    TORCH_CHECK(GLOBAL_K % WMMA_K == 0, "GLOBAL_K is not divisible by WMMA_K");

    at::DeviceGuard guard(x.device());
    torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    torch::Tensor output = torch::zeros(std::vector<int64_t>{GLOBAL_M, GLOBAL_N}, options);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, x.get_device());
    int64_t grid_size = static_cast<int64_t>(deviceProp.multiProcessorCount);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    decode_matmul_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            output.data_ptr<int32_t>(),
            x.data_ptr<int8_t>(),
            weights_compressed.data_ptr<int16_t>(),
            codebook_abs.data_ptr<int64_t>(),
            GLOBAL_M,
            GLOBAL_N,
            GLOBAL_K);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_matmul", &decode_matmul, "Fused Decode Matrix Multiplication");
}
