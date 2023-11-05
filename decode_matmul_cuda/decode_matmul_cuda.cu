#include <iostream>
#include <cuda.h>
#include <cuda_pipeline_primitives.h>
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

#define MMA_M                   8
#define MMA_N                   16
#define MMA_K                   32

#define BLOCK_SIZE              1024
#define WARP_SIZE               32
#define PREFETCH_DIST           2

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

    __shared__ uint32_t shared_scratch[BLOCK_SIZE/WARP_SIZE * (MMA_M*MMA_N+1)];
    uint32_t* local_scratch = shared_scratch + warpId * (MMA_M*MMA_N+1);

    __shared__ uint64_t prefetch_x[PREFETCH_DIST * (BLOCK_SIZE+1)];
    __shared__ uint32_t prefetch_weights_compressed_two[PREFETCH_DIST * (BLOCK_SIZE+1)];

    int64_t TILES_M = GLOBAL_M / MMA_M;
    int64_t TILES_N = GLOBAL_N / MMA_N;
    int64_t TILES_K = GLOBAL_K / MMA_K;

    uint32_t A[4];
    uint32_t B[2];
    int32_t C[4] = {};

    for (int64_t blockPos = blockIdx.x; blockPos < TILES_M * TILES_N; blockPos += gridDim.x) {
        int64_t M_TILE = blockPos / TILES_N;
        int64_t N_TILE = blockPos % TILES_N;

        // fill the prefetch buffer
        for (int64_t cursor = 0; cursor < PREFETCH_DIST; cursor += 1) {
            int64_t K_TILE = warpId + cursor * BLOCK_SIZE/WARP_SIZE;

            int64_t bRow = M_TILE * MMA_M + (laneId/4);
            int64_t bCol = K_TILE * MMA_K + 8 * (laneId%4);
            __pipeline_memcpy_async(prefetch_x + cursor*(BLOCK_SIZE+1) + threadIdx.x, x + bRow*GLOBAL_K + bCol, 8);

            uint64_t offset = (N_TILE * TILES_K + K_TILE) * (MMA_N * MMA_K/8) + laneId * 2;
            __pipeline_memcpy_async(prefetch_weights_compressed_two + cursor*(BLOCK_SIZE+1) + threadIdx.x, weights_compressed + offset, 4);

            __pipeline_commit();
        }

        for (int64_t K_TILE = warpId; K_TILE < TILES_K; K_TILE += BLOCK_SIZE/WARP_SIZE) {
            int64_t cursor = K_TILE / (BLOCK_SIZE/WARP_SIZE) % PREFETCH_DIST;

            __pipeline_wait_prior(PREFETCH_DIST - 1);
            *reinterpret_cast<uint64_t *>(B) = prefetch_x[cursor * (BLOCK_SIZE+1) + threadIdx.x];
            uint32_t weight_compressed_two = prefetch_weights_compressed_two[cursor * (BLOCK_SIZE+1) + threadIdx.x];

            int64_t K_TILE_NEXT = K_TILE + PREFETCH_DIST * BLOCK_SIZE/WARP_SIZE;
            if (K_TILE_NEXT < TILES_K) {
                int64_t bRow = M_TILE * MMA_M + (laneId/4);
                int64_t bCol = K_TILE_NEXT * MMA_K + 8 * (laneId%4);
                __pipeline_memcpy_async(prefetch_x + cursor*(BLOCK_SIZE+1) + threadIdx.x, x + bRow*GLOBAL_K + bCol, 8);

                uint64_t offset = (N_TILE * TILES_K + K_TILE_NEXT) * (MMA_N * MMA_K/8) + laneId * 2;
                __pipeline_memcpy_async(prefetch_weights_compressed_two + cursor*(BLOCK_SIZE+1) + threadIdx.x, weights_compressed + offset, 4);

                __pipeline_commit();
            }

            uint64_t decoded1 = decode8weights(weight_compressed_two & 0xFFFF, codebook_abs);
            uint64_t decoded2 = decode8weights((weight_compressed_two >> 16) & 0xFFFF, codebook_abs);

            A[0] = decoded1;
            A[1] = decoded2;
            A[2] = decoded1 >> 32;
            A[3] = decoded2 >> 32;

            asm(
                "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"
                " { %0, %1, %2, %3 },"
                " { %4, %5, %6, %7 },"
                " { %8, %9 },"
                " { %10, %11, %12, %13 };\n"
                : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                  "r"(B[0]), "r"(B[1]),
                  "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3])
            );
        }

        local_scratch[laneId*4] = C[0];
        local_scratch[laneId*4+1] = C[1];
        local_scratch[laneId*4+2] = C[2];
        local_scratch[laneId*4+3] = C[3];

        __syncthreads();

        for (int64_t i = threadIdx.x; i < MMA_M*MMA_N; i += BLOCK_SIZE) {
            int32_t acc = 0;
            for (int64_t j = 0; j < BLOCK_SIZE/WARP_SIZE; j += 1) {
                acc += shared_scratch[j * (MMA_M*MMA_N+1) + i];
            }

            int64_t rowIdx[] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7};

            int64_t cRow = M_TILE * MMA_M + rowIdx[i%16];
            int64_t cCol = N_TILE * MMA_N + i/16 + (i%4/2) * 8;
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

    TORCH_CHECK(GLOBAL_M % MMA_M == 0, "GLOBAL_M is not divisible by MMA_M");
    TORCH_CHECK(GLOBAL_N % MMA_N == 0, "GLOBAL_N is not divisible by MMA_N");
    TORCH_CHECK(GLOBAL_K % (MMA_K * PREFETCH_DIST) == 0, "GLOBAL_K is not divisible by (MMA_K * PREFETCH_DIST)");

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
