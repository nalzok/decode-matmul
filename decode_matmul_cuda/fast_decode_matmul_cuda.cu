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

#define WMMA_M                  8
#define WMMA_N                  32
#define WMMA_K                  16

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


#define WARPS_PER_BLOCK 8
#define WEIGHTS_PER_UINT 16


__device__ uint64_t decode8weights(uint16_t weight_compressed,
        const uint64_t *__restrict__ codebook_abs) {

    uint16_t bits_abs = weight_compressed & ((1 << 8) - 1);
    bool bit_shift = (weight_compressed >> 15) & ((1 << 1) - 1);

    uint32_t bits_sign = (weight_compressed >> 7) & (((1 << 7) - 1) << 1);
    bits_sign |= __popc(bits_sign) & 1;
    uint64_t decoded_sign = bits_sign;
    decoded_sign |= (decoded_sign << (32-4));
    decoded_sign |= (decoded_sign << (16-2));
    decoded_sign |= (decoded_sign << (8-1));
    decoded_sign &= 0x0101010101010101;
    decoded_sign *= (255 - 3);

    // uint64_t packed = weight_compressed; packed = packed * packed; packed = packed * packed;
    uint64_t packed = codebook_abs[bits_abs] ^ decoded_sign;
    packed -= bit_shift * 0x0202020202020202;
    packed |= 0x0101010101010101;

    return packed;
}


__global__ static void
__launch_bounds__(256)
    fast_decode_matmul_kernel(
            int32_t *__restrict__ output,
            const int8_t *__restrict__ x,
            const uint32_t *__restrict__ weights_compressed,
            const uint64_t *__restrict__ codebook_abs,
            int64_t GLOBAL_M,
            int64_t GLOBAL_N,
            int64_t GLOBAL_K) {

        int64_t warpId = threadIdx.x / WARP_SIZE;
        int64_t laneId = threadIdx.x % WARP_SIZE;

        __shared__ uint64_t shared_scratch[WARPS_PER_BLOCK * WMMA_M * WMMA_N / 2];
        uint64_t* local_scratch = &shared_scratch[(WMMA_M * WMMA_N / 2) * warpId];

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag;

        for (int bid = blockIdx.x; bid < 4 * GLOBAL_N / WMMA_N; bid += gridDim.x) {
            int64_t blockId = bid / 4;
            int64_t tockId = bid % 4;

            for (int64_t i = threadIdx.x; i < (WMMA_M * WMMA_N / 2); i += BLOCK_SIZE) {
                local_scratch[i] = 0;
            }
            wmma::fill_fragment(acc_frag, 0);

            for (long r_pos = 0; r_pos < GLOBAL_K / (4 * WEIGHTS_PER_UINT * WARPS_PER_BLOCK); r_pos++) {
                long reduction_pos = r_pos + tockId * (GLOBAL_K / (4 * WEIGHTS_PER_UINT * WARPS_PER_BLOCK));

                // if (r_pos + 1 < GLOBAL_K / (4 * WEIGHTS_PER_UINT2 * WARPS_PER_BLOCK)) {
                //     pfL2_priority((uint64_t*)&weights_compressed[blockId * (GLOBAL_K/32) + (reduction_pos + 1) * WARP_SIZE * WARPS_PER_BLOCK + threadIdx.x]);
                // }
                uint32_t wrc_ui = weights_compressed[blockId * (GLOBAL_K/32) + reduction_pos * WARP_SIZE * WARPS_PER_BLOCK + threadIdx.x];

                long tile_pos = reduction_pos * 2 + (GLOBAL_K / WARPS_PER_BLOCK) * warpId;

                local_scratch[2*laneId+0] = decode8weights((uint16_t)wrc_ui, codebook_abs);
                local_scratch[2*laneId+1] = decode8weights((uint16_t)(wrc_ui >> 16), codebook_abs);
                wmma::load_matrix_sync(a_frag, x + tile_pos * WMMA_K, GLOBAL_K);
                wmma::load_matrix_sync(b_frag, (int8_t*)local_scratch, WMMA_K);
                // wmma::fill_fragment(a_frag, 1);
                // wmma::fill_fragment(b_frag, 1);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            // // store to local scratch
            wmma::store_matrix_sync((int32_t*)local_scratch, acc_frag, WMMA_N, wmma::mem_row_major);

            __syncthreads();

            int32_t acc = 0;
            for (long iacc = 0; iacc < WARPS_PER_BLOCK; iacc++) {
                acc += ((int32_t*)shared_scratch)[threadIdx.x + iacc * (WARPS_PER_BLOCK * WARP_SIZE)];
            }

            // output[warpId * GLOBAL_N + blockId * WARP_SIZE + laneId] = acc;
            atomicAdd(&output[warpId * GLOBAL_N + blockId * WARP_SIZE + laneId], acc);
        }
    }




__host__ torch::Tensor decode_matmul(
        torch::Tensor x,
        torch::Tensor weights_compressed,
        torch::Tensor codebook_abs,
        torch::Tensor output) {
    CHECK_INPUT(x);
    CHECK_INPUT(weights_compressed);
    CHECK_INPUT(codebook_abs);
    CHECK_INPUT(output);

    TORCH_CHECK(x.scalar_type() == torch::kInt8);
    TORCH_CHECK(weights_compressed.scalar_type() == torch::kInt16);
    TORCH_CHECK(codebook_abs.scalar_type() == torch::kInt64);
    TORCH_CHECK(x.size(-1) == weights_compressed.size(-1) << 3);
    TORCH_CHECK(output.scalar_type() == torch::kInt32);

    int64_t GLOBAL_M = x.size(-2);
    int64_t GLOBAL_N = weights_compressed.size(-2);
    int64_t GLOBAL_K = x.size(-1);

    TORCH_CHECK(GLOBAL_M == WMMA_M, "GLOBAL_M is not equal to WMMA_M");
    TORCH_CHECK(GLOBAL_N % WMMA_N == 0, "GLOBAL_N is not divisible by WMMA_N");
    TORCH_CHECK(GLOBAL_K % (4 * WEIGHTS_PER_UINT * WARPS_PER_BLOCK) == 0, "GLOBAL_K is not divisible by (WEIGHTS_PER_UINT * WARPS_PER_BLOCK)");

    TORCH_CHECK(output.size(0) == GLOBAL_M, "output.size(0) == GLOBAL_M");
    TORCH_CHECK(output.size(1) == GLOBAL_N, "output.size(0) == GLOBAL_N");

    at::DeviceGuard guard(x.device());

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, x.get_device());
    int64_t grid_size = 4 * GLOBAL_N / WMMA_N;
    int64_t block_size = WARP_SIZE * WARPS_PER_BLOCK;
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    fast_decode_matmul_kernel<<<grid_size, block_size, 0, stream>>>(
            output.data_ptr<int32_t>(),
            x.data_ptr<int8_t>(),
            (uint32_t*)weights_compressed.data_ptr<int16_t>(),
            (uint64_t*)codebook_abs.data_ptr<int64_t>(),
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
