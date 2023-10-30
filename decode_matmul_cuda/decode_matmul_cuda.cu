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

#define WARP_SIZE               32

#define SMEM_SIZE               (48 * 1024)


__host__ static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}


__global__ static void decode_matmul_kernel(
        int32_t *__restrict__ output,
        const int8_t *__restrict__ x,
        const int16_t *__restrict__ weights_compressed,
        const int64_t *__restrict__ codebook_abs,
        const int64_t *__restrict__ codebook_sign,
        int64_t GLOBAL_M,
        int64_t GLOBAL_N,
        int64_t GLOBAL_K) {

    int64_t warpId = threadIdx.x / WARP_SIZE;
    int64_t laneId = threadIdx.x % WARP_SIZE;

    extern __shared__ __align__(sizeof(int64_t)) char decoded_block_raw[];
    int64_t *decoded_block = reinterpret_cast<int64_t *>(decoded_block_raw);
    int64_t *decoded = decoded_block + warpId * WMMA_N * WMMA_K / 8;

    int64_t TILES_M = GLOBAL_M / WMMA_M;
    int64_t TILES_N = GLOBAL_N / WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag;

    for (int64_t block_pos = blockIdx.x; ; block_pos += gridDim.x) {
        int64_t warp_pos = block_pos / WARP_SIZE + warpId;
        int64_t WARP_M = warp_pos / TILES_N;
        int64_t WARP_N = warp_pos % TILES_N;

        if (WARP_M >= TILES_M) {
            break;
        }

        wmma::fill_fragment(acc_frag, 0);

        for (int64_t k = 0; k < GLOBAL_K; k += WMMA_K) {
            int64_t aRow = WARP_M * WMMA_M;
            int64_t aCol = k;

            wmma::load_matrix_sync(a_frag, x + aRow*GLOBAL_K + aCol, GLOBAL_K);

            int64_t bRow = WARP_N * WMMA_N;
            int64_t bCol = k;

            for (int64_t i = bRow; i < bRow + WMMA_N; i += 1) {
                for (int64_t j = bCol + 8 * laneId; j < bCol + WMMA_K; j += 8 * WARP_SIZE) {
                    int16_t weight_compressed = weights_compressed[i * (GLOBAL_K/8) + j/8];
                    int16_t bits_abs = weight_compressed & ((1 << 8) - 1);
                    int16_t bits_sign = (weight_compressed >> 8) & ((1 << 7) - 1);
                    bool bit_shift = (weight_compressed >> 15) & ((1 << 1) - 1);

                    int64_t packed = codebook_abs[bits_abs] ^ codebook_sign[bits_sign];
                    packed -= bit_shift * 0x0202020202020202;
                    packed |= 0x0101010101010101;

                    decoded[(i-bRow)*(WMMA_K/8) + (j-bCol)/8] = packed;   // little-endian
                }
            }

            wmma::load_matrix_sync(b_frag, reinterpret_cast<int8_t *>(decoded), WMMA_K);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        int64_t cRow = WARP_M * WMMA_M;
        int64_t cCol = WARP_N * WMMA_N;

        wmma::store_matrix_sync(output + cRow*GLOBAL_N + cCol, acc_frag, GLOBAL_N, wmma::mem_row_major);
    }
}


__host__ torch::Tensor decode_matmul(
        torch::Tensor x,
        torch::Tensor weights_compressed,
        torch::Tensor codebook_abs,
        torch::Tensor codebook_sign) {
    CHECK_INPUT(x);
    CHECK_INPUT(weights_compressed);
    CHECK_INPUT(codebook_abs);
    CHECK_INPUT(codebook_sign);

    TORCH_CHECK(x.scalar_type() == torch::kInt8);
    TORCH_CHECK(weights_compressed.scalar_type() == torch::kInt16);
    TORCH_CHECK(codebook_abs.scalar_type() == torch::kInt64);
    TORCH_CHECK(codebook_sign.scalar_type() == torch::kInt64);
    TORCH_CHECK(x.size(-1) == weights_compressed.size(-1) << 3);

    at::DeviceGuard guard(x.device());

    int64_t GLOBAL_M = x.size(-2);
    int64_t GLOBAL_N = weights_compressed.size(-2);
    int64_t GLOBAL_K = x.size(-1);
    torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    torch::Tensor output = torch::zeros(std::vector<int64_t>{GLOBAL_M, GLOBAL_N}, options);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, x.get_device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    decode_matmul_kernel<<<deviceProp.multiProcessorCount, 256, SMEM_SIZE, stream>>>(
            output.data_ptr<int32_t>(),
            x.data_ptr<int8_t>(),
            weights_compressed.data_ptr<int16_t>(),
            codebook_abs.data_ptr<int64_t>(),
            codebook_sign.data_ptr<int64_t>(),
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
