#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "decode_matmul_cuda.fatbin.c"
static void __device_stub__Z20decode_matmul_kernelPiPKaPKsPKllll(int32_t *__restrict__, const int8_t *__restrict__, const int16_t *__restrict__, const int64_t *__restrict__, int64_t, int64_t, int64_t);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
static void __device_stub__Z20decode_matmul_kernelPiPKaPKsPKllll(int32_t *__restrict__ __par0, const int8_t *__restrict__ __par1, const int16_t *__restrict__ __par2, const int64_t *__restrict__ __par3, int64_t __par4, int64_t __par5, int64_t __par6){ int32_t *__T115;
 const int8_t *__T116;
 const int16_t *__T117;
 const int64_t *__T118;
__cudaLaunchPrologue(7);__T115 = __par0;__cudaSetupArgSimple(__T115, 0UL);__T116 = __par1;__cudaSetupArgSimple(__T116, 8UL);__T117 = __par2;__cudaSetupArgSimple(__T117, 16UL);__T118 = __par3;__cudaSetupArgSimple(__T118, 24UL);__cudaSetupArgSimple(__par4, 32UL);__cudaSetupArgSimple(__par5, 40UL);__cudaSetupArgSimple(__par6, 48UL);__cudaLaunch(((char *)((void ( *)(int32_t *__restrict__, const int8_t *__restrict__, const int16_t *__restrict__, const int64_t *__restrict__, int64_t, int64_t, int64_t))decode_matmul_kernel)));}
# 67 "decode_matmul_cuda.cu"
static void decode_matmul_kernel( int32_t *__restrict__ __cuda_0,const int8_t *__restrict__ __cuda_1,const int16_t *__restrict__ __cuda_2,const int64_t *__restrict__ __cuda_3,int64_t __cuda_4,int64_t __cuda_5,int64_t __cuda_6)
# 74 "decode_matmul_cuda.cu"
{__device_stub__Z20decode_matmul_kernelPiPKaPKsPKllll( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 168 "decode_matmul_cuda.cu"
}
# 1 "decode_matmul_cuda.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T418) {  __nv_dummy_param_ref(__T418); __nv_save_fatbinhandle_for_managed_rt(__T418); __cudaRegisterEntry(__T418, ((void ( *)(int32_t *__restrict__, const int8_t *__restrict__, const int16_t *__restrict__, const int64_t *__restrict__, int64_t, int64_t, int64_t))decode_matmul_kernel), _Z20decode_matmul_kernelPiPKaPKsPKllll, 1024); __cudaRegisterVariable(__T418, __shadow_var(_ZN52_INTERNAL_a3f58b29_21_decode_matmul_cuda_cu_2b51f7716thrust6system6detail10sequential3seqE,::thrust::system::detail::sequential::seq), 0, 1UL, 0, 0); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
