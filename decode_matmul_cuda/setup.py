import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME

ext_modules = []

if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'decode_matmul_cuda', [
            'decode_matmul_cuda.cu',
        ],
        extra_compile_args={'cxx': ['-O3'],
                            'nvcc': ['-O3', '-keep']})
    ext_modules.append(extension)

setup(
    name='decode_matmul_cuda',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
