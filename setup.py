from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

"""
    How to add headers when building the project using `python3 -m build` (https://stackoverflow.com/a/6681343/22855002)
    - Add the relative path to MANIFEST.in file

    What didn't work:
    - headers parameter of setup function
    - include_dirs parameter of CUDAExtension
"""

def get_cuda_extension(name, sources):
    return CUDAExtension(
        name=name,
        sources=sources,
    )

setup(
    packages=find_packages(exclude=['examples']),
    # headers=['/nfs/scistore19/alistgrp/imodoran/workplace/ISTA-DASLab-Optimizers/kernels/utils.h'],
    ext_modules=[
        get_cuda_extension(
            name=f'ista_daslab_tools',
            sources=[
                './kernels/tools/tools.cpp',
                './kernels/tools/tools_kernel.cu'
            ],
        ),
        get_cuda_extension(
            name=f'ista_daslab_dense_mfac',
            sources=[
                './kernels/dense_mfac/dense_mfac.cpp',
                './kernels/dense_mfac/dense_mfac_kernel.cu',
            ],
        ),
        get_cuda_extension(
            name=f'ista_daslab_sparse_mfac',
            sources=[
                './kernels/sparse_mfac/sparse_mfac.cpp',
                './kernels/sparse_mfac/sparse_mfac_SP_kernel.cu',
                './kernels/sparse_mfac/sparse_mfac_LCG_kernel.cu',
            ],
        ),
        get_cuda_extension(
            name=f'ista_daslab_micro_adam',
            sources=[
                './kernels/micro_adam/micro_adam.cpp',
                './kernels/micro_adam/micro_adam_update.cu',
                './kernels/micro_adam/micro_adam_asymm_block_quant.cu',
                './kernels/micro_adam/micro_adam_asymm_block_quant_inv.cu',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension.with_options(verbose=True)},
)
