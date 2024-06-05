from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch

cc = torch.cuda.get_device_capability(0)

setup(
    name='ista_daslab_optimizers',
    version='0.0.1',
    author='Ionut-Vlad Modoranu',
    author_email='ionut-vlad.modoranu@ist.ac.at',
    description='Deep Learning optimizers developed in the Distributed Algorithms and Systems group '
                '(DASLab) @ Institute of Science and Technology Austria (ISTA).',
    packages=find_packages(),
    url='https://github.com/IST-DASLab/ISTA-DASLab-Optimizers',
    ext_modules=[
        CUDAExtension(
            name=f'cuda_daslab_tools_sm{cc[0]}{cc[1]}',
            sources=[
                'kernels/cuda_daslab_tools/daslab_tools.cpp',
                'kernels/cuda_daslab_tools/daslab_tools.cu',
                # 'kernels/utils.cu',
            ],
            extra_link_flags=['--diag-suppress 20050', '--diag-suppress 816']),
        CUDAExtension(
            name=f'cuda_micro_adam_sm{cc[0]}{cc[1]}',
            sources=[
                'kernels/cuda_micro_adam/microadam.cpp',
                'kernels/cuda_micro_adam/microadam_update.cu',
                'kernels/cuda_micro_adam/microadam_asymm_block_quant.cu',
                'kernels/cuda_micro_adam/microadam_asymm_block_quant_inv.cu',
                # 'kernels/utils.cu',
            ]),
            # extra_compile_args=['-diag-suppress 20050']),
        # CUDAExtension(
        #     name=f'cuda_dense_mfac_sm{cc[0]}{cc[1]}',
        #     sources=[
        #         'kernels/cuda_dense_mfac/densemfac.cpp',
        #         'kernels/cuda_dense_mfac/densemfac_kernel.cu',
        #     ],
        #     extra_compile_args=['-diag-suppress 20050-D']),
        # CUDAExtension(
        #     name=f'cuda_sparse_mfac_sm{cc[0]}{cc[1]}',
        #     sources=[
        #         'kernels/cuda_sparse_mfac/sparsemfac.cpp',
        #         'kernels/cuda_sparse_mfac/sparsemfac_SP_kernel.cu',
        #         'kernels/cuda_sparse_mfac/sparsemfac_LCG_kernel.cu',
        #     ],
        #     extra_compile_args=['-diag-suppress 20050-D']),
    ],
    cmdclass={'build_ext': BuildExtension}
)
