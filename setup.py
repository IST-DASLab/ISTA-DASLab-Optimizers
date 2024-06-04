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
            ]),
        CUDAExtension(
            name=f'cuda_micro_adam_sm{cc[0]}{cc[1]}',
            sources=[
                'kernels/cuda_micro_adam/microadam.cpp',
                'kernels/cuda_micro_adam/microadam_update.cu',
                'kernels/cuda_micro_adam/microadam_symm_block_quant.cu',
                'kernels/cuda_micro_adam/microadam_symm_block_quant_inv.cu',
                'kernels/cuda_micro_adam/microadam_asymm_block_quant.cu',
                'kernels/cuda_micro_adam/microadam_asymm_block_quant_inv.cu',
                'kernels/cuda_micro_adam/microadam_asymm_global_quant.cu',
                'kernels/cuda_micro_adam/microadam_asymm_global_quant_inv.cu',
            ]),
        CUDAExtension(
            name=f'cuda_sparse_mfac_sm{cc[0]}{cc[1]}',
            sources=[
                'kernels/cuda_sparse_mfac/sparsemfac.cpp',
                'kernels/cuda_sparse_mfac/sparsemfac_SP_kernel.cu',
                'kernels/cuda_sparse_mfac/sparsemfac_LCG_kernel.cu',
            ]),


    ],
    cmdclass={'build_ext': BuildExtension}
)
