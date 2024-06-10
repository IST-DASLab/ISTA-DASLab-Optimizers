from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='ista_daslab_optimizers',
    version='0.0.1',
    author='Ionut-Vlad Modoranu',
    author_email='ionut-vlad.modoranu@ist.ac.at',
    description='Deep Learning optimizers developed in the Distributed Algorithms and Systems group (DASLab) @ Institute of Science and Technology Austria (ISTA)',
    url='https://github.com/IST-DASLab/ISTA-DASLab-Optimizers',
    
    packages=find_packages(),
    cmdclass={'build_ext': BuildExtension},

    ext_modules=[
        CUDAExtension(
            name=f'ista_daslab_tools',
            sources=[
                'kernels/tools/tools.cpp',
                'kernels/tools/tools_kernel.cu',
            ]),
        CUDAExtension(
            name=f'ista_daslab_dense_mfac',
            sources=[
                'kernels/dense_mfac/dense_mfac.cpp',
                'kernels/dense_mfac/dense_mfac_kernel.cu',
            ]),
        CUDAExtension(
            name=f'ista_daslab_sparse_mfac',
            sources=[
                'kernels/sparse_mfac/sparse_mfac.cpp',
                'kernels/sparse_mfac/sparse_mfac_SP_kernel.cu',
                'kernels/sparse_mfac/sparse_mfac_LCG_kernel.cu',
            ]),
        CUDAExtension(
            name=f'ista_daslab_micro_adam',
            sources=[
                'kernels/micro_adam/micro_adam.cpp',
                'kernels/micro_adam/micro_adam_update.cu',
                'kernels/micro_adam/micro_adam_asymm_block_quant.cu',
                'kernels/micro_adam/micro_adam_asymm_block_quant_inv.cu',
            ]),
    ])
