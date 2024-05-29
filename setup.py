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
            name=f'micro_adam_sm{cc[0]}{cc[1]}',
            sources=[
                'setup_micro_adam/microadam.cpp',
                'setup_micro_adam/microadam_update.cu',
                'setup_micro_adam/microadam_tools.cu',
                'setup_micro_adam/microadam_symm_block_quant.cu',
                'setup_micro_adam/microadam_symm_block_quant_inv.cu',
                'setup_micro_adam/microadam_asymm_block_quant.cu',
                'setup_micro_adam/microadam_asymm_block_quant_inv.cu',
                'setup_micro_adam/microadam_asymm_global_quant.cu',
                'setup_micro_adam/microadam_asymm_global_quant_inv.cu',
            ]),
        CUDAExtension(
            name=f'sparse_mfac_sm{cc[0]}{cc[1]}',
            sources=[
                'setup_sparse_mfac/sparsemfac.cpp',
                'setup_sparse_mfac/sparsemfac_SP_kernel.cu',
                'setup_sparse_mfac/sparsemfac_LCG_kernel.cu',
                'setup_sparse_mfac/utils.cu'
            ]),

    ],
    cmdclass={'build_ext': BuildExtension}
)
