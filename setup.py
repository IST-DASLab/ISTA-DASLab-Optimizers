from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import pybind11
import torch
import os

cc = torch.cuda.get_device_capability(0)
cc = f'{cc[0]}{cc[1]}'
CUDA_HOME = os.environ['CUDA_HOME']

def get_cuda_extension(name, sources):
    return CUDAExtension(
        # name=name,
        name=f'ista_daslab_optimizers.{name}',
        sources=sources,
        # dlink=True,
        # dlink_libraries=['dlink_lib'],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                f'-arch=sm_{cc}',
                '-lcublas',
                '-lcudart',
                '-O3',
                # f'--default-stream per-thread',
                f'--output-directory=build/temp.linux-x86_64-cpython-39/{name}/'
            ],
            'nvcclink': [
                f'-arch=sm_{cc}',
                '--device-link'
            ]
        },
        include_dirs=[os.path.abspath('./kernels'), pybind11.get_include()],
        library_dirs=[os.path.join(CUDA_HOME, 'lib64')],
        # libraries=['cusparse', 'cublas'],
    )

setup(
    name='ista_daslab_optimizers',
    version='0.0.1',
    author='Ionut-Vlad Modoranu',
    author_email='ionut-vlad.modoranu@ist.ac.at',
    description='Deep Learning optimizers developed in the Distributed Algorithms and Systems group '
                '(DASLab) @ Institute of Science and Technology Austria (ISTA).',
    url='https://github.com/IST-DASLab/ISTA-DASLab-Optimizers',

    packages=find_packages(),
    # install_requires=['torch', 'pybind11', 'ninja'],
    cmdclass={'build_ext': BuildExtension},

    ext_modules=[
        get_cuda_extension(
            name=f'cuda_daslab_tools_sm{cc}',
            sources=[
                'kernels/cuda_daslab_tools/daslab_tools.cpp',
                'kernels/cuda_daslab_tools/daslab_tools.cu',
            ]),
        # get_cuda_extension(
        #     name=f'cuda_micro_adam_sm{cc}',
        #     sources=[
        #         'kernels/cuda_micro_adam/microadam.cpp',
        #         'kernels/cuda_micro_adam/microadam_update.cu',
        #         'kernels/cuda_micro_adam/microadam_asymm_block_quant.cu',
        #         'kernels/cuda_micro_adam/microadam_asymm_block_quant_inv.cu',
        #     ]),
        # get_cuda_extension(
        #     name=f'cuda_sparse_mfac_sm{cc}',
        #     sources=[
        #         'kernels/cuda_sparse_mfac/sparsemfac.cpp',
        #         'kernels/cuda_sparse_mfac/sparsemfac_SP_kernel.cu',
        #         'kernels/cuda_sparse_mfac/sparsemfac_LCG_kernel.cu',
        #     ]),
        # get_cuda_extension(
        #     name=f'cuda_dense_mfac_sm{cc}',
        #     sources=[
        #         'kernels/cuda_dense_mfac/densemfac.cpp',
        #         'kernels/cuda_dense_mfac/densemfac_kernel.cu',
        #     ]),
    ]
)

# extra_link_flags=['--diag-suppress 20050', '--diag-suppress 816']),
# include_dirs = [pybind11.get_include()],
# language = 'c++',
# extra_compile_args = ['-std=c++17', '-lcudart', '-lcuda'],
# extra_link_args = [f'-L{CUDA_HOME}']
# install_requires=['pybind11'],
