import os
import gpustat
import torch
from importlib import import_module

def get_first_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if torch.distributed.is_initialized():
        return torch.device(f'cuda:{torch.distributed.get_rank()}')
    return torch.device('cuda:0')

def get_cuda_capability(device=0):
    cc = torch.cuda.get_device_capability(device) # tuple, for example (8, 6) for CUDA Capability 8.6
    return f'{cc[0]}{cc[1]}'

def import_cuda_module(name):
    """
    Import a CUDA module based on the name, assuming that the module was installed for a specific cuda capability.
    For example, if the package 'my_cuda' was installed for CUDA capability 8.0, then the imported module will be 'my_cuda_sm80'.
    :param name: the module to be imported
    :return:
    """
    try:
        cc = get_cuda_capability()
        module = import_module(f'{name}_sm{cc}')
        return module
    except ModuleNotFoundError as e:
        raise RuntimeError(f'The library "{name}" was not compiled for sm{cc}!')


def get_gpu_mem_usage():
    """
        This method returns the GPU memory usage for the current process.
        It uses gpustat to query the GPU used by the current process (using CUDA_VISIBLE_DEVICES)

        GPUSTAT usage:
        stat = gpustat.new_query().gpus # this is a list containing information about each GPU indexed from 0 to 7
        stat[i] (GPU #i) has the following keys:
            - 'index'
            - 'uuid'
            - 'name'
            - 'temperature.gpu'
            - 'fan.speed'
            - 'utilization.gpu'
            - 'utilization.enc'
            - 'utilization.dec'
            - 'power.draw'
            - 'enforced.power.limit'
            - 'memory.used'
            - 'memory.total'
            - 'processes'
        Among these keys, only the key 'processes' is used here.
        stat[i].processes is a list of dicts, where each dict contains information about each process currently running on the GPU #i
            - 'username'
            - 'command'
            - 'full_command'
            - 'gpu_memory_usage'
            - 'cpu_percent'
            - 'cpu_memory_usage'
            - 'pid'
    """
    gpus = gpustat.new_query().gpus
    gids = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    gpu_mem = sum([int(proc['gpu_memory_usage']) for gid in gids for proc in gpus[gid]['processes'] if int(proc['pid']) == os.getpid()])
    return gpu_mem

def block_split(model_size, block_size):
    if model_size < block_size:
        return 1, model_size
    ### this is the shorter version that only returns the number of full blocks of size "block_size"
    ### and the starting position of the last and smallest block
    blocks_count = int(model_size / block_size)
    start_index_last_block = model_size - model_size % block_size
    return blocks_count, start_index_last_block
