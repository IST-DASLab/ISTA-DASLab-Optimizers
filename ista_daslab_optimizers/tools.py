import os
import gpustat
import torch
from enum import Enum
from importlib import import_module
import ista_daslab_tools

def get_cuda_capability(device=0):
    cc = torch.cuda.get_device_capability(device) # tuple, for example (8, 6) for CUDA Capability 8.6
    return f'{cc[0]}{cc[1]}'

class CopyDirection(Enum):
    k2d = 0
    d2k = 1

class Strategy(Enum):
    """Apply Top-K globally"""
    GLOBAL = 1

    """Apply Top-K in blocks of specific size"""
    BLOCK = 2

    @staticmethod
    def factory(name: str):
        if name == 'gl': return Strategy.GLOBAL
        if name == 'bl': return Strategy.BLOCK
        raise RuntimeError('Invalid strategy name')

def get_first_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if torch.distributed.is_initialized():
        return torch.device(f'cuda:{torch.distributed.get_rank()}')
    return torch.device('cuda:0')


def get_gpus():
    if not torch.cuda.is_available():
        return ['cpu']
    device = get_first_device()
    if torch.cuda.device_count() == 1:
        return [device]

    return [
        torch.device(f'cuda:{i}')
        for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    ]

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

def get_weights_and_gradients(params, get_weights, get_grad=True, grad_bf16=False):
    """
        This method returns:
        - w: the raw weights collected from the model if get_weights=True
        - g: the gradients (without WD added)
    """
    w, g = [], []
    for group in params:
        for p in group['params']:
            if p.grad is None or not p.requires_grad:
                continue

            if get_weights:
                w.append(p.reshape(-1))
            if get_grad:
                if grad_bf16:
                    if p.grad.dtype != torch.bfloat16:
                        g.append(p.grad.reshape(-1).to(dtype=torch.bfloat16))
                    else:
                        g.append(p.grad.reshape(-1))
                else:
                    g.append(p.grad.reshape(-1))

    if get_weights and get_grad:
        return torch.cat(w), torch.cat(g)
    if get_weights:
        return torch.cat(w)
    if get_grad:
        return torch.cat(g)
    raise RuntimeError(f'invalid combination of parameters: {get_weights=}, {get_grad=}')


def update_model(params, update, weight_decay=0, alpha=None, multiply_wd_w_lr=False):
    """
        Applies the `update` to the model
        When alpha=None, alpha is set to lr in the group
        Returns the shrinking factor for the weights
    """
    count = 0
    for group in params:
        lr = group['lr']
        wd = group.get('weight_decay', weight_decay) # if the param groups do not have weight decay, then use the externally provided one
        for p in group['params']:
            u = update[count:(count + p.numel())].reshape(p.shape).to(p.device)
            if wd > 0:
                if multiply_wd_w_lr:
                    p.mul_(1 - lr * wd)
                else:
                    p.mul_(1 - wd)
            p.add_(u, alpha=-lr if alpha is None else alpha)
            count += p.numel()

class KernelVersionsManager:
    def __init__(self, version_SP, version_LCG, m, d, d_block_size):
        self.version_SP = version_SP
        self.version_LCG = version_LCG
        self.m = m
        self.d = d
        self.d_block_size = d_block_size

        self.BLOCK_INDEX = 0
        self.THREAD_INDEX = 1

        # set number of blocks (initially None) based on the number of threads (see page 80 in the PhD #8)
        # if self.d > 300_000_000:
        #     print(f'Model size is larger than 300M. Switching SP version from {self.version_SP} to 252')
        #     self.version_SP = 252

        self.SP_BLOCKS_THREADS = {
            23: [self.m, self.m],
            # 24: [1024, 1024],
            # 251: [None, 1024],
            # 252: [None, self.m],
            # 261: [None, 128],
            # 262: [None, 128],
            # 272: [None, 1024],
        }

        self.LCG_BLOCKS_THREADS = {
            # 42: [68, 256],
            # 43: [117, 32],
            51: [None, 1024],
            # 524: [None, 128],
            # 53: [None, 128],
            # 54: [None, 128],
        }

        self.set_blocks_count(self.SP_BLOCKS_THREADS, self.version_SP, op='SP')
        self.set_blocks_count(self.LCG_BLOCKS_THREADS, self.version_LCG, op='LCG')
        # self.SP_BLOCKS_THREADS[self.version_SP][0] = 10

    def set_blocks_count(self, op_blocks_threads, op_version, op):
        """
        Safety measure: for small models, there might be too many thread blocks launched and most of them will process data out of bounds of arrays out, indices and values
        """
        def div_inc(a, b):
            r = a // b
            return (r + 1) if (a % b > 0) else r

        if op_blocks_threads[op_version][self.BLOCK_INDEX] is None:
            blocks_count = div_inc(self.d, self.d_block_size)
            op_max_blocks = ista_daslab_tools.get_sm_count()
            op_required_blocks = min(blocks_count, op_max_blocks)
            if op_required_blocks < op_max_blocks:
                print(f'Maximum number of blocks for {op} is {op_max_blocks}, but this model requires only {op_required_blocks}')
                # return op_required_blocks
                op_blocks_threads[op_version][self.BLOCK_INDEX] = op_required_blocks
            op_blocks_threads[op_version][self.BLOCK_INDEX] = op_max_blocks

        print(f'{op_blocks_threads=}, {op_version=}, {op=}, {op_blocks_threads[op_version][self.BLOCK_INDEX]=}')

    def get_SP_blocks(self):
        return self.SP_BLOCKS_THREADS[self.version_SP][self.BLOCK_INDEX]

    def get_SP_threads(self):
        return self.SP_BLOCKS_THREADS[self.version_SP][self.THREAD_INDEX]

    def get_LCG_blocks(self):
        return self.LCG_BLOCKS_THREADS[self.version_LCG][self.BLOCK_INDEX]

    def get_LCG_threads(self):
        return self.LCG_BLOCKS_THREADS[self.version_LCG][self.THREAD_INDEX]