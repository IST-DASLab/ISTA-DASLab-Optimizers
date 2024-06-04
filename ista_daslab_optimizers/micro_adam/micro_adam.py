
import torch
import math
import time
import wandb
from torch.distributed import is_initialized, get_rank
from ..tools import get_first_device, get_gpu_mem_usage, block_split, import_cuda_module

cuda_micro_adam = import_cuda_module('cuda_micro_adam')

class MicroAdam(torch.optim.Optimizer):
    def __init__(self, params, m, lr, quant_block_size, k_init=0.01, betas=(0.9, 0.999), weight_decay=0, eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, eps=eps)
        super(MicroAdam, self).__init__(params, defaults)

        self.m = m
        self.lr = lr
        self.quant_block_size = int(quant_block_size)
        self.k_init = k_init
        self.weight_decay = weight_decay
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

        self.model_size = sum([p.numel() for group in self.param_groups for p in group['params']])

        self.steps = 0  # how many optimization steps were performed so far
        self.log_interval = 100
        self.device = get_first_device()
        self._is_state_initialized = False
        self.shared_memory_carveout = 100
        self.blocks = cuda_micro_adam.get_sm_count() * int(100 / self.shared_memory_carveout)
        self.threads = 512

        self.dict_size_count = {}  # key = layer size, value = how many layers of that size the model has
        for param in self.param_groups:
            for p in param['params']:
                size = p.numel()
                self.dict_size_count[size] = 1 + self.dict_size_count.get(size, 0)

        self._init_state()

    def _init_state(self):
        max_floats = cuda_micro_adam.get_max_floats_for_shared_memory_per_thread_block()
        d_block_size = max_floats // 2 // int(100 / self.shared_memory_carveout)
        count = 0
        for group in self.param_groups:
            lr = group['lr']
            wd = group.get('weight_decay', self.weight_decay) # if the param groups do not have weight decay, then use the external one
            for p in group['params']:
                if not p.requires_grad:
                    continue
                count += 1
                layer_size = p.numel()
                st = self.state[p]

                # B * t / d * nt
                st['blocks'] = max(1, int(math.floor(self.blocks * layer_size * self.dict_size_count[layer_size] / self.model_size)))

                st['lr'] = lr
                st['weight_decay'] = wd
                st['d'] = layer_size

                ##### variables for Top-K: d_index_topk is the index where the last, smaller topk block starts
                st['d_block_size'] = layer_size if layer_size < d_block_size else d_block_size
                st['topk_full_blocks_count'], st['d_index_topk'] = block_split(st['d'], st['d_block_size'])
                st['k_block_size_many'] = int(math.ceil(st['d_block_size'] * self.k_init))
                st['k_block_size_few'] = int(math.ceil((st['d'] - st['d_index_topk']) * self.k_init))  # 0 for d % d_block_size = 0
                st['k_index'] = st['topk_full_blocks_count'] * st['k_block_size_many']
                st['k'] = st['k_block_size_many'] * st['topk_full_blocks_count'] + st['k_block_size_few']

                ##### variables for the ring buffer
                st['index'] = 0  # the position to place a new gradient at
                st['I'] = torch.zeros(self.m, st['k'], dtype=torch.int16, device=self.device)  # 2mk bytes
                st['V'] = torch.zeros(self.m, st['k'], dtype=torch.bfloat16, device=self.device)  # 2mk bytes

                ### variables for error feedback: d_index_quant is the index where the last, smaller quantization block starts
                # st['quant_block_size'] = layer_size if layer_size < self.quant_block_size else self.quant_block_size
                st['quant_full_blocks_count'], st['d_index_quant'] = block_split(st['d'], self.quant_block_size)
                st['error'] = torch.zeros(int(math.ceil(st['d'] / 2)), dtype=torch.uint8, device=self.device)  # ceil(d/2) bytes
                st['min_vals'] = torch.zeros(st['quant_full_blocks_count'] + 1, dtype=torch.bfloat16, device=self.device)  # ceil(d/q_bsz)*2 bytes
                st['max_vals'] = torch.zeros(st['quant_full_blocks_count'] + 1, dtype=torch.bfloat16, device=self.device)  # ceil(d/q_bsz)*2 bytes

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1

        self._update_lr_wd()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        time_start = time.time()

        norm_g, norm_u, norm_e, sparsity_u = 0, 0, 0, 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                ng, nu, ne, sp_u = self.update_step(p)
                norm_g += ng
                norm_u += nu
                norm_e += ne
                sparsity_u += sp_u

        # torch.cuda.synchronize()
        time_end = time.time()
        elapsed_step = time_end - time_start
        self._log(norm_g, norm_u, norm_e, sparsity_u, elapsed_step)

        return loss

    @torch.no_grad()
    def update_step(self, p):
        norm_g, norm_u, norm_e, sp_u = 0, 0, 0, 0

        st = self.state[p]
        grad = p.grad.view(-1)

        if self.steps % self.log_interval == 0:
            norm_g = grad.norm(p=2) ** 2

        blocks = st['blocks']
        lr = st['lr']
        wd = st['weight_decay']
        d = st['d']
        d_block_size = st['d_block_size']
        topk_full_blocks_count, d_index_topk = st['topk_full_blocks_count'], st['d_index_topk']
        k_block_size_many = st['k_block_size_many']
        k_block_size_few = st['k_block_size_few']
        k_index = st['k_index']
        k = st['k']

        # HuggingFace has a setting that converts st['I'] to bfloat16, even though it is declared as int16
        # This happens somewhere between constructor call and step call. Converting it to int16 was the simplest solution
        if st['I'].dtype != torch.int16:
            st['I'] = st['I'].to(torch.int16)

        index = st['index']
        I = st['I']
        V = st['V']

        quant_full_blocks_count, d_index_quant = st['quant_full_blocks_count'], st['d_index_quant']
        error = st['error']
        min_vals = st['min_vals']
        max_vals = st['max_vals']

        ##### STEP 4
        cuda_micro_adam.asymm_block_quant_inv(d, self.quant_block_size, error, min_vals, max_vals, grad)

        ##### STEP 5 + 9 (only for I)
        I[index, :k_index] = torch.topk(input=grad[0:d_index_topk].abs().view(topk_full_blocks_count, d_block_size),
                                        k=k_block_size_many,
                                        sorted=False).indices.to(dtype=torch.int16).view(-1)

        if k_block_size_few > 0:  # there is a small block left
            I[index, k_index:] = torch.topk(input=grad[d_index_topk:].abs(),
                                            k=k_block_size_few,  # example: slice has size 1, but ks[-1] is 4
                                            sorted=False).indices.to(dtype=torch.int16).view(-1)

        cuda_micro_adam.copy_values_large_to_small(d,
                                                   k,
                                                   d_block_size,
                                                   k_block_size_many,
                                                   I[index, :],
                                                   grad,
                                                   V[index, :], )  # this does V[index,:] = a[I[index]]
        st['index'] = (index + 1) % self.m

        ##### STEP 6
        cuda_micro_adam.zerorize_block_components(grad, I[index, :], d, k, d_block_size, k_block_size_many)  # this does a[I[index]] = 0

        ##### STEP 7
        if quant_full_blocks_count == 1:
            min_vals[:quant_full_blocks_count] = grad[:d_index_quant].min()
            max_vals[:quant_full_blocks_count] = grad[:d_index_quant].max()
        else:
            min_vals[:quant_full_blocks_count] = grad[:d_index_quant].view(quant_full_blocks_count, self.quant_block_size).min(dim=1).values
            max_vals[:quant_full_blocks_count] = grad[:d_index_quant].view(quant_full_blocks_count, self.quant_block_size).max(dim=1).values
        if d_index_quant < d:
            min_vals[quant_full_blocks_count] = grad[d_index_quant:].min()
            max_vals[quant_full_blocks_count] = grad[d_index_quant:].max()

        ##### STEP 8
        cuda_micro_adam.asymm_block_quant(d, self.quant_block_size, error, min_vals, max_vals, grad) # error = Q(a, min, max)

        ##### STEPS 10-11
        grad.zero_()
        cuda_micro_adam.compute_cadam_update(blocks,  # blocks
                                        self.threads,  # threads
                                        self.shared_memory_carveout,  # carveout
                                        self.steps,  # optimization step
                                        self.beta1,  # beta1
                                        self.beta2,  # beta2
                                        self.eps,  # eps
                                        d_block_size,  # d_block_size
                                        k_block_size_many,  # k_block_size
                                        d,  # d
                                        self.m,  # m
                                        k,  # k
                                        I,  # indices
                                        V,  # values
                                        grad)  # update will be stored here

        ##### STEP 12
        p.mul_(1 - lr * wd).add_(p.grad, alpha=-lr)

        # compute error norm
        if self.steps % self.log_interval == 0:
            norm_u = grad.norm(p=2) ** 2
            sp_u = (grad == 0).sum() # check sparsity before zerorizing

            grad.zero_()
            cuda_micro_adam.asymm_block_quant_inv(d, self.quant_block_size, error, min_vals, max_vals, grad)

            norm_e = grad.norm(p=2) ** 2

        return norm_g, norm_u, norm_e, sp_u

    def _log(self, norm_g, norm_u, norm_e, sparsity_u, elapsed_step):
        if self.steps % self.log_interval == 0:
            wandb_data = {
                'step/optimizer_steps': self.steps,
                'step/gpu_mem_usage': get_gpu_mem_usage(),
                'step/norm_g': math.sqrt(norm_g),
                'step/norm_u': math.sqrt(norm_u),
                'step/norm_error': math.sqrt(norm_e),
                'step/sparsity_u': sparsity_u / self.model_size * 100.,
                'step/elapsed_step': elapsed_step,
            }

            if not is_initialized() or get_rank() == 0:
                wandb.log(wandb_data, commit=False)

    def _update_lr_wd(self):
        # copy the learning rate group to parameter state because the lr scheduler updates the one in the group
        for group in self.param_groups:
            lr = group['lr']
            wd = group.get('weight_decay', self.weight_decay)  # if the param groups do not have weight decay, then use the external one
            for p in group['params']:
                self.state[p]['lr'] = lr
                self.state[p]['wd'] = wd
