import os
import torch
import math
import time
import wandb
from torch.distributed import is_initialized, get_rank, all_reduce, ReduceOp
from ..tools import get_first_device, get_gpu_mem_usage, block_split, CopyDirection

import ista_daslab_tools
import ista_daslab_micro_adam


class MicroAdam(torch.optim.Optimizer):
    def __init__(self, params, m, lr, quant_block_size, k_init=0.01, alpha=0, betas=(0.9, 0.999), weight_decay=0, eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, eps=eps, alpha=alpha)
        super(MicroAdam, self).__init__(params, defaults)

        assert (0 <= alpha < 1) or alpha == -2, 'Alpha must be in the [0, 1) interval or -2'

        self.m = m
        self.lr = lr
        self.quant_block_size = int(quant_block_size)
        self.k_init = k_init
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps

        self.densify_update_using_ef = (self.alpha > 0)
        self.densify_update_using_quant_error = (self.alpha == -2)

        self.model_size = sum([p.numel() for group in self.param_groups for p in group['params']])

        self.steps = 0  # how many optimization steps were performed so far
        self.log_interval = 100
        self.device = get_first_device()
        self._is_state_initialized = False
        self.shared_memory_carveout = 100
        self.blocks = ista_daslab_tools.get_sm_count() * int(100 / self.shared_memory_carveout)
        self.threads = 512

        self.max_floats = ista_daslab_tools.get_max_floats_for_shared_memory_per_thread_block()
        self.d_block_size = self.max_floats // 2 // int(100 / self.shared_memory_carveout)

        if torch.distributed.is_initialized():
            self.fsdp_dict_size_count = [{} for _ in range(
                torch.distributed.get_world_size())]  # key = layer size, value = how many layers of that size the model has (per worker)
        else:
            self.fsdp_dict_size_count = [{}]

        self.dict_size_count = {}  # key = layer size, value = how many layers of that size the model has
        for param in self.param_groups:
            for p in param['params']:
                size = p.numel()
                # print(p.shape, p.numel())
                self.dict_size_count[size] = 1 + self.dict_size_count.get(size, 0)

        # self._init_state()

    def _initialize_parameter_state(self, p, lr, wd):
        layer_size = p.numel()
        st = self.state[p]

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if self.densify_update_using_quant_error:
            st['quant_err'] = torch.zeros_like(p)

        st['blocks'] = max(1, int(math.floor(self.blocks * layer_size * self.fsdp_dict_size_count[rank][layer_size] / self.model_size)))

        st['lr'] = lr
        st['weight_decay'] = wd
        st['d'] = layer_size

        ##### variables for Top-K: d_index_topk is the index where the last, smaller topk block starts
        st['d_block_size'] = layer_size if layer_size < self.d_block_size else self.d_block_size
        st['topk_full_blocks_count'], st['d_index_topk'] = block_split(st['d'], st['d_block_size'])
        st['k_block_size_many'] = int(math.ceil(st['d_block_size'] * self.k_init))
        st['k_block_size_few'] = int(math.ceil((st['d'] - st['d_index_topk']) * self.k_init))  # 0 for d % self.d_block_size = 0
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

        # self._update_lr_wd()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.steps == 1:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            for param in self.param_groups:
                for p in param['params']:
                    if p is not None:
                        size = p.numel()
                        if size > 0:
                            self.fsdp_dict_size_count[rank][size] = 1 + self.fsdp_dict_size_count[rank].get(size, 0)

        time_start = time.time()

        norm_qe, norm_g, norm_u, norm_e, sparsity_u, sparsity_qe = 0, 0, 0, 0, 0, 0

        for group in self.param_groups:
            lr = group['lr']
            wd = group.get('weight_decay', self.weight_decay)

            for p in group['params']:
                if p.grad is None:
                    continue

                if p is None:
                    continue

                nqe, ng, nu, ne, sp_u, sp_qe = self.update_step(p, lr, wd)
                norm_qe += nqe
                norm_g += ng
                norm_u += nu
                norm_e += ne
                sparsity_u += sp_u
                sparsity_qe += sp_qe

        # torch.cuda.synchronize()
        time_end = time.time()
        elapsed_step = time_end - time_start
        self._log(norm_qe, norm_g, norm_u, norm_e, sparsity_u, sparsity_qe, elapsed_step)

        return loss

    @torch.no_grad()
    def update_step(self, p, lr, wd):
        norm_qe, norm_g, norm_u, norm_e, sp_u, sp_qe = 0, 0, 0, 0, 0, 0

        # if p.grad.dtype != torch.bfloat16:
        #     grad = p.grad.to(dtype=torch.bfloat16).reshape(-1)
        # else:
        grad = p.grad.view(-1)

        if self.steps % self.log_interval == 0:
            norm_g = grad.norm(p=2) ** 2

        st = self.state[p]
        if len(st) == 0:
            self._initialize_parameter_state(p, lr, wd)

        # print('rank=',torch.distributed.get_rank(), 'keys=',st.keys())

        blocks = st['blocks']
        # lr = st['lr']
        # wd = st['weight_decay']
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
        ista_daslab_micro_adam.asymm_block_quant_inv(d, self.quant_block_size, error, min_vals, max_vals, grad, 1.0) # alpha=1 here

        ##### STEP 5 + 9 (only for I)
        I[index, :k_index] = torch.topk(input=grad[0:d_index_topk].abs().view(topk_full_blocks_count, d_block_size),
                                        k=k_block_size_many,
                                        sorted=False).indices.to(dtype=torch.int16).view(-1)

        if k_block_size_few > 0:  # there is a small block left
            I[index, k_index:] = torch.topk(input=grad[d_index_topk:].abs(),
                                            k=k_block_size_few,  # example: slice has size 1, but ks[-1] is 4
                                            sorted=False).indices.to(dtype=torch.int16).view(-1)

        ista_daslab_tools.copy_values(d,  # V = error[I[buffer_index, :]]
                                      k,
                                      d_block_size,
                                      k_block_size_many,
                                      I[index, :],  # indices
                                      grad,  # inp
                                      V[index, :],  # output
                                      CopyDirection.d2k.value)

        st['index'] = (index + 1) % self.m

        ##### STEP 6
        ista_daslab_tools.zerorize_block_components(grad, I[index, :], d, k, d_block_size, k_block_size_many)  # this does a[I[index]] = 0

        ##### STEP 7
        def _update_quantization_statistics():
            if quant_full_blocks_count == 1:
                min_vals[:quant_full_blocks_count] = grad[:d_index_quant].min()
                max_vals[:quant_full_blocks_count] = grad[:d_index_quant].max()
            else:
                min_vals[:quant_full_blocks_count] = grad[:d_index_quant].view(quant_full_blocks_count, self.quant_block_size).min(dim=1).values
                max_vals[:quant_full_blocks_count] = grad[:d_index_quant].view(quant_full_blocks_count, self.quant_block_size).max(dim=1).values
            if d_index_quant < d:
                min_vals[quant_full_blocks_count] = grad[d_index_quant:].min()
                max_vals[quant_full_blocks_count] = grad[d_index_quant:].max()

        _update_quantization_statistics()

        ##### STEP 8
        ista_daslab_micro_adam.asymm_block_quant(d, self.quant_block_size, error, min_vals, max_vals, grad)  # error = Q(a, min, max)

        # weight decay step
        if wd > 0:
            p.mul_(1 - lr * wd)

        ##### NEW: densify using quant error
        if self.densify_update_using_quant_error:
            # When entering this if-statement, we have:
            #     - p is theta_t
            #     - p.grad is a_t (from step 6 in algorithm 1)
            #     - error is e_t+1 (from step 8 in algorithm 1)
            #
            # Below we have the formula to update the model parameters:
            # [a = -1] with lr
            #     theta_t+1 = theta_t - lr * (a_t - Qinv(e_t+1)) - lr * u_t
            #               = theta_t - lr * a_t + lr * Qinv(e_t+1) - lr * u_t
            #               = theta_t - lr * a_t              # STEP A below, in this if statmenet
            #                         + lr * Qinv(e_t+1)      # STEP B below, in this if statmenet
            #                         - lr * u_t              # this is steps 10-11
            #
            # [a = -2] without lr
            #     theta_t+1 = theta_t - (a_t - Qinv(e_t+1)) - lr * u_t
            #               = theta_t - a_t + Qinv(e_t+1) - lr * u_t
            #               = theta_t - a_t              # STEP A below, in this if statmenet
            #                         + Qinv(e_t+1)      # STEP B below, in this if statmenet
            #                         - lr * u_t              # this is steps 10-11
            quant_err = st['quant_err']
            quant_err.zero_()
            quant_err.add_(p.grad)

            ##### STEP A
            p.add_(p.grad, alpha=-1)

            ##### STEP B
            p.grad.zero_() # zerorize to prepare the accumulator for Qinv
            ista_daslab_micro_adam.asymm_block_quant_inv(d, self.quant_block_size, error, min_vals, max_vals, grad, 1)
            p.add_(p.grad)

            quant_err.sub_(p.grad)

            norm_qe = quant_err.norm(p=2) ** 2
            sp_qe = (quant_err == 0).sum()

        ##### STEPS 10-11
        grad.zero_()
        ista_daslab_micro_adam.compute_microadam_update(blocks,  # blocks
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

        ##### STEP 12: # side idea: only decay the weights that are update

        ##### if PRETRAINING #1
        if self.densify_update_using_ef: # we add alpha * EF to update that is stored in grad buffer
            # p.grad += alpha * Qinv(error), alpha=0.1
            ista_daslab_micro_adam.asymm_block_quant_inv(d, self.quant_block_size, error, min_vals, max_vals, grad, self.alpha)
        ##### END IF PRETRAINING #1

        # if alpha > 0, then the update u=p.grad is dense now

        # update model using MicroAdam update stored in p.grad
        p.add_(p.grad, alpha=-lr)

        if self.steps % self.log_interval == 0:
            norm_u = grad.norm(p=2) ** 2
            sp_u = (grad == 0).sum()  # check sparsity before zerorizing

        ##### if PRETRAINING #2
        if self.densify_update_using_ef:
            grad.zero_()
            ista_daslab_micro_adam.asymm_block_quant_inv(d, self.quant_block_size, error, min_vals, max_vals, grad, 1-self.alpha)

            _update_quantization_statistics() # step 7 again
            ista_daslab_micro_adam.asymm_block_quant(d, self.quant_block_size, error, min_vals, max_vals, grad) # step 8 again
        ##### END IF PRETRAINING #2

        # compute error norm
        if self.steps % self.log_interval == 0:
            grad.zero_()
            ista_daslab_micro_adam.asymm_block_quant_inv(d, self.quant_block_size, error, min_vals, max_vals, grad, 1.0)

            norm_e = grad.norm(p=2) ** 2

        # p.grad = p.grad.to(dtype=original_grad_type)

        return norm_qe, norm_g, norm_u, norm_e, sp_u, sp_qe

    def _log(self, norm_qe, norm_g, norm_u, norm_e, sparsity_u, sparsity_qe, elapsed_step):
        if self.steps % self.log_interval == 0:
            if is_initialized():
                sync_data = torch.tensor([norm_qe, norm_g, norm_u, norm_e, sparsity_u, sparsity_qe, elapsed_step], dtype=torch.float,
                                         requires_grad=False).cuda()  # correct, loss, size
                all_reduce(sync_data, op=ReduceOp.SUM)
                norm_qe, norm_g, norm_u, norm_e, sparsity_u, sparsity_qe, elapsed_step = sync_data

            if not is_initialized() or get_rank() == 0:
                wandb_data = {
                    'step/optimizer_steps': self.steps,
                    'step/gpu_mem_usage': get_gpu_mem_usage(),
                    'step/norm_quant_err': math.sqrt(norm_qe),
                    'step/sparsity_quant_err': sparsity_qe / self.model_size * 100.,
                    'step/norm_g': math.sqrt(norm_g),
                    'step/norm_u': math.sqrt(norm_u),
                    'step/norm_error': math.sqrt(norm_e),
                    'step/sparsity_u': sparsity_u / self.model_size * 100.,
                    'step/elapsed_step': elapsed_step,
                }
                wandb.log(wandb_data, commit=False)

    # def _update_lr_wd(self):
    #     # copy the learning rate group to parameter state because the lr scheduler updates the one in the group
    #     for group in self.param_groups:
    #         lr = group['lr']
    #         wd = group.get('weight_decay', self.weight_decay)  # if the param groups do not have weight decay, then use the external one
    #         for p in group['params']:
    #             self.state[p]['lr'] = lr
    #             self.state[p]['wd'] = wd


    # def _init_state(self):
    #     count = 0
    #     for group in self.param_groups:
    #         lr = group['lr']
    #         wd = group.get('weight_decay', self.weight_decay) # if the param groups do not have weight decay, then use the external one
    #         for p in group['params']:
    #             if not p.requires_grad:
    #                 continue

    #             print(f'[init_state] rank={torch.distributed.get_rank()}, p.shape={p.shape}')

    #             count += 1
    #             layer_size = p.numel()
    #             st = self.state[p]

    #             # B * t / d * nt
    #             st['blocks'] = max(1, int(math.floor(self.blocks * layer_size * self.dict_size_count[layer_size] / self.model_size)))

    #             st['lr'] = lr
    #             st['weight_decay'] = wd
    #             st['d'] = layer_size

    #             ##### variables for Top-K: d_index_topk is the index where the last, smaller topk block starts
    #             st['d_block_size'] = layer_size if layer_size < self.d_block_size else self.d_block_size
    #             st['topk_full_blocks_count'], st['d_index_topk'] = block_split(st['d'], st['d_block_size'])
    #             st['k_block_size_many'] = int(math.ceil(st['d_block_size'] * self.k_init))
    #             st['k_block_size_few'] = int(math.ceil((st['d'] - st['d_index_topk']) * self.k_init))  # 0 for d % self.d_block_size = 0
    #             st['k_index'] = st['topk_full_blocks_count'] * st['k_block_size_many']
    #             st['k'] = st['k_block_size_many'] * st['topk_full_blocks_count'] + st['k_block_size_few']

    #             ##### variables for the ring buffer
    #             st['index'] = 0  # the position to place a new gradient at
    #             st['I'] = torch.zeros(self.m, st['k'], dtype=torch.int16, device=self.device)  # 2mk bytes
    #             st['V'] = torch.zeros(self.m, st['k'], dtype=torch.bfloat16, device=self.device)  # 2mk bytes

    #             ### variables for error feedback: d_index_quant is the index where the last, smaller quantization block starts
    #             # st['quant_block_size'] = layer_size if layer_size < self.quant_block_size else self.quant_block_size
    #             st['quant_full_blocks_count'], st['d_index_quant'] = block_split(st['d'], self.quant_block_size)
    #             st['error'] = torch.zeros(int(math.ceil(st['d'] / 2)), dtype=torch.uint8, device=self.device)  # ceil(d/2) bytes
    #             st['min_vals'] = torch.zeros(st['quant_full_blocks_count'] + 1, dtype=torch.bfloat16, device=self.device)  # ceil(d/q_bsz)*2 bytes
    #             st['max_vals'] = torch.zeros(st['quant_full_blocks_count'] + 1, dtype=torch.bfloat16, device=self.device)  # ceil(d/q_bsz)*2 bytes
