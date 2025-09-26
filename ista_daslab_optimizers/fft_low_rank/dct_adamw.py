import torch
import math
import numpy as np
import torch.distributed as dist
from fast_hadamard_transform import hadamard_transform
from optimizers.adam.low_rank.dct import _init_dct3_transform
from ..utils.quantizers import Quantizer4bit, Quantizer8bit
from optimizers.adam.low_rank.clr_projector import FFTLowRankProjector

# STRATEGY_FIRST = 'first'
STRATEGY_TOPK_LARGEST = 'topk-largest'
# STRATEGY_TOPK_SMALLEST = 'topk-smallest'
# STRATEGY_RANDOM = 'random'
# STRATEGY_WINDOW = 'window'

ALL_STRATEGIES = [
    # STRATEGY_FIRST, # choose first `r` columns
    STRATEGY_TOPK_LARGEST,
    # STRATEGY_TOPK_SMALLEST,
    # STRATEGY_RANDOM,
    # STRATEGY_WINDOW,
]

PROJ_DCT = 'dct'
PROJ_HDM = 'hdm'
PROJ_RAND_QR = 'rqr'

ALL_PROJ = [
    PROJ_DCT, # use the projection matrix from DCT
    PROJ_HDM, # use the projection
    PROJ_RAND_QR,
]

STATE_M = 'm'
STATE_V = 'v'
STATE_Q = 'Q'
STATE_ID = 'param-id'
STATE_EF = 'ef'
# STATE_EF_MIN = 'ef-min-vals'
# STATE_EF_MAX = 'ef-max-vals'
STATE_FFT_LRP = 'fft-low-rank-projector'
STATE_BROADCAST_SOURCE = 'broadcast-src' # the process rank that computes the update for a parameter p will broadcast the parameter p to other workers


class DCTAdamW(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr,
                 weight_decay,
                 rank,
                 proj,
                 # strategy,
                 use_ef=False,
                 q_ef=False,
                 distributed=False,
                 update_proj_gap=1,
                 rotate_subspace=False,
                 sim_type='matmul',
                 ell_norm=1,
                 max_shape=32_000,
                 betas=(0.9, 0.999),
                 eps=1e-8):
        # assert strategy in ALL_STRATEGIES
        assert proj in ALL_PROJ

        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

        self.rank = rank
        self.proj = proj
        self.use_ef = use_ef
        self.q_ef = q_ef
        self.distributed = distributed
        self.update_proj_gap = update_proj_gap
        self.rotate_subspace = rotate_subspace
        self.sim_type = sim_type
        self.ell_norm = ell_norm
        self.max_shape = max_shape # apply low-rank to 2D parameters that have both dimensions smaller than max_shape
        self.betas = betas
        self.eps = eps

        self.steps = 0
        self.is_state_initialized = False
        self.Q = None # the full transformation matrix (non-truncated, all rows and columns)
        self.Q_cols_norm = None
        self.use_theoretical_similarity = (self.ell_norm < 0)
        self.ell_norm = abs(self.ell_norm)

        if proj == PROJ_DCT:
            assert sim_type in ['matmul', 'makhoul']
        else:
            assert sim_type == 'matmul'

    def setup_Q(self, p):
        if self.Q is None:
            size = min(p.shape)
            if self.proj == PROJ_DCT:
                Qdct3 = _init_dct3_transform(size).to(device=p.device, dtype=p.dtype)  # first row is zero
                if self.sim_type == 'makhoul':
                    self.Q = Qdct3.t()
                    print(f'\n\t!!!!! Initialized DCT-2 matrix of size {size} !!!!!\n')
                elif self.sim_type == 'matmul':
                    self.Q = Qdct3
                    print(f'\n\t!!!!! Initialized DCT-3 matrix of size {size} !!!!!\n')
                else:
                    raise RuntimeError(f'Unknown sim_type: {self.sim_type}')
            elif self.proj == PROJ_HDM:
                self.Q = hadamard_transform(torch.eye(size).to(device=p.device, dtype=p.dtype), scale=1. / math.sqrt(size))
                print(f'\n\t!!!!! Initialized Hadamard matrix of size {size} !!!!!\n')
            elif self.proj == PROJ_RAND_QR:
                random = torch.randn(size, size, dtype=p.dtype, device=p.device)
                self.Q, _ = torch.linalg.qr(random)
                del random
            else:
                raise RuntimeError(f'Projection {self.proj} is currently not supported!')

            if self.use_theoretical_similarity:
                self.Q_cols_norm = self.Q.norm(p=self.ell_norm, dim=0)

    def should_compute_update(self, p):
        """
            This function returns a boolean indicating whether the update for the parameter p should be computed on the current GPU
        """
        state = self.state[p]
        param_id = state[STATE_ID]
        return param_id % dist.get_world_size() == dist.get_rank()

    def should_update_projection(self):
        return self.steps == 1 or self.steps % self.update_proj_gap == 0

    def init_state(self, p):
        state = self.state[p]
        if p.ndim == 1: # adam update
            print(f'Parameter of size {tuple(p.shape)} will receive original AdamW update with state shape {tuple(p.shape)}')
            state[STATE_M] = torch.zeros_like(p)
            state[STATE_V] = torch.zeros_like(p)
        elif p.ndim == 2: # low-rank adam update
            n, m = p.shape
            if n >= self.max_shape or m >= self.max_shape:  # apply full-rank
                print(f'Parameter of size {tuple(p.shape)} will receive original AdamW update with state shape {tuple(p.shape)}')
                state[STATE_M] = torch.zeros_like(p)
                state[STATE_V] = torch.zeros_like(p)
            else: # apply low-rank using the DCT transform as orthogonal matrix
                if n >= m:
                    low_rank_shape = (n, self.rank)
                else:
                    # fix for Llama-3-8B that has a layer of size (1024, 4096)
                    # fix for Qwen2.5-7B that has a layer of size (512, 3584)
                    if n in [512, 1024] and m in [3584, 4096]:
                        low_rank_shape = (n, self.rank)
                    else:
                        low_rank_shape = (self.rank, m)
                # low_rank_shape = (n, self.rank) if n >= m else (self.rank, m)
                print(f'Parameter of size {tuple(p.shape)} will receive low-rank update with state shape {low_rank_shape}')
                state[STATE_M] = torch.zeros(*low_rank_shape, dtype=p.dtype, device=p.device)
                state[STATE_V] = torch.zeros(*low_rank_shape, dtype=p.dtype, device=p.device)
                state[STATE_FFT_LRP] = FFTLowRankProjector(p,
                                                          rank=self.rank,
                                                          proj=self.proj,
                                                          rotate_subspace=self.rotate_subspace,
                                                          sim_type=self.sim_type,
                                                          ell_norm=self.ell_norm,
                                                          use_th_sim=self.use_theoretical_similarity)
                if self.use_ef:
                    if self.q_ef > 0:
                        # state[STATE_EF] = torch.zeros(p.numel() // 2, dtype=torch.uint8, device=p.device)
                        # state[STATE_EF_MIN] = torch.zeros(p.shape[0], dtype=torch.bfloat16, device=p.device)
                        # state[STATE_EF_MAX] = torch.zeros(p.shape[0], dtype=torch.bfloat16, device=p.device)
                        quantClass = {4: Quantizer4bit, 8: Quantizer8bit}[self.q_ef]
                        if self.q_ef == 4:
                            quantClass = Quantizer4bit
                            print(f'\n\t!!!!! Quantizing EF to 4 bits !!!!!\n')
                        elif self.q_ef == 8:
                            quantClass = Quantizer8bit
                            print(f'\n\t!!!!! Quantizing EF to 8 bits !!!!!\n')
                        else:
                            raise RuntimeError(f'Quantization on {self.q_ef} bits is currently not supported!')
                        state[STATE_EF] = quantClass(shape=p.shape, device=p.device, dtype=p.dtype, bucket_size=p.shape[1])
                    else:
                        state[STATE_EF] = torch.zeros_like(p)

                ### initialize Q
                print('calling setup_Q')
                self.setup_Q(p)
        # end if

    def init(self):
        # init broadcast info
        self.is_state_initialized = True
        bcast_src_list = []
        param_id = 0 # parameter id
        for group in self.param_groups:
            for p in group['params']:
                if p is None: continue
                if p.grad is None: continue

                state = self.state[p]
                if len(state) == 0:
                    if self.distributed:
                        state[STATE_ID] = param_id
                        param_id += 1
                        if self.should_compute_update(p):
                            # if the current process computes the update, then it will also broadcast the parameters to all other workers
                            state[STATE_BROADCAST_SOURCE] = torch.tensor(dist.get_rank(), dtype=torch.int32, device=f'cuda:{dist.get_rank()}')
                            self.init_state(p)
                        else:
                            # p.register_hook(lambda grad: None) # set gradient to None
                            # p.requires_grad = False # disable gradient computation for this layer
                            state[STATE_BROADCAST_SOURCE] = torch.tensor(0, dtype=torch.int32, device=f'cuda:{dist.get_rank()}') # zero means empty here because we will do an all reduce
                        bcast_src_list.append(state[STATE_BROADCAST_SOURCE].item())
                    else:
                        self.init_state(p)
        # end for group

        if self.distributed:
            dist.barrier()

            # with open(f'broadcast-{dist.get_rank()}.txt', 'w') as w:
            # sync broadcast source
            # w.write(f'Broadcast SRC on worker {dist.get_rank()} before all_reduce: {",".join(map(str, bcast_src_list))}\n')
            bcast_src_list = []
            for group in self.param_groups:
                for p in group['params']:
                    if p is None: continue
                    if p.grad is None: continue

                    state = self.state[p]
                    dist.all_reduce(state[STATE_BROADCAST_SOURCE], op=dist.ReduceOp.SUM)
                    state[STATE_BROADCAST_SOURCE] = state[STATE_BROADCAST_SOURCE].item()
                    bcast_src_list.append(state[STATE_BROADCAST_SOURCE])
            # end for group
            # w.write(f'Broadcast SRC on worker {dist.get_rank()} after all_reduce: {",".join(map(str, bcast_src_list))}\n')
            dist.barrier()
        # end if
        torch.cuda.empty_cache()

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.is_state_initialized:
            self.init() # init broadcast info

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p is None: continue
                if p.grad is None: continue

                if wd > 0:
                    p.mul_(1 - lr * wd)

                if self.distributed:
                    if self.should_compute_update(p):
                        self.update_step(p, lr)
                else:
                    self.update_step(p, lr)
        # end for group

        if self.distributed:
            for group in self.param_groups:
                for p in group['params']:
                    if p is None: continue
                    if p.grad is None: continue

                    dist.broadcast(p, src=self.state[p][STATE_BROADCAST_SOURCE])

            # end for group
            dist.barrier() # wait for all GPUs to compute the update for all layers
        # end if distributed
        return loss

    @torch.no_grad()
    def update_step(self, p, lr):
        if p.ndim == 1:  # adam update
            self.adamw_step(p, lr)
        elif p.ndim == 2: # low-rank adam update
            n, m = p.shape
            if n >= self.max_shape or m >= self.max_shape:  # apply full-rank for parameters that have at least one dimension >= max_size (e.g. embeddings and lm_head)
                self.adamw_step(p, lr)
            else:
                self.cheap_low_rank_step(p, lr)

    def cheap_low_rank_step(self, p, lr):
        beta1, beta2 = self.betas
        bc1 = 1 - beta1 ** self.steps
        sqrt_bc2 = math.sqrt(1 - beta2 ** self.steps)
        adjusted_lr = -lr * sqrt_bc2 / bc1

        A = p.grad # initially, the accumulator stores gradient and a bit later we will add the error feedback
        state = self.state[p]

        mt = state[STATE_M]
        vt = state[STATE_V]

        if self.use_ef:
            E = state[STATE_EF]
            if self.q_ef:
                # see step 4 from Algorithm 1 in the MicroAdam paper https://arxiv.black/pdf/2405.15593
                A.add_(E.quantize_inv()) # p.grad += Qinv(EF)
            else:
                A.add_(E)

        clrp: FFTLowRankProjector = state[STATE_FFT_LRP]
        clrp.inc_step()

        if self.should_update_projection():
            a = clrp.change_subspace(self.Q, A, col_norms=self.Q_cols_norm)
        else:
            ### compute low-rank accumulator a
            a = clrp.from_higher_to_lower_dimensions(self.Q, A)

        if self.use_ef:
            A_reconstructed = clrp.from_lower_to_higher_dimensions(self.Q, a)
            if self.q_ef:
                A.sub_(A_reconstructed) # the full precision EF is stored now in A
                # see step 8 from Algorithm 1 in the MicroAdam paper https://arxiv.black/pdf/2405.15593
                E.quantize(A)
            else:
                E.copy_(A).sub_(A_reconstructed)
            del A_reconstructed

        ### update momentum m and v (rotate first, if needed)
        if self.steps > 1 and self.rotate_subspace and self.should_update_projection():
            R = clrp.get_subspace_rotation_matrix(self.Q)
            clrp.rotate_subspace(R, mt)
            clrp.rotate_subspace(R, vt)
            vt.abs_()  # make sure vt is positive
            del R

        mt.mul_(beta1).add_(a, alpha=1 - beta1)
        vt.mul_(beta2).addcmul_(a, a, value=1 - beta2)

        u = mt / (self.eps * sqrt_bc2 + vt.sqrt())
        clrp.from_lower_to_higher_dimensions(self.Q, u, out=p.grad)
        del u, a

        p.add_(p.grad, alpha=adjusted_lr)

    @torch.no_grad()
    def adamw_step(self, p, lr):
        state = self.state[p]
        g = p.grad

        mt = state[STATE_M]
        vt = state[STATE_V]

        beta1, beta2 = self.betas
        bc1 = 1 - beta1 ** self.steps
        sqrt_bc2 = math.sqrt(1 - beta2 ** self.steps)
        adjusted_lr = -lr * sqrt_bc2 / bc1

        # update momentum m and v
        mt.mul_(beta1).add_(g, alpha=1-beta1)
        vt.mul_(beta2).addcmul_(g, g, value=1-beta2)

        # U = mt / (self.eps * sqrt_bc2 + vt.sqrt())
        g.copy_(vt).sqrt_().add_(self.eps * sqrt_bc2).div_(mt).reciprocal_()
        p.add_(g, alpha=adjusted_lr)
