import os
import torch
import torch.distributed as dist
from typing_extensions import override
from functools import partial

from ..ista_optimizer import ISTAOptimizer
from .dash_config import DashConfig
from .types import DashAlgoOneDim
from .processors import DashGpuProcessor1D, DashGpuProcessor2D

STATE_PROCESSOR_2D = "dash_layer_processor_2D"

class DashGpu(ISTAOptimizer):
    """
        Features we implement from the DistributedShampoo paper https://arxiv.org/pdf/2309.06497"

        Section 3.1.1: First and Second Moment Estimation
            [yes] use momentum for the gradient G, equivalent to momentum M in AdamW, using beta1. In the paper, this is tilde(G)
            [yes] bias correction doesn't make sense for Shampoo, only for grafting!
                [no] we can apply bias correction to tilde(G), L and R, obtaining hat(G), hat(L) and hat(R)
                [no] bias correction for L/R? see page 11, (between formulas 26 and 27) - llm_baselines used use_bias_correction=True
        Section 3.1.2: ℓ2-Regularization and (Decoupled) Weight Decay.
            - this section adds a hyper-parameter for decoupled weight decay
            [yes] use decoupled weight decay by default without any hyper-parameter
        Section 3.1.3: Momentum and Nesterov Acceleration
            [yes] Standard Momentum computed on the Shampoo update (formulas 31, 32), mu=0.9 usually
            [yes] Nesterov Momentum (formulas 33, 34)
        Section 3.1.4: Exponent Override and Exponent Multiplier
            [no] do not implement this for now
        Section 3.2.1: Matrix Root Inverse Solvers
            - check the gray rectangle from (1) Symmetric Eigendecomposition
            [yes] inherited this technique from Shampoo (gray rectangle)
        Section 4.1.1: Preconditioner Assignment and Load-Balancing via Greedy Algorithm
            [yes] implement Algorithm 3 to balance the load on each worker
                [yes] 1D parameters will be updated in a redundant way (on all GPUs)

        USE redundancy for now for all layers
            # How this Shampoo version updates the parameter groups:
            # - 1-D layers: updated using AdamW on all GPUs
            # - N-D layers: updated using Shampoo (one layer per GPU and then communicate parameter p across all GPUs)
    """
    def __init__(self, param_groups, lr, weight_decay, config: DashConfig):
        # assert len(param_groups) == 1, f'EfficientShampoo accepts only one parameter group that contains the entire optimization set'
        super().__init__(param_groups, lr=lr, weight_decay=weight_decay)
        self.config = config
        self.dash_processor_1d: DashGpuProcessor1D = None # this will be a custom object of type DashLayerProcessor that will process all 1D layers
        self.dash_processor_2d: DashGpuProcessor2D = None  # this will be a custom object of type DashLayerProcessor that will process all 2D layers
        """
        self.buckets is a dictionary with:
        - key: GPU index 
        - value: a list of 3-tuple containing (group, state, param) that will be processed on the GPU index as value.
        This dictionary is created in a greedy manner by sorting all parameters by the total number of parameters and
        assigning the parameter to the bucket (GPU index) that has the fewest number of parameters.
        """
        self.buckets = None # dict: key=rank, value=list with all parameters p updated on rank
        self.owners = None # dict: key=id(p), value=rank that updates the parameter p and which broadcast p to all other ranks
        self.numel_per_bucket = None # list: value at index i holds the total number of parameters processed by GPU #i
        self.create_param_buckets()

    @torch.no_grad()
    def create_param_buckets(self):
        params = sorted([
            (index, group, state, p) # also save the index!
            for index, (group, state, p) in enumerate(self.loop_params(check_grad=False))
        ], key=lambda x: x[3].numel(), reverse=True) # sort DESC by number of elements

        if dist.is_initialized(): # DDP is enabled
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            self.owners = {} # key = id(p), value = GPU rank
            self.buckets = {i: [] for i in range(world_size)}
            self.numel_per_bucket = [0] * world_size # position i stores how many parameters GPU #i has to process

            for index, group, state, p in params: # iterate through the list of parameters sorted by number of elements
                if p.ndim == 1: # 1D params will be splitted on all GPUs based on the GPU index
                    owner = index % world_size
                    self.owners[id(p)] = owner
                    if owner == rank:
                        self.buckets[rank].append((index, group, state, p))
                elif p.ndim == 2: # scatter 2D params across all ranks in a balanced way based on the number of params in numel_per_bucket
                    bucket_id = self.numel_per_bucket.index(min(self.numel_per_bucket)) # the bucket with minimum number of parameters so far
                    self.numel_per_bucket[bucket_id] += p.numel()
                    self.buckets[bucket_id].append((index, group, state, p))
                    self.owners[id(p)] = bucket_id
                else:
                    raise RuntimeError(f'Found a parameter with {p.ndim} dimensions! EfficientShampoo currently supports only 1D and 2D!')
            # end for index, group, state, p
        else: # all in a single bucket
            self.buckets = params

    @torch.no_grad()
    def get_ndim_from_current_bucket(self, ndim):
        assert ndim in [1, 2]
        if dist.is_initialized():
            rank = dist.get_rank()
            looping_list = self.buckets[rank]
        else:
            looping_list = self.buckets

        for index, group, state, p in looping_list:
            if p.ndim == ndim:
                yield index, group, state, p

    @override
    def init_optimizer_states(self):
        cfg: DashConfig = self.config
        algo_one_dim = cfg.algo_one_dim

        bucket_func_1d = partial(self.get_ndim_from_current_bucket, ndim=1)
        bucket_func_2d = partial(self.get_ndim_from_current_bucket, ndim=2)

        self.dash_processor_1d = DashGpuProcessor1D(bucket_func=bucket_func_1d, cfg=cfg)
        self.dash_processor_2d = DashGpuProcessor2D(bucket_func=bucket_func_2d, cfg=cfg)

    @override
    @torch.no_grad()
    def optimizer_step(self):
        """
        Optimization using Shampoo is run on the current bucket.
        At the end, we sync all updated parameters across all ranks
        """
        cfg = self.config

        ##### Apply weight decay to 2D layers
        for index, group, state, p in self.get_ndim_from_current_bucket(ndim=2):
            lr = group['lr'] # this will also be available after the for-loop
            wd = group['weight_decay']

            # apply weight decay only for 2D layers
            if wd > 0:
                p.mul_(1 - lr * wd)
        # end for

        ##### UPDATE 1D & 2D layers with Shampoo (AdaGrad)
        self.dash_processor_1d.update_layer(t=self.optim_steps, lr=lr)
        self.dash_processor_2d.update_layer(t=self.optim_steps, lr=lr)

        self.sync_params()

    @torch.no_grad()
    def sync_params(self):
        if dist.is_initialized():
           # iterate through all parameters
           for _, _, p in self.loop_params(check_grad=False):
               owner = self.owners[id(p)]
               if owner is not None:
                   dist.broadcast(p.data, src=owner)

    @torch.no_grad()
    def log_layer_stats(self):
        """
        This EXPENSIVE function is designed to be called outside of optimizer step.
        This way, the running time of optimizer step is not affected.
        """
        t = self.optim_steps
        self.dash_processor_1d.log_stats(t)
        self.dash_processor_2d.log_stats(t)

    @torch.no_grad()
    def log_bucket_stats(self, path):
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                params = sorted([
                    (index, group, state, p)  # also save the index!
                    for index, (group, state, p) in enumerate(self.loop_params(check_grad=False))
                ], key=lambda x: x[3].numel(), reverse=True)  # sort DESC by number of elements

                with open(os.path.join(path, f'general_layer_stats_rank={rank}.txt'), 'w') as w:
                    for index, group, state, p in params:
                        w.write(f'{index}: {tuple(p.shape)} => {p.numel():_}\n')

            with open(os.path.join(path, f'bucket_stats_rank={rank}.txt'), 'w') as w:
                w.write(f'Buckets for rank {rank}:\n')
                for bucket_id, bucket_content in self.buckets.items():
                    indexes = ','.join([str(index) for index, group, state, p in bucket_content])
                    params_per_bucket = self.numel_per_bucket[bucket_id]
                    w.write(f'\trank {bucket_id} ({params_per_bucket:_} params): {indexes}\n')