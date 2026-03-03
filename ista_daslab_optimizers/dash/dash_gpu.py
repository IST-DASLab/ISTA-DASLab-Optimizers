import os
import torch
import torch.distributed as dist
from typing_extensions import override
from functools import partial

from ..ista_optimizer import ISTAOptimizer
from .dash_base import DashBase
from .dash_config import DashConfig
from .types import DashAlgoOneDim
from .processors import DashGpuProcessor1D, DashGpuProcessor2D

STATE_PROCESSOR_2D = "dash_layer_processor_2D"

class DashGpu(DashBase):
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
    """
    def __init__(self, param_groups, lr, weight_decay, config: DashConfig):
        # assert len(param_groups) == 1, f'DASH accepts only one parameter group that contains the entire optimization set'
        super().__init__(param_groups, lr=lr, weight_decay=weight_decay, config=config)
        self.dash_processor_1d: DashGpuProcessor1D = None # will process all 1D layers (after squeezing)
        self.dash_processor_2d: DashGpuProcessor2D = None  # will process all 2D layers (after squeezing & reshaping)

    @torch.no_grad()
    def get_ndim_params_from_current_bucket(self, ndim):
        """
            This function returns 1D or 2D layers from the buckets
            - if ndim = 1, we want 1D layers and we need to check the p_squeezed (psq) term
            - if ndim = 2, we want 2-3-4D layers and we need to check the p_squeezed_2d (psq2d) because it is already squeezed and reshaped to 2d
        """
        assert ndim in [1, 2]

        if dist.is_initialized():
            rank = dist.get_rank()
            looping_list = self.buckets[rank]
        else:
            looping_list = self.buckets

        for index, group, state, p, psq, psq2d in looping_list:
            if ndim == 1:
                if psq.ndim == 1: # check psq because 1D params have psq2D = None
                    yield index, group, state, p, psq, psq2d
            elif ndim == 2:
                if psq2d is not None and psq2d.ndim == 2:
                    yield index, group, state, p, psq, psq2d

    @override
    def init_optimizer_states(self):
        cfg: DashConfig = self.config
        algo_one_dim = cfg.algo_one_dim

        if self._optim_set_contains_dim(num_dims=1):
            bucket_func_1d = partial(self.get_ndim_params_from_current_bucket, ndim=1)
            self.dash_processor_1d = DashGpuProcessor1D(bucket_func=bucket_func_1d, cfg=cfg)

        if self._optim_set_contains_dim(num_dims=2):
            bucket_func_2d = partial(self.get_ndim_params_from_current_bucket, ndim=2)
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
        for index, group, state, p, psq, psq2d in self.get_ndim_params_from_current_bucket(ndim=2): # apply weight decay only for 2D layers
            lr = group['lr'] # this will also be available after the for-loop
            wd = group['weight_decay']

            if wd > 0:
                p.mul_(1 - lr * wd)
        # end for

        ##### UPDATE 1D & 2D layers with Shampoo (AdaGrad)
        if self.dash_processor_1d is not None:
            self.dash_processor_1d.update_layer(t=self.optim_steps, lr=self.param_groups[0]['lr'])

        if self.dash_processor_2d is not None:
            self.dash_processor_2d.update_layer(t=self.optim_steps, lr=self.param_groups[0]['lr'])

        self.sync_params()

    @torch.no_grad()
    def log_layer_stats(self):
        """
        This EXPENSIVE function is designed to be called outside of optimizer step.
        This way, the running time of optimizer step is not affected.
        """
        t = self.optim_steps

        if self.dash_processor_1d is not None:
            self.dash_processor_1d.log_stats(t)

        if self.dash_processor_2d is not None:
            self.dash_processor_2d.log_stats(t)
