import os
import torch
import torch.distributed as dist
import wandb
from typing_extensions import override

from .dash_layerwise_block_partitioner import DashLayerwiseBlockPartitioner, DashMatrixBlock, DashFakeTensorWithGrad
from .dash_configs import DashConfig, DashAlgoOneDim
from .dash_layerwise_processor import DashLayerwiseProcessor
from ..ista_optimizer import ISTAOptimizer

STATE_PROCESSOR = "dash_layer_processor"

class DashLayerwise(ISTAOptimizer):
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
        self.norm_layers_stack = None # this will be an object of type FakeLayerWithGrad with shape (#num_norm_layers, embedding_size)
        self.norm_layers_processor = None # this will be a custom object of type ShampooLayerProcessor that will process `norm_layers_stack`
        """
        This is a dictionary with:
        - key: GPU index 
        - value: a list of 3-tuple containing (group, state, param) that will be processed on the GPU index as value.
        This dictionary is created in a greedy manner by sorting all parameters by the total number of parameters and
        assigning the parameter to the bucket (GPU index) that has the fewest number of parameters.
        """
        self.buckets = None # dict: key=rank, value=list with all parameters p updated on rank
        self.owners = None # dict: key=id(p), value=rank that updates the parameter p and which broadcast p to all other ranks
        self.numel_per_bucket = None # list: value at index i holds the total number of parameters processed by GPU #i

        # self.ids_params_1d = None

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

            self.owners = {} # key = id(p), value = rank that will process the layer and will broadcast it to others
            self.buckets = {i: [] for i in range(world_size)}
            self.numel_per_bucket = [0] * world_size

            # self.ids_params_1d = {i: [] for i in range(world_size)}

            for index, group, state, p in params:
                if p.ndim == 1: # 1D params will be processed locally by all ranks
                    owner = index % world_size
                    self.owners[id(p)] = owner
                    if owner == rank:
                        self.buckets[rank].append((index, group, state, p))
                        # self.ids_params_1d[rank].append(index)

                    # self.owners[id(p)] = None # special marker
                    # for r in range(world_size):
                    #     self.buckets[rank].append((index, group, state, p))

                    # # process all 1D params on the last GPUs that usually has the lowest load
                    # self.buckets[world_size-1].append((index, group, state, p))
                    # self.owners[id(p)] = world_size - 1
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
    def get_current_bucket(self, get_1d=False):
        if dist.is_initialized():
            rank = dist.get_rank()
            looping_list = self.buckets[rank]
        else:
            looping_list = self.buckets

        for index, group, state, p in looping_list:
            if get_1d:
                if p.ndim == 1:
                    yield index, group, state, p
            else:
                yield index, group, state, p

    @override
    def init_optimizer_states(self):
        cfg: DashConfig = self.config
        algo_one_dim = cfg.algo_one_dim

        ##### UPDATE 2D layers and 1D layers with AdamW
        for index, group, state, p in self.get_current_bucket():
            ndim = p.ndim
            if (ndim == 2) or (ndim == 1 and algo_one_dim == DashAlgoOneDim.ADAMW):
                state[STATE_PROCESSOR] = DashLayerwiseProcessor(
                    param=p,
                    cfg=self.config,
                    is_norm_layer_stack=False, # important !!!
                    name=f'{index:03d}')
            elif ndim not in [1, 2]:
                raise RuntimeError(f'Efficient Shampoo is currently implemented only for 1D and 2D layers')

        ##### UPDATE 1D layers with Shampoo (AdaGrad)
        if algo_one_dim == DashAlgoOneDim.SHAMPOO:
            pool = [p for index, group, state, p in self.get_current_bucket(get_1d=True)]
            N = len(pool)
            E = pool[0].shape[0] # this is embedding size
            dtype = pool[0].dtype
            device = pool[0].device

            self.norm_layers_stack = DashFakeTensorWithGrad(shape=(N, E, 1), dtype=dtype, device=device)
            self.norm_layers_processor = DashLayerwiseProcessor(
                param=self.norm_layers_stack,
                cfg=self.config,
                is_norm_layer_stack=True, # important !!!
                name=f'norm-layers-stack')

    @override
    @torch.no_grad()
    def optimizer_step(self):
        """
        Optimization using Shampoo is run on the current bucket.
        At the end, we sync all updated parameters across all ranks
        """
        cfg = self.config
        algo_one_dim = cfg.algo_one_dim

        ##### UPDATE 2D layers and 1D layers with AdamW
        for index, group, state, p in self.get_current_bucket():
            ndim = p.ndim
            if (ndim == 2) or (ndim == 1 and algo_one_dim == DashAlgoOneDim.ADAMW):
                lr = group['lr']
                wd = group['weight_decay']

                # apply weight decay only for 2D layers
                if (ndim == 2) and (wd > 0):
                    p.mul_(1 - lr * wd)

                state[STATE_PROCESSOR].update_layer(t=self.optim_steps, lr=lr)

        ##### UPDATE 1D layers with Shampoo (AdaGrad)
        if algo_one_dim == DashAlgoOneDim.SHAMPOO:
            nls = self.norm_layers_stack
            ########################################
            ##### STEP 1
            ##### Update the fake tensor object `norm_layers_stack`:
            ##### the gradient of each normalization layer  of shape (E,)
            ##### will be placed in `norm_layers_stack` at index  `nli`
            ##### in both p and grad fields
            ########################################
            for nli, (index, group, state, p) in enumerate(self.get_current_bucket(get_1d=True)): # nli stands for norm-layer-index
                nls.p[nli, :, 0].copy_(p)
                nls.grad[nli, :, 0].copy_(p.grad)

            ########################################
            ##### STEP 2
            ##### Call update_layer on the processor, which has the flag `is_norm_layer_stack=True`
            ##### It should handle this particular layer correctly: it won't check the grad field,
            ##### but will work directly with the values, which are already gradients set in Step 1
            ########################################
            self.norm_layers_processor.update_layer(t=self.optim_steps, lr=lr)

            ########################################
            ##### STEP 3
            ##### The call update_layer on the processor will update the `p` field in the FakeTensorWithGrad object
            ##### Now we need to copy back from the `p` field of the FakeTensorWithGrad to actual model parameters
            ##### This step can be seen as the reverse of STEP 1
            ########################################
            for nli, (index, group, state, p) in enumerate(self.get_current_bucket(get_1d=True)): # nli stands for norm-layer-index
                p.copy_(nls.p[nli, :, 0])
                # p.grad.copy_(nls.grad[nli, :, 0]) # no need to copy the gradient

        self.sync_params()

    @torch.no_grad()
    def sync_params(self):
        if dist.is_initialized():
           # iterate through all parameters
           for _, _, p in self.loop_params(check_grad=False):
               owner = self.owners.get(id(p), None)
               if owner is not None:
                   dist.broadcast(p.data, src=owner)

    @torch.no_grad()
    def log_layer_stats(self):
        """
        This EXPENSIVE function is designed to be called outside of optimizer step.
        This way, the running time of optimizer step is not affected.
        """
        algo_one_dim = self.config.algo_one_dim
        if algo_one_dim == DashAlgoOneDim.ADAMW:
            for index, group, state, p in self.get_current_bucket():
                if (ndim == 2) or (ndim == 1 and algo_one_dim == DashAlgoOneDim.ADAMW):
                    state[STATE_PROCESSOR].log_stats(t=self.optim_steps)
        elif algo_one_dim == DashAlgoOneDim.SHAMPOO:
            self.norm_layers_processor.log_stats(t=self.optim_steps)

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
                # w.write(f'\n\n\n1d params:\n')
                # for bucket_id, ids_params_1d in self.ids_params_1d.items():
                #     indexes = ','.join([str(i) for i in ids_params_1d])
                #     # params_per_bucket = self.numel_per_bucket[bucket_id]
                #     w.write(f'\trank {bucket_id}: {indexes}\n')
