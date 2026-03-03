from abc import abstractmethod
from typing import Union, List, Dict
import torch
import torch.distributed as dist

from ..ista_optimizer import ISTAOptimizer

class DashBase(ISTAOptimizer):
    def __init__(self, param_groups, lr, weight_decay, config):
        super().__init__(param_groups, lr=lr, weight_decay=weight_decay)
        self.config = config

        """
        self.buckets is a dictionary with:
            - key: GPU index 
            - value: a list of 3-tuple containing (group, state, param) that will be processed on the GPU index as value.
        This dictionary is created in a greedy manner by sorting all parameters by the total number of parameters and
        assigning the parameter to the bucket (GPU index) that has the fewest number of parameters.
        """
        self.buckets: Dict = None  # dict: key=rank, value=list with all parameters p updated on rank
        self.owners: Dict = None  # dict: key=id(p), value=rank that updates the parameter p and which broadcast p to all other ranks
        self.numel_per_bucket: Union[Dict, List] = None  # numel_per_bucket[i] = the total number of parameters processed by GPU #i
        self.create_param_buckets()

    @torch.no_grad()
    def loop_params(self, check_grad=True):
        for group in self.param_groups:
            for p in group['params']:
                if check_grad:
                    if p.grad is None:
                        continue
                p_squeezed = p.squeeze() # p reshaped
                if p_squeezed.ndim == 1:
                    yield group, self.state[p], p, p_squeezed, None
                else: # 2D, 3D, 4D viewed as 2D
                    p_squeezed_2d = p_squeezed.view(p_squeezed.shape[0], p_squeezed.numel() // p_squeezed.shape[0])
                    yield group, self.state[p], p, p_squeezed, p_squeezed_2d

    @torch.no_grad()
    def _optim_set_contains_dim(self, num_dims):
        """
        This function should be called to check whether the optimization set contains parameter with num_dim dimensions (after squeezing)
        """
        assert num_dims in [1, 2]

        for group, state, p, psq, psq2d in self.loop_params(check_grad=False):
            if num_dims == 1:
                if psq.ndim == 1:
                    return True
            elif num_dims == 2:
                if psq2d is not None and psq2d.ndim == 2:
                    return True
        return False

    @torch.no_grad()
    def create_param_buckets(self):
        """
            This method creates buckets with parameters for each GPU worker.
            - 1D params: will be balanced across all GPUs because the sizes are small
            - 2D params: they are actually 2D or higher-dimensional parameters that turned out to be 2D after squeezing and reshaping
        """
        params = sorted([
            (index, group, state, p, psq, psq2d)  # also save the index!
            for index, (group, state, p, psq, psq2d) in enumerate(self.loop_params(check_grad=False))
        ], key=lambda x: x[3].numel(), reverse=True)  # sort DESC by number of elements

        if dist.is_initialized():  # DDP is enabled
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            self.owners = {}  # key = id(p), value = GPU rank
            self.buckets = {i: [] for i in range(world_size)}
            self.numel_per_bucket = [0] * world_size  # position i stores how many parameters GPU #i has to process

            for index, group, state, p, psq, psq2d in params:  # the list of parameters is sorted by number of parameters
                bucket_item = (index, group, state, p, psq, psq2d)

                if psq2d is None: # means p is a 1D param that will be splitted on all GPUs based on the GPU index
                    owner = index % world_size
                    self.owners[id(p)] = owner
                    if owner == rank:
                        self.buckets[rank].append(bucket_item)
                else: # means p is a 2D,3D or 4D layer: scatter params across all ranks in a balanced way based on the number of params in numel_per_bucket
                    # here we actually apply the greedy approach from Algorithm 3 in DistributedShampoo paper
                    bucket_id = self.numel_per_bucket.index(min(self.numel_per_bucket))  # the bucket with minimum number of parameters so far
                    self.numel_per_bucket[bucket_id] += p.numel()
                    self.buckets[bucket_id].append(bucket_item)
                    self.owners[id(p)] = bucket_id
            # end for index, group, state, p
        else:  # all in a single bucket
            self.buckets = params

    @torch.no_grad()
    def sync_params(self):
        if dist.is_initialized():
           # iterate through all parameters
           for _, _, p, psq, psq2d in self.loop_params(check_grad=False):
               owner = self.owners.get(id(p), None)
               if owner is not None:
                   dist.broadcast(p.data, src=owner)

    @torch.no_grad()
    def log_bucket_stats(self, path):
        """
            This function is common for both DashLayerwise and DashGpu
        """
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                params = sorted([
                    (index, group, state, p, psq, psq2d)  # also save the index!
                    for index, (group, state, p, psq, psq2d) in enumerate(self.loop_params(check_grad=False))
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

    ##################################################
    #################### Abstract Classes
    ##################################################

    @torch.no_grad()
    @abstractmethod
    def log_layer_stats(self):
        raise NotImplementedError(f'The function log_layer_stats must be implemented in actual Dash classes!')