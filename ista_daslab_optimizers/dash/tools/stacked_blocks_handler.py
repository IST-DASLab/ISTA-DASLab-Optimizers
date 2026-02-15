import torch
from torch import Tensor
import torch.distributed as dist

from ...utils.triton_kernels import update_dash_preconditioner

class DashStackedBlocksHandler:
    """
    This class handles a set of rests for the stacked preconditioners in DashGpu.
    For example, if for one GPU we have:
                    shape_G_2d: {
                        (1024, 1024): DashShape3D(b=110, m=1024, n=1024),
                        (256, 1024): DashShape3D(b=2, m=256, n=1024),
                        (1024, 512): DashShape3D(b=4, m=1024, n=512),
                        (512, 1024): DashShape3D(b=4, m=512, n=1024)
                    }
                    OR
                    shape_LRrest_2d: {
                        (256, 256): DashShape3D(b=2, m=256, n=256),
                        (512, 512): DashShape3D(b=8, m=512, n=512)
                    }

    then this class should store:
                    state_G: {
                        (1024, 1024): Tensor(110, 1024, 1024),
                        (256, 1024): Tensor(2, 256, 1024),
                        (1024, 512): Tensor(4, 1024, 512),
                        (512, 1024): Tensor(4, 512, 1024)
                    }
                    OR
                    state_LR: {
                        (256, 256): Tensor(2, 256, 256),
                        (512, 512): Tensor(8, 512, 512)
                    },
    and be able to add a tensor of shape (256, 256) to the correct location by
    holding some indices
    """
    def __init__(self, factor_shape: dict, block_size: int, dtype: torch.dtype, device: torch.device):
        """
        Arguments:
        - factor_shape: {
            (256, 256): DashShape3D(b=2, m=256, n=256),
            (512, 512): DashShape3D(b=8, m=512, n=512)
         }
        - block_size: the block size used in DASH, which is used to make sure the rests have a smaller
                      size than the global block
        """
        self._factor_shapes = factor_shape
        self._block_size = block_size # used to make sure we integrated rests
        self._dtype = dtype
        self._device = device

        self._state = {}
        self._stacking_indices = {}
        self._unstacking_indices = {}
        self._is_triton_init = {}
        for shape_tuple, shape3d in factor_shape.items():
            self._state[shape_tuple] = torch.zeros(shape3d.as_tuple(), dtype=dtype, device=device, requires_grad=False)
            self._stacking_indices[shape_tuple] = 0
            self._unstacking_indices[shape_tuple] = 0
            self._is_triton_init[shape_tuple] = False

    def reset_stacking_indices(self):
        for shape in self._stacking_indices:
            self._stacking_indices[shape] = 0

    def reset_unstacking_indices(self):
        for shape in self._stacking_indices:
            self._unstacking_indices[shape] = 0

    def copy_block(self, block: Tensor): # used for Pshmp
        """
        This function is used to directly copy the argument `block` to the corresponding shape in state
        This is particularly useful when we want to update Pshmp (see if statement t < start_prec_step)
        where we anticipate the next step.
        """
        assert block.ndim == 3 # it has to be 3D
        n, r, c = block.shape
        assert (r, c) in self._state # the block shape has to be in the state
        assert self._state[(r, c)].shape == block.shape # make sure we copy full blocks to avoid errors

        self._state[(r, c)].copy_(block)

    def stack_block(self, block: Tensor, action: str, lerp_weight=None):
        """
        This function should be used when stacking a block obtained as G @ G.T or G.T @ G using bmm.
        Arguments:
        - `factor`: must be a tensor of shape (n, r, c), where:
            - n is the batch size, which will be added to self.indices[(r, c)]
            - r is the number of rows
            - c is the number of columns
            - r and c can be different, as this class stores bblocks for both G and L/R
        - `action`: specifies which function we should use to integrate the factor:
            - add: simply adds the factor
            - lerp: integrates the factor in a moving average (requires ema_decay
            - copy: copies the factor
        """
        if block is None:
            return

        assert block.ndim == 3 # it has to be 3D
        # assert block.shape[1] == block.shape[2] # it has to contain batches of squared matrices
        # assert block.shape[1] < self._block_size # for safety
        n, r, c = block.shape
        assert (r, c) in self._state
        assert (r, c) in self._stacking_indices
        assert action in ['add', 'lerp', 'copy']

        i = self._stacking_indices[(r, c)]
        buffer = self._state[(r, c)]

        match action:
            case 'add':
                buffer[i: i + n].add_(block)
            case 'lerp':
                assert type(lerp_weight) == float
                assert 0 < lerp_weight < 1
                buffer[i: i + n].lerp_(block, weight=lerp_weight)
            case 'copy':
                buffer[i: i + n].copy_(block)
        # end match-case
        self._stacking_indices[(r, c)] += n

    def stack_grad_product(self, G: Tensor, beta=None):
        """
        This function should be used as an alternative to stack_block (bmm) via triton kernel.
        It receives the gradient and calls the triton kernel to integrate G @ G.T for L and G.T @ G for R in-place.
        Since the triton kernel is stateful (modifies the matrix in-place), it takes care of the triton initialization.
        This function must be called only from the LR object, not from the gradient
        """
        n, r, c = G.shape
        for z, is_left in [(r, True), (c, False)]: # buffers L and R
            shape = (z, z)
            assert shape in self._state
            assert shape in self._stacking_indices
            buffer = self._state[shape]

            # first, check initialization
            if self._is_triton_init[shape] == False:
                self._is_triton_init[shape] = True

                rank = dist.get_rank() if dist.is_initialized() else 0
                print(f'[rank={rank}] Running triton kernel for the first time for shape {shape}')
                update_dash_preconditioner(X=buffer[0:n], G=G, beta=beta, compute_left=is_left)

                buffer.zero_()  # IMPORTANT: reset buffer because the kernel modifies X in-place during warmup

            # second, do the actual update
            i = self._stacking_indices[shape]
            update_dash_preconditioner(X=buffer[i:i+n], G=G, beta=beta, compute_left=is_left)  # update matrix L
            self._stacking_indices[shape] += n
        # end if

    def unstack_block(self, shape, length):
        """
        If `shape` exists in _state and _unstacking_indices, then we extract the current index as
                i = _unstacking_indices[shape]
        and extract from _state[shape] `length` matrices starting with index i as follows:
                _state[shape][i : i + length]
        then update the index as
                _unstacking_indices[shape] += length
        """
        assert shape in self._state
        assert shape in self._unstacking_indices
        i = self._unstacking_indices[shape]
        block = self._state[shape][i : i + length]
        self._unstacking_indices[shape] += length
        return block

    def iter_shapes_blocks(self):
        for shape, block in self._state.items():
            yield shape, block

    def iter_shapes(self):
        for shape in self._state.keys():
            yield shape

    def iter_blocks(self):
        for block in self._state.values():
            yield block

    @classmethod
    def like(cls, other):
        return cls(factor_shape=other._factor_shapes,
                   block_size=other._block_size,
                   dtype=other._dtype,
                   device=other._device)
