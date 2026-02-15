from torch import Tensor
from typing import Callable

from ..tools import PartitionStats

STATE_DASH_SHAPE = 'dash_shape'


class DashGpuPartitioner:
    """
    Given a full block and a rest block, a PartitionStats object and an output parameter,
    reconstructs the full/rest blocks into an output parameter
    """
    @staticmethod
    def reconstruct_from_blocks_for_2d(full_block: Tensor, rest_block: Tensor, block_size: int, stats: PartitionStats, out: Tensor):
        B = block_size
        R_full = stats.R_full
        C_full = stats.C_full
        num_row_blocks = stats.num_row_blocks
        num_col_blocks = stats.num_col_blocks
        col_rest = stats.col_rest
        row_rest = stats.row_rest

        full = full_block.view(num_row_blocks, num_col_blocks, B, B)
        full = full.transpose(1, 2)  # (num_row_blocks, B, num_col_blocks, B)
        out[:R_full, :C_full].copy_(full.reshape(R_full, C_full))
        
        if rest_block is not None:
            if col_rest > 0: # Case A: remainder in columns (shape: (num_row_blocks, B, col_rest))
                right = rest_block.reshape(R_full, col_rest)
                out[:R_full, C_full:].copy_(right)
            elif row_rest > 0: # Case B: remainder in rows (shape: (num_col_blocks, row_rest, B))
                bottom = rest_block.transpose(0, 1)  # shape => (row_rest, C_full)
                bottom = bottom.reshape(row_rest, C_full)
                out[R_full:, :C_full].copy_(bottom)

