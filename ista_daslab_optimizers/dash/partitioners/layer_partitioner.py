import torch
from torch import Tensor
from typing import Union

from ..types import DashPartitionInfo
from ..tools import DashMatrixBlock, DashFakeParam


class DashLayerPartitioner:
    """
    Partitions a matrix into blocks of size (B, B) when possible, and
    remainder blocks of size (B, b) or (b, B), with the assumption that
    B divides either R or C (but not necessarily both). This guarantees
    at most one type of remainder.
    """

    def __init__(self, param: Union[Tensor, DashFakeParam], B: int, is_norm_layer_stack: bool):
        if isinstance(param, Tensor):
            assert not is_norm_layer_stack, \
                'DashLayerwisePartitioner does not support param:Tensor and is_norm_layer_stack:True'
        if isinstance(param, DashFakeParam):
            assert is_norm_layer_stack, \
                'DashLayerwisePartitioner does not support param:DashFakeParam and is_norm_layer_stack:False'
        if is_norm_layer_stack:
            assert param.shape[1] % B == 0, \
                'DashLayerwisePartitioner does not support block rests for stacked norm layers.'
        self.param = param
        self.B = B
        self.is_norm_layer_stack = is_norm_layer_stack

        # ---------------------------------------------------------
        # Support for stacked normalization layers
        # ---------------------------------------------------------
        if self.is_norm_layer_stack:
            # self.is_1d = True

            N = self.param.shape[0] # number of normalization layers
            E = self.param.shape[1] # embedding size
            B = self.B

            self.N = N
            self.R = E
            self.C = 1  # Treated as (n, 1)

            self.R_full = (E // B) * B
            self.C_full = 0  # Not used for 1D logic in the same way

            self.num_row_blocks = self.R_full // B
            self.num_col_blocks = 1
            self.num_blocks_full = N * self.num_row_blocks

            self.row_rest = E - self.R_full
            self.col_rest = 0

            # For 1D, the "full" blocks are just segments of the vector of size B.
            # Shape is (num_blocks, B, 1).
            self.shape_full = (self.num_blocks_full, B, 1)

            if self.row_rest > 0:
                self.num_blocks_rest = 1
                # Remainder block is the tail of the vector
                self.shape_rest = (N * 1, self.row_rest, 1)
            else:
                self.num_blocks_rest = None
                self.shape_rest = None
        else:
            R, C = param.shape
            self.is_1d = False

            # ---------------------------------------------------------
            # Existing 2D
            # ---------------------------------------------------------
            assert (R % B == 0) or (C % B == 0), f"Block size must be a multiple of rows and columns: {R=}, {C=}, {B=}"

            self.R = R
            self.C = C

            # Compute full block extents
            self.R_full = (R // B) * B
            self.C_full = (C // B) * B

            self.num_row_blocks = self.R_full // B
            self.num_col_blocks = self.C_full // B
            self.num_blocks_full = self.num_row_blocks * self.num_col_blocks
            # Remainders
            self.row_rest = R - self.R_full
            self.col_rest = C - self.C_full

            self.shape_full = (self.num_blocks_full, B, B)
            if self.col_rest > 0:
                self.num_blocks_rest = self.num_row_blocks
                self.shape_rest = (self.num_row_blocks, B, self.col_rest)
            elif self.row_rest > 0:
                self.num_blocks_rest = self.num_col_blocks
                self.shape_rest = (self.num_col_blocks, self.row_rest, B)
            else:
                self.num_blocks_rest = None
                self.shape_rest = None

    def get_regular_gradient_block(self):
        return DashMatrixBlock(
            shape_full=self.shape_full,
            shape_rest=self.shape_rest,
            info=DashPartitionInfo.REGULAR_BLOCK,
            dtype=self.param.dtype,
            device=self.param.device)

    def populate_gradient_block_partition(self, srcG: torch.Tensor, dstG: DashMatrixBlock):
        """
            Splits X of shape (R, C) into:
                - full (B, B) blocks of shape (N, B, B)
                - remainder blocks, either (M, B, b) OR (M, b, B)
            Copies the results into block_partition.full and block_partition.rest
        """
        # assert tuple(srcG.shape) == tuple(self.param.shape)
        # ------------------------------
        # Support for stacked normalization layers
        # ------------------------------
        if self.is_norm_layer_stack:
            if self.R_full > 0:
                # srcG_view = srcG.view(self.shape_full)
                dstG.full.copy_(srcG.view(self.shape_full))

            # we know there are no rests for this case
            # # Remainder block: (1, row_rest)
            # if self.row_rest > 0:
            #     G_rest = srcG[self.R_full:]
            #     dstG.rest.copy_(G_rest.view(self.shape_rest))
        else:
            # ------------------------------
            # 2D Logic
            # ------------------------------
            R, C = srcG.shape
            B = self.B

            # ------------------------------
            # 1. Full BxB blocks
            # ------------------------------
            X_full = srcG[:self.R_full, :self.C_full]
            if self.R_full > 0 and self.C_full > 0:
                view_shape = (self.num_row_blocks, B, self.num_col_blocks, B)
                blocks_full = X_full.view(view_shape).transpose(1, 2).reshape(-1, B, B)
                dstG.full.copy_(blocks_full)

            # ------------------------------
            # 2. Remainders (only one direction possible)
            # ------------------------------
            blocks_rest = None

            # Case A: remainder in columns → blocks of shape (num_row_blocks, B, col_rest)
            if self.col_rest > 0:
                right = srcG[:self.R_full, self.C_full:]  # (R_full, col_rest)
                blocks_rest = right.view(self.num_row_blocks, B, self.col_rest)
                dstG.rest.copy_(blocks_rest)

            # Case B: remainder in rows → blocks of shape (num_col_blocks, row_rest, B)
            elif self.row_rest > 0:
                bottom = srcG[self.R_full:, :self.C_full]  # (row_rest, C_full)
                blocks_rest = bottom.view(self.row_rest, self.num_col_blocks, B).transpose(0, 1) # (num_col_blocks, row_rest, B)
                dstG.rest.copy_(blocks_rest)

    def get_preconditioner_blocks_efficiently_grouped(self):
        """
            We provide a symmetric example to understand what this function does

            ####################
            ##### Example 1:
            ####################
            Given a layer `p` of shape `(R, C) = (32_000, 2048)` and a block size `B=1024`, the gradient will be grouped as follows:
                - G_full = (62, 1024, 1024)
                - G_rest = (2, 256, 1024)

            As a consequence, L and R matrices will have the following shapes:
                - L_full = (62, 1024, 1024)
                - L_rest = (2, 256, 256)
                - R_full = (62, 1024, 1024)
                - R_rest = (2, 1024, 1024)

            In order to be efficient, we can group these 4 matrices as follows:
            - group L_full, R_full and R_rest into a single matrix called LRfull_Rrest = (62 x 2 + 2 = 126, 1024, 1024)
            - one matrix L_rest = (2, 256, 256)

            This grouping can be encoded in a DashPartitionInfo object as DashPartitionInfo.FULL_LR_WITH_REST_R_THEN_REST_L, meaning:
            - type_LR_full_rest='R': we group matrices L_full, R_full (will always be the case) and the rest from R
            - type_rest='L': we simply keep the rest from L here

            ####################
            ##### Example 2: the symmetric case of Example 1
            ####################
            Given a layer `p` of shape `(R, C) = (2048, 32_000)` and a block size `B=1024`, the gradient will be grouped as follows:
                - G_full = (62, 1024, 1024)
                - G_rest = (2, 1024, 256) (here is the difference compared to example 1)

            As a consequence, L and R matrices will have the following shapes:
                - L_full = (62, 1024, 1024)
                - L_rest = (2, 1024, 1024) (here is the difference compared to example 1)
                - R_full = (62, 1024, 1024)
                - R_rest = (2, 256, 256) (here is the difference compared to example 1)

            In order to be efficient, we can group these 4 matrices as follows:
            - group L_full, R_full and R_rest into a single matrix called LRfull_Lrest = (62 x 2 + 2 = 126, 1024, 1024)
            - one matrix R_rest = (2, 256, 256)

            This grouping can be encoded in a DashPartitionInfo object as DashPartitionInfo.FULL_LR_WITH_REST_L_THEN_REST_R, meaning:
            - type_LR_full_rest='L': we group matrices L_full, R_full (will always be the case) and the rest from L
            - type_rest='R': we simply keep the rest from R here

            ####################
            ##### Example 3: Stacked Normalization Layers
            ####################
            We will take the Llama-953M model with `N = 33` normalization layers with embedding size `E = 2048` and block size `B = 1024`.
            We want to stack all normalization layers into a single tensor of shape `(N, E) = (33, 2048)` and split each individual row.
            We split each normalization layer into 2 blocks, resulting in a partition of shape `(66, 1024, 1)`, for which we will compute
            the outer product for the `L` matrix of shape `(66, 1024, 1024)`.
        """
        shape_full = self.shape_full
        shape_rest = self.shape_rest

        # ------------------------------
        # Support for stacked normalization layers
        # ------------------------------
        if self.is_norm_layer_stack:
            # For 1D, we typically want square statistics blocks for the segments.
            # Gradient is (N, B) -> Preconditioner Stat is (N, B, B)
            final_shape_full = (shape_full[0], self.B, self.B)

            if shape_rest is not None:
                # Rest is (1, rest) -> Preconditioner Stat is (1, rest, rest)
                rest_dim = shape_rest[1]
                final_shape_rest = (1, rest_dim, rest_dim)
            else:
                final_shape_rest = None

            # We don't use the special grouping flags for 1D, as there are no L/R factors
            return DashMatrixBlock(shape_full=final_shape_full,
                                   shape_rest=final_shape_rest,
                                   info=DashPartitionInfo.REGULAR_BLOCK,
                                   dtype=self.param.dtype,
                                   device=self.param.device)
        # ------------------------------
        # 2D Logic
        # ------------------------------
        if shape_rest is None:
            final_shape_full = (2 * shape_full[0], self.B, self.B)
            final_shape_rest = None
            info = DashPartitionInfo.NO_REST
        else:
            final_shape_full = (2 * shape_full[0] + shape_rest[0], self.B, self.B)
            if shape_rest[1] == self.B:
                info = DashPartitionInfo.REST_R
                final_shape_rest = (shape_rest[0], shape_rest[2], shape_rest[2])
            elif shape_rest[2] == self.B:
                info = DashPartitionInfo.REST_L
                final_shape_rest = (shape_rest[0], shape_rest[1], shape_rest[1])

        return DashMatrixBlock(shape_full=final_shape_full, shape_rest=final_shape_rest, info=info, dtype=self.param.dtype, device=self.param.device)

    def reconstruct_from_blocks(self, block: DashMatrixBlock, out: Tensor=None):
        """
        Reconstructs a matrix of shape (R, C) from:
        - blocks_full: (N, B, B)
        - blocks_rest: (M, B, b) OR (M, b, B)
        """
        full, rest = block.full, block.rest
        B = self.B

        # ------------------------------
        # Support for stacked normalization layers
        # ------------------------------
        if self.is_norm_layer_stack:
            if out is None:
                # out = torch.zeros(self.N, self.R, dtype=full.dtype, device=full.device)
                out = torch.zeros_like(self.param.p)
            out.copy_(full.view_as(out))
            # # Full: (num_blocks, B) -> flatten
            # if full is not None:
            #     out[:self.R_full] = full.reshape(-1)

            ##### we know there are no rests for this case
            # # Rest: (1, row_rest) -> flatten
            # if rest is not None and self.row_rest > 0:
            #     out[self.R_full:] = rest.reshape(-1)
            return out

        if out is None:
            out = torch.zeros(self.R, self.C, dtype=full.dtype, device=full.device)
        # ------------------------------
        # 2D Logic
        # ------------------------------
        # ------------------------------
        # 1. Scatter full blocks
        # ------------------------------
        if full is not None:
            full = full.view(self.num_row_blocks, self.num_col_blocks, B, B)
            full = full.transpose(1, 2)  # (num_row_blocks, B, num_col_blocks, B)
            out[:self.R_full, :self.C_full] = full.reshape(self.R_full, self.C_full)

        # ------------------------------
        # 2. Scatter remainders
        # ------------------------------
        if rest is not None:
            # Case A: remainder in columns (shape: (num_row_blocks, B, col_rest))
            if self.col_rest > 0:
                right = rest.reshape(self.R_full, self.col_rest)
                out[:self.R_full, self.C_full:] = right

            # Case B: remainder in rows (shape: (num_col_blocks, row_rest, B))
            elif self.row_rest > 0:
                bottom = rest.transpose(0, 1)  # shape → (row_rest, C_full)
                bottom = bottom.reshape(self.row_rest, self.C_full)
                out[self.R_full:, :self.C_full] = bottom

        return out