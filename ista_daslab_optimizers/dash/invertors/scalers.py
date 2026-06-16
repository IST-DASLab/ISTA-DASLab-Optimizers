import torch
from torch import Tensor, bmm

from ..dash_config import DashConfig
from ..types import DashMatrixScalingType


class DashMatrixScaling:
    @staticmethod
    @torch.no_grad()
    def get_matrix_scaling(A: Tensor, cfg: DashConfig, eps: float=1e-8):
        scaling_type = cfg.matrix_scaling_type

        B, N, _ = A.shape
        idx = torch.arange(N).to(A.device)
        A16 = A.bfloat16()

        ##### START EXPLANATION FOR DAMPING
        """
            We need to dampen the bf16 version of A because in modded-nanogpt some layers have gradient that is completely zero at the first step.
            When inputting a zero matrix block DASH invertor, the power iteration performs 0/0 operation which results in NaNs.
            These NaNs will then propagate to the model and therefore at the second optimization step the loss became NaN. 
        """
        A16[:, idx, idx] += eps
        ##### END EXPLANATION FOR DAMPING

        if scaling_type == DashMatrixScalingType.POWER_ITER:
            scale, _ = DashMatrixScaling.max_eigval_power_iter(A16, num_iters=cfg.matrix_scaling_pi_steps)
            multiplier = cfg.matrix_scaling_const
        elif scaling_type == DashMatrixScalingType.POWER_ITER_MULTI:
            scale, _ = DashMatrixScaling.max_eigval_power_iter_multi(A16, num_iters=cfg.matrix_scaling_pi_steps, num_vecs=16)
            multiplier = cfg.matrix_scaling_const
        elif scaling_type == DashMatrixScalingType.FRO:
            scale = A.norm(p='fro', dim=(1, 2), keepdim=True)
            multiplier = 1
        else:
            raise RuntimeError(f'Received unknown NDBScalingType: {scaling_type}')

        return scale * multiplier

    @staticmethod
    @torch.no_grad()
    def max_eigval_power_iter(A: Tensor, num_iters: int):
        N, B, _ = A.shape

        # random initial vector
        v = torch.randn(N, B, 1, device=A.device, dtype=A.dtype)
        v = v / v.norm(p=2, dim=(1, 2), keepdim=True)

        Av = torch.empty_like(v)

        for it in range(num_iters):
            bmm(A, v, out=Av)
            v = Av / Av.norm(p=2, dim=(1, 2), keepdim=True)

        eigvals = bmm(v.transpose(1, 2), bmm(A, v))
        return eigvals, v

    @staticmethod
    @torch.no_grad()
    def max_eigval_power_iter_multi(A: Tensor, num_iters: int, num_vecs: int):
        """
        Performs Power-Iteration with `num_vecs` in parallel to minimize the chances
        of converging to an eigen-vector that has a corresponding eigen-value smaller
        than the largest eigen-value.
        """
        N, B, _ = A.shape

        # random initial vector
        v = torch.randn(N, B, num_vecs, device=A.device, dtype=A.dtype)
        v = v / v.norm(p=2, dim=1, keepdim=True)

        Av = torch.empty_like(v)

        for it in range(num_iters):
            bmm(A, v, out=Av) # (N, B, B) @ (N, B, num_vecs) => (N, B, num_vecs)
            v = Av / Av.norm(p=2, dim=1, keepdim=True)
        # end for it

        bmm(A, v, out=Av)
        eig_vals_all = (v * Av).sum(dim=1) # (N, B, num_vecs).sum(dim=1) => (N, num_vecs)
        print(eig_vals_all)
        max_vals, max_indices = eig_vals_all.max(dim=1) # (N, num_vecs).max(dim=1) => (N,)

        idx_expanded = max_indices.view(N, 1, 1).expand(N, B, 1) # (N, B, 1)
        best_vals = max_vals.view(N, 1, 1)
        best_vecs = v.gather(2, idx_expanded) # (N, B, 1)

        best_vals = best_vals.to(torch.float32)

        return best_vals, best_vecs
