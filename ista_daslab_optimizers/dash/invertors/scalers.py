import torch
from torch import Tensor, bmm

from ..dash_config import DashConfig
from ..types import DashMatrixScalingType


class DashMatrixScaling:
    @staticmethod
    @torch.no_grad()
    def get_matrix_scaling(A: Tensor, cfg: DashConfig):
        scaling_type = cfg.matrix_scaling_type

        match scaling_type:
            case DashMatrixScalingType.POWER_ITER:
                scale, _ = DashMatrixScaling.max_eigval_power_iter(A, num_iters=cfg.matrix_scaling_pi_steps)
                return scale * cfg.matrix_scaling_const # multiply by the constant here
            case DashMatrixScalingType.POWER_ITER_MULTI:
                A16 = A.bfloat16()
                scale, _ = DashMatrixScaling.max_eigval_power_iter_multi(A16, num_iters=cfg.matrix_scaling_pi_steps, num_vecs=16)
                # if torch.isnan(scale).sum() > 0:
                #     print(f'Found NaN in PIM-scaling!!!')
                del A16
                return scale * cfg.matrix_scaling_const # multiply by the constant here
            case DashMatrixScalingType.FRO:
                scale = A.norm(p='fro', dim=(1, 2), keepdim=True)
                return scale # do not scale the frobenius norm by the constant set by user
            case _:
                raise RuntimeError(f'Received unknown NDBScalingType: {scaling_type}')

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

        max_vals, max_indices = eig_vals_all.max(dim=1) # (N, num_vecs).max(dim=1) => (N,)

        idx_expanded = max_indices.view(N, 1, 1).expand(N, B, 1) # (N, B, 1)
        best_vals = max_vals.view(N, 1, 1)
        best_vecs = v.gather(2, idx_expanded) # (N, B, 1)

        best_vals = best_vals.to(torch.float32)

        return best_vals, best_vecs
