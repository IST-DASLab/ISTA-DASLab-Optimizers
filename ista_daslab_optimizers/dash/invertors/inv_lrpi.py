import torch
from torch import Tensor, bmm

from ..dash_config import DashConfig
from .scalers import DashMatrixScaling


class DashRootInvLowRankPowerIter:
    @staticmethod
    @torch.no_grad()
    def _matrix_root_inv_low_rank_power_iter(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        """
            This way of computing the inverse 4th power is based on the eigen-value decomposition of A:
            A = sum_i=1^n lambda_i q_i @ q_i^T, where (lambda_i, q_i) is the i-th eigen-value/vector. A^-1/4 uses lambda_i^-1/4

            We can efficiently compute the largest eigen-value/vector using Power-Iteration. Using this sum of outer products of eigen-vectors scaled by
            eigen-values, we can call power iteration r times, where r is the desired rank:

            for i in range(r):
                lambda_i, q_i = power_iter(A)
                A_IFR += lambda_i ^ -1/4 * q_i @ q_i^T
                A -= lambda_i * q_i @ q_i^T
            return A_IFR
        """
        rank = cfg.lrpi_rank

        fro = inp.norm(p='fro', dim=(1, 2), keepdim=True)
        A = inp / fro
        # A = inp.clone() # this will be modified in place
        # A = torch.empty_like(inp)
        # A.copy_(inp)
        # A.diagonal(dim1=-2, dim2=-1).add_(cfg.eps_inv_root)

        out.zero_()

        # should_break = A.shape[0] > 50
        # should_break = False

        for i in range(rank):
            # eig_vals, eig_vecs = MatrixScaling.max_eigval_power_iter_multi(A, num_iters=cfg.lrpi_steps, num_vecs=16)
            eig_vals, eig_vecs = DashMatrixScaling.max_eigval_power_iter(A, num_iters=cfg.lrpi_steps)

            # if (eig_vals < 0).sum() > 0:
            #     breakpoint()
            # if torch.isnan(eig_vecs).sum() > 0:
            #     breakpoint()
            # if should_break: breakpoint()

            # eig_vals.mul_(2)

            outer = bmm(eig_vecs, eig_vecs.transpose(1, 2))
            # if should_break: breakpoint()
            out.add_((eig_vals ** (-1. / root)) * outer)
            # if should_break: breakpoint()
            A.sub_(eig_vals * outer)
            # if should_break: breakpoint(); print(f'')
        # for i
        del A
