import torch
from torch import Tensor, bmm

from ..dash_config import DashConfig
from .scalers import DashMatrixScaling


class DashRootInvJorge:
    @staticmethod
    @torch.no_grad()
    def _matrix_root_inv_jorge(inp: Tensor, out: Tensor, cfg: DashConfig):
        """
        For this implementation, we have the following mapping between variables in the code and
        variables in the paper:

        inp: represents X_L or X_R in the paper
        out: represents hat(L)_t-1 or hat(R)_t-1 in the paper
             !!!!! the result MUST also be written here !!!!!

        IMPORTANT:
            Since we apply grafting, we do not have to multiply by beta ^ -0.25 as shown in lines
            6 and 8 in the paper because grafting will wipe them out, so the update can avoid it.

            A = (1-beta)/beta * inp
            S = scaling for A
            X = A / S

            a = -1 / 4
            b = 5 / 32

            hat(L/R)_t = hat(L/R)_t-1 @ (eye + a * X + b * X @ X)
            bin_approx = eye + a * X + b * X @ X
            out = out @ bin_approx
        """
        N, B, _ = inp.shape
        idx = torch.arange(B, device=inp.device)
        betaLR = cfg.beta_LR
        beta_frac = (1 - betaLR) / betaLR

        A = beta_frac * inp

        scale = DashMatrixScaling.get_matrix_scaling(A, cfg)
        X = A / scale

        a = -0.25
        b = 0.15625 # 5 / 32
        bin_approx = a * X + b * bmm(X, X) # cannot use baddbmm because consts a and b are N-dim tensors and baddbmm accepts only scalars for a and b
        bin_approx[:, idx, idx] += 1

        bmm(input=out, mat2=bin_approx, out=X)
        out.copy_((betaLR ** (-0.25)) * X)

        del bin_approx
