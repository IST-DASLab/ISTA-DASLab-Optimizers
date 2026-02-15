import numpy as np
import torch
from torch import Tensor, bmm

from ...utils.global_cache import GlobalCache
from ..dash_config import DashConfig
from .scalers import DashMatrixScaling

class DashRootInvChebyshev:
    @staticmethod
    @torch.no_grad()
    def _matrix_root_inv_chebyshev(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        matmul_dtype = cfg.matmul_dtype

        match matmul_dtype:
            case torch.float32:
                func = DashRootInvChebyshev._clenshaw_fp32
            case torch.float16:
                func = DashRootInvChebyshev._clenshaw_fp16
            case torch.bfloat16:
                func = DashRootInvChebyshev._clenshaw_bf16
            case _:
                raise RuntimeError(f'NewtonDB is not implemented for dtype {matmul_dtype}')
        # end match-case
        func(inp, out, cfg, root)

    @staticmethod
    @torch.no_grad()
    def _clenshaw_fp32(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        assert inp.ndim == 3

        N, B, _ = inp.shape
        scaling = 2. / DashMatrixScaling.get_matrix_scaling(inp, cfg) # inp.norm(p='fro', dim=(1, 2), keepdim=True)

        S = scaling * inp
        S.diagonal(dim1=-2, dim2=-1).sub_(1.)

        coeffs = DashRootInvChebyshev._get_chebyshev_coefficients(cfg, device=inp.device, root=root)
        d = coeffs.numel() - 1

        Bk2 = torch.zeros(N, B, B, device=inp.device, dtype=torch.float32)
        Bk2.diagonal(dim1=-2, dim2=-1).add_(coeffs[d]) # initialize Bk2 = coeffs[d] * identity

        Bk1 = (2 * coeffs[d]) * S
        Bk1.diagonal(dim1=-2, dim2=-1).add_(coeffs[d - 1])

        for k in range(d - 2, 0, -1): # k from d-2 down to 1
            bmm(S, Bk1, out=out)
            Bk = 2 * out - Bk2
            Bk.diagonal(dim1=-2, dim2=-1).add_(coeffs[k])

            Bk2 = Bk1
            Bk1 = Bk
        # end for k

        bmm(S, Bk1, out=out)
        out.sub_(Bk2)
        out.diagonal(dim1=-2, dim2=-1).add_(coeffs[0])

        del S, Bk1, Bk2, Bk, scaling

        # old code:
        #   for loop
        #       Bk = torch.baddbmm(alpha=2.0, batch1=S, batch2=Bk1, beta=-1.0, input=Bk2, out_dtype=torch.float32)
        #       Bk = 2 * bmm(S, Bk1) - Bk2
        #   after for loop:
        #       Bk = S @ Bk1 - Bk2
        #       Bk.diagonal(dim1=-2, dim2=-1).add_(coeffs[0])
        #       out.copy_(Bk)

    @staticmethod
    @torch.no_grad()
    def _clenshaw_fp16(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        assert inp.ndim == 3

        N, B, _ = inp.shape
        scaling = 2. / DashMatrixScaling.get_matrix_scaling(inp, cfg)  # inp.norm(p='fro', dim=(1, 2), keepdim=True)

        S = scaling * inp
        S.diagonal(dim1=-2, dim2=-1).sub_(1.)

        coeffs = DashRootInvChebyshev._get_chebyshev_coefficients(cfg, device=inp.device, root=root)
        d = coeffs.numel() - 1

        Bk2 = torch.zeros(N, B, B, device=inp.device, dtype=torch.float32)
        Bk2.diagonal(dim1=-2, dim2=-1).add_(coeffs[d])  # initialize Bk2 = coeffs[d] * identity

        Bk1 = (2 * coeffs[d]) * S
        Bk1.diagonal(dim1=-2, dim2=-1).add_(coeffs[d - 1])

        S = S.half()

        for k in range(d - 2, 0, -1):  # k from d-2 down to
            bmm(S, Bk1.half(), out=out, out_dtype=torch.float32)
            Bk = 2 * out - Bk2
            Bk.diagonal(dim1=-2, dim2=-1).add_(coeffs[k])

            Bk2 = Bk1
            Bk1 = Bk
        # end for k

        bmm(S, Bk1.half(), out=out, out_dtype=torch.float32)
        out.sub_(Bk2)
        out.diagonal(dim1=-2, dim2=-1).add_(coeffs[0])

        del S, Bk1, Bk2, Bk, scaling

    @staticmethod
    @torch.no_grad()
    def _clenshaw_bf16(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        assert inp.ndim == 3

        N, B, _ = inp.shape
        scaling = 2. / DashMatrixScaling.get_matrix_scaling(inp, cfg)  # inp.norm(p='fro', dim=(1, 2), keepdim=True)

        S = scaling * inp
        S.diagonal(dim1=-2, dim2=-1).sub_(1.)

        coeffs = DashRootInvChebyshev._get_chebyshev_coefficients(cfg, device=inp.device, root=root)
        d = coeffs.numel() - 1

        Bk2 = torch.zeros(N, B, B, device=inp.device, dtype=torch.float32)
        Bk2.diagonal(dim1=-2, dim2=-1).add_(coeffs[d])  # initialize Bk2 = coeffs[d] * identity

        Bk1 = (2 * coeffs[d]) * S
        Bk1.diagonal(dim1=-2, dim2=-1).add_(coeffs[d - 1])

        S = S.bfloat16()

        for k in range(d - 2, 0, -1):  # k from d-2 down to
            bmm(S, Bk1.bfloat16(), out=out, out_dtype=torch.float32)
            Bk = 2 * out - Bk2
            Bk.diagonal(dim1=-2, dim2=-1).add_(coeffs[k])

            Bk2 = Bk1
            Bk1 = Bk
        # end for k

        bmm(S, Bk1.bfloat16(), out=out, out_dtype=torch.float32)
        out.sub_(Bk2)
        out.diagonal(dim1=-2, dim2=-1).add_(coeffs[0])

        del S, Bk1, Bk2, Bk, scaling

    @staticmethod
    @torch.no_grad()
    def _fit_chebyshev_coefficients(eps: float, d: int, root: int, N: int = 10_000):
        """
            Compute exact Chebyshev coefficients for f(x)=x^(-1/p) on interval [a,b] = [eps, 1+eps].
            Args:
                eps (float): regularization parameter
                d (int): degree of the polynomial
                root (int): compute this inverse root of the matrix
                N (int): number of points in interval [-1, 1] used to fit the polynomial to (default: 10_000)
        """
        # add torch implementation here (from the jupyter-notebook) to fit the parameters in one shot without loops.
        a, b = eps, 1 + eps
        pwr = -1 / root

        j = np.arange(N)
        theta_j = np.pi * (j + 0.5) / N
        t_j = np.cos(theta_j)

        x_j = 0.5 * (b - a) * t_j + 0.5 * (b + a)
        f_j = x_j ** pwr

        c = np.zeros(d + 1)
        for k in range(d + 1):
            c[k] = (2 / N) * np.sum(f_j * np.cos(k * theta_j))
        c[0] /= 2
        return c

    @staticmethod
    @torch.no_grad()
    def _get_chebyshev_coefficients(cfg: DashConfig, device: torch.device, root: int):
        eps = cfg.eps_inv_root
        degree = cfg.cbshv_degree

        cache_categ = 'chebyshev-coeffs'
        cache_key = (eps, degree, root)
        if GlobalCache.contains(category=cache_categ, key=cache_key):
            coeffs = GlobalCache.get(category=cache_categ, key=cache_key)
        else:
            coeffs = DashRootInvChebyshev._fit_chebyshev_coefficients(eps, degree, root)
            coeffs = torch.tensor(coeffs, device=device, dtype=torch.float32)
            GlobalCache.add(category=cache_categ, key=cache_key, item=coeffs)
        return coeffs
