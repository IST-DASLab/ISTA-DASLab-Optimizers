import enum
import torch
from torch import Tensor, bmm, baddbmm
import torch.distributed as dist
from dataclasses import dataclass
import numpy as np

from ..utils.global_cache import GlobalCache
from .dash_layerwise_block_partitioner import DashMatrixBlock
from .dash_configs import (
    DashInverseRootMethodType,
    DashConfig,
    DashEVDHeuristic,
    DashNDBReturnType,
    DashMatrixScalingType,
)
from ..utils.newton_schulz_triton import ns_line_1

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





####################################################################################################
#################### EIGEN-VALUE-DECOMPOSITION (EVD)
####################################################################################################
class DashBlockRootInvEVD:
    @staticmethod
    @torch.no_grad()
    def _matrix_root_inv_evd(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        """
            This method uses eps in eigh and then again in the heuristic, exactly as the original DistributedShampoo (see A_ridge in their implementation).
        """
        heuristic = cfg.evd_heuristic
        assert heuristic is not None

        block_ranks = None # this will be set for RELU heuristic and returned at the end of the function if not None

        _, B, _ = inp.shape
        eps = cfg.eps_inv_root

        eye = torch.eye(B, dtype=inp.dtype, device=inp.device)
        L, Q = torch.linalg.eigh(inp + eps * eye)  # automatic broadcasting of eye to batch matrix
        # here, L already contains the eps and we should subtract eps from L to avoid adding regularization 2 times (one in EVD call and one in match-case),
        # but this is how DistributedShampoo implements it and we follow their implementation to compare apples to apples
        match heuristic:
            case DashEVDHeuristic.ABS:
                L.sub_(eps).abs_()
            case DashEVDHeuristic.ABS_ADD:
                L.sub_(eps).abs_().add_(eps)
            case DashEVDHeuristic.RELU:
                L.sub_(eps)
                # torch.nn.functional.relu(L, inplace=True)
                L[L < eps] = 0
                if (dist.is_initialized() and dist.get_rank() == 0) or (not dist.is_initialized()): # make sure we are on master process
                    block_ranks = (L > 0).sum(dim=1)

            case DashEVDHeuristic.SHAMPOO:
                """
                This heuristic is presented in section 3.2.1 (1) of Distributed Shampoo paper: https://arxiv.org/pdf/2309.06497#page=15 (gray box):
                L_min = min_i lambda_i
                L_new = L - min(L_min, 0) * ones + epsilon * ones
                      = L + (epsilon - min(L_min, 0)) * ones
                """
                L_min = L.min(dim=1, keepdim=True).values
                L.add_(eps - L_min.clamp(max=0))

        if heuristic == DashEVDHeuristic.RELU:
            L[L > 0] **= (-1. / root) # raise to the -1/root only the non-zero values
            out.copy_(Q @ torch.diag_embed(L) @ Q.transpose(1, 2))
        else:
            out.copy_(Q @ torch.diag_embed(L.pow(-1. / root)) @ Q.transpose(1, 2))

        del L, Q, eye

        if block_ranks is not None:
            return block_ranks





####################################################################################################
#################### CHEBYSHEV
####################################################################################################
class DashBlockRootInvChebyshev:
    @staticmethod
    @torch.no_grad()
    def _matrix_root_inv_chebyshev(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        matmul_dtype = cfg.matmul_dtype

        match matmul_dtype:
            case torch.float32:
                func = DashBlockRootInvChebyshev._clenshaw_fp32
            case torch.float16:
                func = DashBlockRootInvChebyshev._clenshaw_fp16
            case torch.bfloat16:
                func = DashBlockRootInvChebyshev._clenshaw_bf16
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

        coeffs = DashBlockRootInvChebyshev._get_chebyshev_coefficients(cfg, device=inp.device, root=root)
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

        coeffs = DashBlockRootInvChebyshev._get_chebyshev_coefficients(cfg, device=inp.device, root=root)
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

        coeffs = DashBlockRootInvChebyshev._get_chebyshev_coefficients(cfg, device=inp.device, root=root)
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
            coeffs = DashBlockRootInvChebyshev._fit_chebyshev_coefficients(eps, degree, root)
            coeffs = torch.tensor(coeffs, device=device, dtype=torch.float32)
            GlobalCache.add(category=cache_categ, key=cache_key, item=coeffs)
        return coeffs





####################################################################################################
#################### COUPLED NEWTON FROM SHAMPOO 2020 Appendix i: https://arxiv.org/pdf/2002.09018
####################################################################################################
class DashBlockRootInvCoupledNewton:
    @staticmethod
    @torch.no_grad()
    def _matrix_root_inv_coupled_newton(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        """
        Compute inverse root via square root. Currently supports only root in {2, 4}
        """
        matmul_dtype = cfg.matmul_dtype

        match matmul_dtype:
            case torch.float32:
                func = DashBlockRootInvCoupledNewton._coupled_newton_fp32
            case torch.float16:
                func = DashBlockRootInvCoupledNewton._coupled_newton_fp16
            case torch.bfloat16:
                func = DashBlockRootInvCoupledNewton._coupled_newton_bf16
            case _:
                raise RuntimeError(f'CoupledNewton is not implemented for dtype {matmul_dtype}')
        # end match-case
        func(inp, out, cfg, root)

    @staticmethod
    @torch.no_grad()
    def _coupled_newton_fp32(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        """
        This function is "matrix_inverse_root_newton" and taken from DistributedShampoo at:
        https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/preconditioner/matrix_functions.py#L359
        (the line number might be subject to change if the main branch changes)

        Taken from https://github.com/facebookresearch/optimizers/blob/main/matrix_functions.py#L262

        Compute matrix inverse root using coupled inverse Newton iteration.

        alpha <- -1 / p
        X <- 1/c * I
        M <- 1/c^p * A
        repeat until convergence
            M' <- (1 - alpha) * I + alpha * M
            X <- X * M'
            M <- M'^p * M

        where c = (2 |A|_F / (p + 1))^{1/p}. This ensures that |A|_2 <= |A|_F < (p + 1) c^p, which guarantees convergence.
        We will instead use z = (p + 1) / (2 * |A|_F).
        """
        N, B, _ = inp.shape
        tolerance = cfg.cn_tolerance
        eps = cfg.eps_inv_root

        alpha = -1. / root
        identity = torch.eye(B, dtype=inp.dtype, device=inp.device).expand(N, B, B)

        # add regularization
        A_reg = inp.add(identity, alpha=eps)

        # initialize matrices

        scale = DashMatrixScaling.get_matrix_scaling(inp, cfg)
        z = (root + 1) / (2 * scale)
        X = z ** (-alpha) * identity
        M = z * A_reg
        error = torch.dist(M, identity, p=torch.inf)

        for it in range(cfg.newton_steps):
            Meye = M.mul(alpha).add_(identity, alpha=1-alpha) # convex combination of M and identity
            X = bmm(X, Meye)

            sqMeye = bmm(Meye, Meye) # compute square
            if root == 2:
                M = bmm(sqMeye, M)
            elif root == 4:
                M = bmm(sqMeye, bmm(sqMeye, M)) # M = sqMeye @ sqMeye @ M = Meye ^ 4 @ M
            else:
                raise RuntimeError(f'CoupledNewton (CN) currently supports only roots 2 and 4, but got root={root}!')

            error = torch.dist(M, identity, p=torch.inf)
            if error < tolerance:
                break
        # end for it
        out.copy_(X)

    @staticmethod
    @torch.no_grad()
    def _coupled_newton_fp16(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        """
            See documentation of _coupled_newton_fp32
        """
        N, B, _ = inp.shape
        tolerance = cfg.cn_tolerance
        eps = cfg.eps_inv_root

        alpha = -1. / root
        identity = torch.eye(B, dtype=inp.dtype, device=inp.device).expand(N, B, B)

        # add regularization
        A_reg = inp.add(identity, alpha=eps)

        # initialize matrices

        scale = DashMatrixScaling.get_matrix_scaling(inp, cfg)
        z = (root + 1) / (2 * scale)
        X = z ** (-alpha) * identity
        M32 = z * A_reg
        # error = torch.dist(M32, identity, p=torch.inf)

        Npow2_16 = torch.empty_like(inp, dtype=torch.float16)
        if root == 4:
            Npow4_16 = torch.empty_like(inp, dtype=torch.float16)

        for it in range(cfg.newton_steps):
            N32 = M32.mul(alpha).add_(identity, alpha=1 - alpha) # the convex combination of M and identity is called N
            X = bmm(X, N32) # FP32

            M16 = M32.half()
            N16 = N32.half()

            ns_line_1(A=N16, out=Npow2_16)

            if root == 2:
                bmm(out=M32, input=Npow2_16, mat2=M16, out_dtype=torch.float32)
            elif root == 4:
                ns_line_1(A=Npow2_16, out=Npow4_16) # M4_32 = Meye ^ 4
                bmm(out=M32, input=Npow4_16, mat2=M16, out_dtype=torch.float32)

                ### original version which we know it works:
                # M = bmm(sqMeye, bmm(sqMeye, M))  # M = sqMeye @ sqMeye @ M = Meye ^ 4 @ M, where sqMeye was computed in fp32
            else:
                raise RuntimeError(f'CoupledNewton (CN) currently supports only roots 2 and 4, but got root={root}!')

            error = torch.dist(M32, identity, p=torch.inf)
            if error < tolerance:
                break
        # end for it
        out.copy_(X)

    @staticmethod
    @torch.no_grad()
    def _coupled_newton_bf16(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        """
            See documentation of _coupled_newton_fp32
        """
        N, B, _ = inp.shape
        tolerance = cfg.cn_tolerance
        eps = cfg.eps_inv_root

        alpha = -1. / root
        identity = torch.eye(B, dtype=inp.dtype, device=inp.device).expand(N, B, B)

        # add regularization
        A_reg = inp.add(identity, alpha=eps)

        # initialize matrices

        scale = DashMatrixScaling.get_matrix_scaling(inp, cfg)
        z = (root + 1) / (2 * scale)
        X = z ** (-alpha) * identity
        M32 = z * A_reg
        # error = torch.dist(M32, identity, p=torch.inf)

        Npow2_16 = torch.empty_like(inp, dtype=torch.bfloat16)
        if root == 4:
            Npow4_16 = torch.empty_like(inp, dtype=torch.bfloat16)

        for it in range(cfg.newton_steps):
            N32 = M32.mul(alpha).add_(identity, alpha=1 - alpha)  # the convex combination of M and identity is called N
            X = bmm(X, N32)  # FP32

            M16 = M32.bfloat16()
            N16 = N32.bfloat16()

            ns_line_1(A=N16, out=Npow2_16)

            if root == 2:
                bmm(out=M32, input=Npow2_16, mat2=M16, out_dtype=torch.float32)
            elif root == 4:
                ns_line_1(A=Npow2_16, out=Npow4_16)  # M4_32 = Meye ^ 4
                bmm(out=M32, input=Npow4_16, mat2=M16, out_dtype=torch.float32)

                ### original version which we know it works:
                # M = bmm(sqMeye, bmm(sqMeye, M))  # M = sqMeye @ sqMeye @ M = Meye ^ 4 @ M, where sqMeye was computed in fp32
            else:
                raise RuntimeError(f'CoupledNewton (CN) currently supports only roots 2 and 4, but got root={root}!')

            error = torch.dist(M32, identity, p=torch.inf)
            if error < tolerance:
                break
        # end for it
        out.copy_(X)





####################################################################################################
#################### NEWTON-DB
####################################################################################################
class DashBlockRootInvNewtonDB:
    @staticmethod
    @torch.no_grad()
    def _matrix_root_inv_newton_db(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        """
        Compute inverse root via square root. Currently supports only root in {2, 4}
        """
        matmul_dtype = cfg.matmul_dtype

        match matmul_dtype:
            case torch.float32:
                func = DashBlockRootInvNewtonDB._newton_db_fp32_optimized
            case torch.float16:
                func = DashBlockRootInvNewtonDB._newton_db_fp16_optimized
            case torch.bfloat16:
                func = DashBlockRootInvNewtonDB._newton_db_bf16_optimized
            case _:
                raise RuntimeError(f'NewtonDB is not implemented for dtype {matmul_dtype}')

        # eye = torch.eye(inp.shape[1], dtype=inp.dtype, device=inp.device)
        # inp_reg = inp + cfg.eps_inv_root * eye
        # del eye

        scale = DashMatrixScaling.get_matrix_scaling(inp, cfg)

        if root == 2:
            func(inp=inp, out=out, cfg=cfg, scale=scale, return_type=DashNDBReturnType.INV_SQRT)
        elif root == 4:
            inp_sqrt = torch.empty_like(out) # create temp tensor for the square root
            func(inp=inp, out=inp_sqrt, cfg=cfg, scale=scale, return_type=DashNDBReturnType.SQRT)
            func(inp=inp_sqrt, out=out, cfg=cfg, scale=scale.sqrt(), return_type=DashNDBReturnType.INV_SQRT)
            del inp_sqrt
        else:
            raise RuntimeError(f'NewtonDB implements logic only for inverse 2nd and 4th roots, but got root={root}!')

    @staticmethod
    @torch.no_grad()
    def _newton_db_fp32_optimized(inp: Tensor, out: Tensor, cfg: DashConfig, scale: Tensor, return_type: DashNDBReturnType):
        assert out is not None
        N, B, _ = inp.shape  # N=number blocks (batches), B = block size
        sqrt_scale = scale.sqrt()

        idx = torch.arange(B, device=inp.device)

        A = inp / scale
        E = -0.5 * A
        E[:, idx, idx] += 1.5 # after this line, we have E = 1.5 I - 0.5 A
        Y = A @ E
        Z = E.clone()
        tmp = torch.empty_like(inp)

        for s in range(1, cfg.newton_steps):
            bmm(out=E, input=Z, mat2=Y)  # E = ZY
            E.mul_(-0.5)
            E[:, idx, idx] += 1.5

            bmm(out=tmp, input=Y, mat2=E)
            Y.copy_(tmp)

            bmm(out=tmp, input=E, mat2=Z)
            Z.copy_(tmp)
        # end for steps

        match return_type:
            case DashNDBReturnType.SQRT:
                out.copy_(Y).mul_(sqrt_scale)
            case DashNDBReturnType.INV_SQRT:
                out.copy_(Z).div_(sqrt_scale)
            case _:
                raise RuntimeError(f'Got unknown value of return_type: {return_type}')

        del Y, Z, E, tmp, scale, sqrt_scale

    # @staticmethod
    # @torch.no_grad()
    # def _newton_db_fp16_optimized(inp: Tensor, out: Tensor, cfg: ShampooConfig, scale: Tensor, return_type: NDBReturnType):
    #     """
    #     Keeps Y and Z in fp32 and performs only matmul in 16-bit
    #     """
    #     assert out is not None
    #     N, B, _ = inp.shape  # N=number blocks (batches), B = block size
    #     sqrt_scale = scale.sqrt()
    #
    #     idx = torch.arange(B, device=inp.device)
    #
    #     A = inp / scale
    #     E32 = -0.5 * A
    #     E32[:, idx, idx] += 1.5  # after this line, we have E = 1.5 I - 0.5 A
    #     Y32 = A @ E32
    #     Z32 = E32.clone()
    #     tmp = torch.empty_like(inp)
    #
    #     for s in range(1, cfg.newton_steps):
    #         Y16 = Y32.half()
    #         Z16 = Z32.half()
    #
    #         bmm(out=E32, input=Z16, mat2=Y16, out_dtype=torch.float32)  # E = ZY
    #         E32.mul_(-0.5)
    #         E32[:, idx, idx] += 1.5
    #
    #         E16 = E32.half()
    #
    #         bmm(out=tmp, input=Y16, mat2=E16, out_dtype=torch.float32)
    #         Y32.copy_(tmp)
    #
    #         bmm(out=tmp, input=E16, mat2=Z16, out_dtype=torch.float32)
    #         Z32.copy_(tmp)
    #     # end for steps
    #
    #     match return_type:
    #         case NDBReturnType.SQRT:
    #             out.copy_(Y32).mul_(sqrt_scale)
    #         case NDBReturnType.INV_SQRT:
    #             out.copy_(Z32).div_(sqrt_scale)
    #         case _:
    #             raise RuntimeError(f'Got unknown value of return_type: {return_type}')
    #
    #     del Y32, Y16, Z32, Z16, E32, tmp, scale, sqrt_scale
    #
    # @staticmethod
    # @torch.no_grad()
    # def _newton_db_bf16_optimized(inp: Tensor, out: Tensor, cfg: ShampooConfig, scale: Tensor, return_type: NDBReturnType):
    #     """
    #     Keeps Y and Z in fp32 and performs only matmul in 16-bit
    #     """
    #     assert out is not None
    #     N, B, _ = inp.shape  # N=number blocks (batches), B = block size
    #     sqrt_scale = scale.sqrt()
    #
    #     idx = torch.arange(B, device=inp.device)
    #
    #     A = inp / scale
    #     E32 = -0.5 * A
    #     E32[:, idx, idx] += 1.5  # after this line, we have E = 1.5 I - 0.5 A
    #     Y32 = A @ E32
    #     Z32 = E32.clone()
    #     tmp = torch.empty_like(inp)
    #
    #     for s in range(1, cfg.newton_steps):
    #         Y16 = Y32.bfloat16()
    #         Z16 = Z32.bfloat16()
    #
    #         bmm(out=E32, input=Z16, mat2=Y16, out_dtype=torch.float32)  # E = ZY
    #         E32.mul_(-0.5)
    #         E32[:, idx, idx] += 1.5
    #
    #         E16 = E32.bfloat16()
    #
    #         bmm(out=tmp, input=Y16, mat2=E16, out_dtype=torch.float32)
    #         Y32.copy_(tmp)
    #
    #         bmm(out=tmp, input=E16, mat2=Z16, out_dtype=torch.float32)
    #         Z32.copy_(tmp)
    #     # end for steps
    #
    #     match return_type:
    #         case NDBReturnType.SQRT:
    #             out.copy_(Y32).mul_(sqrt_scale)
    #         case NDBReturnType.INV_SQRT:
    #             out.copy_(Z32).div_(sqrt_scale)
    #         case _:
    #             raise RuntimeError(f'Got unknown value of return_type: {return_type}')
    #
    #     del Y32, Y16, Z32, Z16, E32, tmp, scale, sqrt_scale

    @staticmethod
    @torch.no_grad()
    def _newton_db_fp16_optimized(inp: Tensor, out: Tensor, cfg: DashConfig, scale: Tensor, return_type: DashNDBReturnType):
        """
        Keeps Y and Z in 16-bit
        """
        assert out is not None
        N, B, _ = inp.shape  # N=number blocks (batches), B = block size
        sqrt_scale = scale.sqrt()

        idx = torch.arange(B, device=inp.device)

        A = inp / scale
        E32 = -0.5 * A
        E32[:, idx, idx] += 1.5  # after this line, we have E = 1.5 I - 0.5 A
        Y16 = (A @ E32).half()
        Z16 = E32.half()
        tmp = torch.empty_like(inp)

        for s in range(1, cfg.newton_steps):
            bmm(out=E32, input=Z16, mat2=Y16, out_dtype=torch.float32)  # E = ZY
            E32.mul_(-0.5)
            E32[:, idx, idx] += 1.5

            E16 = E32.half()
            bmm(out=tmp, input=Y16, mat2=E16, out_dtype=torch.float32)
            Y16.copy_(tmp)

            bmm(out=tmp, input=E16, mat2=Z16, out_dtype=torch.float32)
            Z16.copy_(tmp)
        # end for steps

        match return_type:
            case DashNDBReturnType.SQRT:
                out.copy_(Y16).mul_(sqrt_scale)
            case DashNDBReturnType.INV_SQRT:
                out.copy_(Z16).div_(sqrt_scale)
            case _:
                raise RuntimeError(f'Got unknown value of return_type: {return_type}')

        del Y16, Z16, E32, tmp, scale, sqrt_scale

    @staticmethod
    @torch.no_grad()
    def _newton_db_bf16_optimized(inp: Tensor, out: Tensor, cfg: DashConfig, scale: Tensor, return_type: DashNDBReturnType):
        """
        Keeps Y and Z in 16-bit
        """
        assert out is not None
        N, B, _ = inp.shape  # N=number blocks (batches), B = block size
        sqrt_scale = scale.sqrt()

        idx = torch.arange(B, device=inp.device)

        A = inp / scale
        E32 = -0.5 * A
        E32[:, idx, idx] += 1.5  # after this line, we have E = 1.5 I - 0.5 A
        Y16 = (A @ E32).bfloat16()
        Z16 = E32.bfloat16()
        tmp32 = torch.empty_like(inp)

        for s in range(1, cfg.newton_steps):
            bmm(out=E32, input=Z16, mat2=Y16, out_dtype=torch.float32)  # E = ZY
            E32.mul_(-0.5)
            E32[:, idx, idx] += 1.5

            E16 = E32.bfloat16()
            bmm(out=tmp32, input=Y16, mat2=E16, out_dtype=torch.float32)
            Y16.copy_(tmp32)

            bmm(out=tmp32, input=E16, mat2=Z16, out_dtype=torch.float32)
            Z16.copy_(tmp32)
        # end for steps

        match return_type:
            case DashNDBReturnType.SQRT:
                out.copy_(Y16).mul_(sqrt_scale)
            case DashNDBReturnType.INV_SQRT:
                out.copy_(Z16).div_(sqrt_scale)
            case _:
                raise RuntimeError(f'Got unknown value of return_type: {return_type}')

        del Y16, Z16, E32, tmp32, scale, sqrt_scale





####################################################################################################
#################### JORGE FROM Algorithm 1 in https://openreview.net/pdf?id=hdCDVSPQ7v
#################### This is numerically instable, it's on our TODO list to fix.
####################################################################################################
class DashBlockRootInvJorge:
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





####################################################################################################
#################### PUBLIC INTERFACE
####################################################################################################
class DashBlockRootInvertor:
    @staticmethod
    @torch.no_grad()
    def invert(Xin: DashMatrixBlock, Xout: DashMatrixBlock, cfg: DashConfig, root: int):
        method = cfg.inv_root_method
        match method:
            case DashInverseRootMethodType.EVD:
                func = DashBlockRootInvEVD._matrix_root_inv_evd
            case DashInverseRootMethodType.CN:
                func = DashBlockRootInvCoupledNewton._matrix_root_inv_coupled_newton
            case DashInverseRootMethodType.JORGE:
                func = DashBlockRootInvJorge._matrix_root_inv_jorge
            case DashInverseRootMethodType.CBSHV:
                func = DashBlockRootInvChebyshev._matrix_root_inv_chebyshev
            case DashInverseRootMethodType.NDB:
                func = DashBlockRootInvNewtonDB._matrix_root_inv_newton_db
            case _:
                raise RuntimeError(f'Invalid method: {method}')
        # end match-case

        ret = func(inp=Xin.full, out=Xout.full, cfg=cfg, root=root)
        if Xin.has_rest and Xout.has_rest: # run this on a new stream?
            func(inp=Xin.rest, out=Xout.rest, cfg=cfg, root=root)

        if ret is not None:
            return ret
