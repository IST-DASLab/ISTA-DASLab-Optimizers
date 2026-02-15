import torch
from torch import Tensor, bmm

from ...utils.newton_schulz_triton import ns_line_1
from ..dash_config import DashConfig
from .scalers import DashMatrixScaling

class DashRootInvCoupledNewton:
    @staticmethod
    @torch.no_grad()
    def _matrix_root_inv_coupled_newton(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        """
        Compute inverse root via square root. Currently supports only root in {2, 4}
        """
        matmul_dtype = cfg.matmul_dtype

        match matmul_dtype:
            case torch.float32:
                func = DashRootInvCoupledNewton._coupled_newton_fp32
            case torch.float16:
                func = DashRootInvCoupledNewton._coupled_newton_fp16
            case torch.bfloat16:
                func = DashRootInvCoupledNewton._coupled_newton_bf16
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
