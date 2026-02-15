import torch
from torch import Tensor, bmm

from ..dash_config import DashConfig
from .scalers import DashMatrixScaling
from ..types import DashNdbReturnType


class DashRootInvNewtonDB:
    @staticmethod
    @torch.no_grad()
    def _matrix_root_inv_newton_db(inp: Tensor, out: Tensor, cfg: DashConfig, root: int):
        """
        Compute inverse root via square root. Currently supports only root in {2, 4}
        """
        matmul_dtype = cfg.matmul_dtype

        match matmul_dtype:
            case torch.float32:
                func = DashRootInvNewtonDB._newton_db_fp32_optimized
            case torch.float16:
                func = DashRootInvNewtonDB._newton_db_fp16_optimized
            case torch.bfloat16:
                func = DashRootInvNewtonDB._newton_db_bf16_optimized
            case _:
                raise RuntimeError(f'NewtonDB is not implemented for dtype {matmul_dtype}')

        # eye = torch.eye(inp.shape[1], dtype=inp.dtype, device=inp.device)
        # inp_reg = inp + cfg.eps_inv_root * eye
        # del eye

        scale = DashMatrixScaling.get_matrix_scaling(inp, cfg)

        if root == 2:
            func(inp=inp, out=out, cfg=cfg, scale=scale, return_type=DashNdbReturnType.INV_SQRT)
        elif root == 4:
            inp_sqrt = torch.empty_like(out) # create temp tensor for the square root
            func(inp=inp,      out=inp_sqrt, cfg=cfg, scale=scale,        return_type=DashNdbReturnType.SQRT)
            func(inp=inp_sqrt, out=out,      cfg=cfg, scale=scale.sqrt(), return_type=DashNdbReturnType.INV_SQRT)
            del inp_sqrt
        else:
            raise RuntimeError(f'NewtonDB implements logic only for inverse 2nd and 4th roots, but got root={root}!')

    @staticmethod
    @torch.no_grad()
    def _newton_db_fp32_optimized(inp: Tensor, out: Tensor, cfg: DashConfig, scale: Tensor, return_type: DashNdbReturnType):
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
            case DashNdbReturnType.SQRT:
                out.copy_(Y).mul_(sqrt_scale)
            case DashNdbReturnType.INV_SQRT:
                out.copy_(Z).div_(sqrt_scale)
            case _:
                raise RuntimeError(f'Got unknown value of return_type: {return_type}')

        del Y, Z, E, tmp, scale, sqrt_scale

    # @staticmethod
    # @torch.no_grad()
    # def _newton_db_fp16_optimized(inp: Tensor, out: Tensor, cfg: DashConfig, scale: Tensor, return_type: DashNdbReturnType):
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
    #         case DashNdbReturnType.SQRT:
    #             out.copy_(Y32).mul_(sqrt_scale)
    #         case DashNdbReturnType.INV_SQRT:
    #             out.copy_(Z32).div_(sqrt_scale)
    #         case _:
    #             raise RuntimeError(f'Got unknown value of return_type: {return_type}')
    #
    #     del Y32, Y16, Z32, Z16, E32, tmp, scale, sqrt_scale
    #
    # @staticmethod
    # @torch.no_grad()
    # def _newton_db_bf16_optimized(inp: Tensor, out: Tensor, cfg: DashConfig, scale: Tensor, return_type: DashNdbReturnType):
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
    #         case DashNdbReturnType.SQRT:
    #             out.copy_(Y32).mul_(sqrt_scale)
    #         case DashNdbReturnType.INV_SQRT:
    #             out.copy_(Z32).div_(sqrt_scale)
    #         case _:
    #             raise RuntimeError(f'Got unknown value of return_type: {return_type}')
    #
    #     del Y32, Y16, Z32, Z16, E32, tmp, scale, sqrt_scale

    @staticmethod
    @torch.no_grad()
    def _newton_db_fp16_optimized(inp: Tensor, out: Tensor, cfg: DashConfig, scale: Tensor, return_type: DashNdbReturnType):
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
            case DashNdbReturnType.SQRT:
                out.copy_(Y16).mul_(sqrt_scale)
            case DashNdbReturnType.INV_SQRT:
                out.copy_(Z16).div_(sqrt_scale)
            case _:
                raise RuntimeError(f'Got unknown value of return_type: {return_type}')

        del Y16, Z16, E32, tmp, scale, sqrt_scale

    @staticmethod
    @torch.no_grad()
    def _newton_db_bf16_optimized(inp: Tensor, out: Tensor, cfg: DashConfig, scale: Tensor, return_type: DashNdbReturnType):
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
            case DashNdbReturnType.SQRT:
                out.copy_(Y16).mul_(sqrt_scale)
            case DashNdbReturnType.INV_SQRT:
                out.copy_(Z16).div_(sqrt_scale)
            case _:
                raise RuntimeError(f'Got unknown value of return_type: {return_type}')

        del Y16, Z16, E32, tmp32, scale, sqrt_scale
