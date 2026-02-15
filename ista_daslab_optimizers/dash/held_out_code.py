# @staticmethod
# @torch.no_grad()
# def _newton_db_fp32(inp: Tensor, out: Tensor, cfg: DashConfig, scale: Tensor, return_type: DashNdbReturnType):
#     """
#         This method uses eps in eigh and then again in the heuristic, exactly as the original DistributedShampoo (see A_ridge in their implementation).
#     """
#     assert out is not None
#     N, B, _ = inp.shape  # N=number blocks (batches), B = block size
#     sqrt_scale = scale.sqrt()
#     Y = inp / scale
#     Z = torch.eye(B, dtype=inp.dtype, device=inp.device).repeat(N, 1, 1)
#     E = torch.empty_like(inp)
#
#     tmp = torch.empty_like(inp)
#
#     for s in range(cfg.newton_steps):
#         # baddbmm(out=E, beta=0, input=Z, alpha=1, batch1=Z, batch2=Y)
#         bmm(input=Z, mat2=Y, out=E)  # E = ZY
#         E.diagonal(dim1=-2, dim2=-1).sub_(3)  # E = ZY - 3I
#
#         bmm(input=Y, mat2=E, out=tmp)
#         tmp.mul_(-0.5)
#         Y.copy_(tmp)
#
#         bmm(input=E, mat2=Z, out=tmp)
#         tmp.mul_(-0.5)
#         Z.copy_(tmp)
#     # end for steps
#
#     match return_type:
#         case DashNdbReturnType.SQRT:
#             out.copy_(Y).mul_(sqrt_scale)
#         case DashNdbReturnType.INV_SQRT:
#             out.copy_(Z).div_(sqrt_scale)
#         case _:
#             raise RuntimeError(f'Got unknown value of return_type: {return_type}')
#
#     del Y, Z, E, tmp, scale, sqrt_scale

# @staticmethod
# @torch.no_grad()
# def _newton_db_fp16(inp: Tensor, out: Tensor, cfg: DashConfig, scale: Tensor, return_type: DashNdbReturnType):
#     """
#         This method uses eps in eigh and then again in the heuristic, exactly as the original DistributedShampoo (see A_ridge in their implementation).
#     """
#     assert out is not None
#     N, B, _ = inp.shape  # N=number blocks (batches), B = block size
#     sqrt_scale = scale.sqrt()
#     Y16 = (inp / scale).half()
#     Z16 = torch.eye(B, dtype=torch.float16, device=inp.device).repeat(N, 1, 1)  # fp16!
#     E32 = torch.empty_like(inp)  # this is saved in fp32!!!
#
#     tmp32 = torch.empty_like(inp)  # this is also fp32
#
#     for s in range(cfg.newton_steps):
#         bmm(input=Z16, mat2=Y16, out_dtype=torch.float32, out=E32)  # E = ZY
#         E32.diagonal(dim1=-2, dim2=-1).sub_(3)  # E = ZY - 3I
#
#         E16 = E32.half()
#
#         bmm(input=Y16, mat2=E16, out_dtype=torch.float32, out=tmp32)
#         tmp32.mul_(-0.5)
#         Y16.copy_(tmp32)
#
#         bmm(input=E16, mat2=Z16, out_dtype=torch.float32, out=tmp32)
#         tmp32.mul_(-0.5)
#         Z16.copy_(tmp32)
#     # end for steps
#
#     match return_type:
#         case DashNdbReturnType.SQRT:
#             Y16.mul_(sqrt_scale)
#             out.copy_(Y16.float())
#         case DashNdbReturnType.INV_SQRT:
#             Z16.div_(sqrt_scale)
#             out.copy_(Z16.float())
#         case _:
#             raise RuntimeError(f'Got unknown value of return_type: {return_type}')
#
#     del Y16, Z16, E32, tmp32, scale, sqrt_scale

# @staticmethod
# @torch.no_grad()
# def _newton_db_bf16(inp: Tensor, out: Tensor, cfg: DashConfig, scale: Tensor, return_type: DashNdbReturnType):
#     """
#         This method uses eps in eigh and then again in the heuristic, exactly as the original DistributedShampoo (see A_ridge in their implementation).
#     """
#     assert out is not None
#     N, B, _ = inp.shape  # N=number blocks (batches), B = block size
#     sqrt_scale = scale.sqrt()
#     Y16 = (inp / scale).bfloat16()
#     Z16 = torch.eye(B, dtype=torch.bfloat16, device=inp.device).repeat(N, 1, 1)  # fp16!
#     E32 = torch.empty_like(inp)  # this is saved in fp32!!!
#
#     tmp32 = torch.empty_like(inp)  # this is also fp32
#
#     for s in range(cfg.newton_steps):
#         bmm(input=Z16, mat2=Y16, out_dtype=torch.float32, out=E32)  # E = ZY
#         E32.diagonal(dim1=-2, dim2=-1).sub_(3)  # E = ZY - 3I
#
#         E16 = E32.bfloat16()
#
#         bmm(input=Y16, mat2=E16, out_dtype=torch.float32, out=tmp32)
#         tmp32.mul_(-0.5)
#         Y16.copy_(tmp32)
#
#         bmm(input=E16, mat2=Z16, out_dtype=torch.float32, out=tmp32)
#         tmp32.mul_(-0.5)
#         Z16.copy_(tmp32)
#     # end for steps
#
#     match return_type:
#         case DashNdbReturnType.SQRT:
#             Y16.mul_(sqrt_scale)
#             out.copy_(Y16.float())
#         case DashNdbReturnType.INV_SQRT:
#             Z16.div_(sqrt_scale)
#             out.copy_(Z16.float())
#         case _:
#             raise RuntimeError(f'Got unknown value of return_type: {return_type}')
#
#     del Y16, Z16, E32, tmp32, scale, sqrt_scale