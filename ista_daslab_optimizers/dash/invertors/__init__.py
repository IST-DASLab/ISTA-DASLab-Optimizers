import torch
from torch import Tensor
from typing import Union

from ..tools import DashMatrixBlock
from ..dash_config import DashConfig
from ..types import DashInverseRootMethodType

from .inv_cbshv import DashRootInvChebyshev
from .inv_cn import DashRootInvCoupledNewton
from .inv_evd import DashRootInvEVD
from .inv_jorge import DashRootInvJorge
from .inv_lrpi import DashRootInvLowRankPowerIter
from .inv_ndb import DashRootInvNewtonDB

class DashRootInvertor:
    @staticmethod
    @torch.no_grad()
    def invert(Xin: Union[DashMatrixBlock, Tensor], Xout: Union[DashMatrixBlock, Tensor], cfg: DashConfig, root: int):
        assert type(Xin) == type(Xout) # make sure they are not mixed
        method = cfg.inv_root_method
        match method:
            case DashInverseRootMethodType.EVD:
                func = DashRootInvEVD._matrix_root_inv_evd
            case DashInverseRootMethodType.CN:
                func = DashRootInvCoupledNewton._matrix_root_inv_coupled_newton
            case DashInverseRootMethodType.JORGE:
                func = DashRootInvJorge._matrix_root_inv_jorge
            case DashInverseRootMethodType.CBSHV:
                func = DashRootInvChebyshev._matrix_root_inv_chebyshev
            case DashInverseRootMethodType.NDB:
                func = DashRootInvNewtonDB._matrix_root_inv_newton_db
            case DashInverseRootMethodType.LRPI:
                func = DashRootInvLowRankPowerIter._matrix_root_inv_low_rank_power_iter
            case _:
                raise RuntimeError(f'Invalid method: {method}')
        # end match-case

        if isinstance(Xin, DashMatrixBlock): # this is for DashLayerwise
            ret = func(inp=Xin.full, out=Xout.full, cfg=cfg, root=root)
            if Xin.has_rest and Xout.has_rest: # run this on a new stream?
                func(inp=Xin.rest, out=Xout.rest, cfg=cfg, root=root)
        elif isinstance(Xin, Tensor): # this is for DashGpu
            ret = func(inp=Xin, out=Xout, cfg=cfg, root=root)

        if ret is not None:
            return ret
