import torch
from torch import Tensor, distributed as dist

from ..dash_config import DashConfig
from ..types import DashEvdHeuristic


class DashRootInvEVD:
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
            case DashEvdHeuristic.ABS:
                L.sub_(eps).abs_()
            case DashEvdHeuristic.ABS_ADD:
                L.sub_(eps).abs_().add_(eps)
            case DashEvdHeuristic.RELU:
                L.sub_(eps)
                # torch.nn.functional.relu(L, inplace=True)
                L[L < eps] = 0
                if (dist.is_initialized() and dist.get_rank() == 0) or (not dist.is_initialized()): # make sure we are on master process
                    block_ranks = (L > 0).sum(dim=1)

            case DashEvdHeuristic.SHAMPOO:
                """
                This heuristic is presented in section 3.2.1 (1) of Distributed Shampoo paper: https://arxiv.org/pdf/2309.06497#page=15 (gray box):
                L_min = min_i lambda_i
                L_new = L - min(L_min, 0) * ones + epsilon * ones
                      = L + (epsilon - min(L_min, 0)) * ones
                """
                L_min = L.min(dim=1, keepdim=True).values
                L.add_(eps - L_min.clamp(max=0))

        if heuristic == DashEvdHeuristic.RELU:
            L[L > 0] **= (-1. / root) # raise to the -1/root only the non-zero values
            out.copy_(Q @ torch.diag_embed(L) @ Q.transpose(1, 2))
        else:
            out.copy_(Q @ torch.diag_embed(L.pow(-1. / root)) @ Q.transpose(1, 2))

        del L, Q, eye

        if block_ranks is not None:
            return block_ranks
