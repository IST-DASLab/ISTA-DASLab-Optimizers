import torch
from dataclasses import dataclass
from .types import * # DashAlgoOneDim, DashInverseRootMethodType, DashGraftingType, DashEvdHeuristic, DashMatrixScalingType


@dataclass
class DashConfig:
    adamw_eps: float = 1e-8
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95

    beta_G: float = 0.9
    beta_LR: float = 0.95 # beta for factors L and R
    beta_graft: float = 0.95 # beta used for grafting

    eps_inv_root: float = 1e-10
    inv_root_method: DashInverseRootMethodType = DashInverseRootMethodType.EVD
    inv_root_freq: int = 10

    grafting_type: DashGraftingType = DashGraftingType.ADAM
    eps_grafting: float = 1e-8

    mu: float = 0.0
    use_nesterov: bool = True
    use_bias_correction: bool = True

    start_prec_step: int = -1
    block_size: int = 1024
    matmul_dtype: torch.dtype = torch.float32

    matrix_scaling_type: DashMatrixScalingType = DashMatrixScalingType.POWER_ITER
    matrix_scaling_pi_steps: int = 10
    matrix_scaling_const: float = 2 # the constant to multiply the scaling by

    newton_steps: int = 10 # for NewtonDB and CoupledNewton

    algo_one_dim: DashAlgoOneDim = DashAlgoOneDim.ADAMW

    ### Eiven-Value Decomposition (EVD)
    evd_heuristic: DashEvdHeuristic = DashEvdHeuristic.SHAMPOO

    ### Coupled Newton (CN)
    cn_tolerance: float = 1e-6

    ### Chebyshev (CBSHV)
    cbshv_degree: int = 20

    ### Low-Rank Power-Iter (LRPI)
    lrpi_rank: int = 100
    lrpi_steps: int = 10
