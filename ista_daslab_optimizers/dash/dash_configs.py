import torch
import enum
from typing import Tuple
from dataclasses import dataclass

@enum.unique
class DashStackingStrategy(enum.Enum):
    """
    Specifies stacking strategy for Dash
    """

    """Stack all layers from a GPU into a single tensor for 1D and a single tensor for 2D"""
    MAXIMIZE_EFFICIENCY = enum.auto()

    """Perform stacking per layer"""
    LAYER_BY_LAYER = enum.auto()


@enum.unique
class DashAlgoOneDim(enum.Enum):
    """
    Specifies how we should update 1D parameters: using Adam (default) or applying Shampoo (actually, AdaGrad):

    Let E = embedding size, which is the dimensionality of all layer norms (RMSNorm for Llama model in llm-baselines)
    G has shape (E,) and has to be converted to (E, 1)
    Compute only preconditioner L as G @ G.T of shape (E, E) and then compute -1/2 power for it: W = W - L^-1/2 @ G
    """
    ADAMW = enum.auto()
    SHAMPOO = enum.auto()

    @classmethod
    def from_string(cls, s: str):
        match s.lower():
            case 'adamw':
                return cls.ADAMW
            case 'shmp':
                return cls.SHAMPOO
            case _:
                raise ValueError(f'Unknown value for AlgoOneDim  {s.lower()}')

@enum.unique
class DashInverseRootMethodType(enum.Enum):
    EVD = enum.auto() # EigenValueDecomposition
    CN = enum.auto() # Coupled Newton
    JORGE = enum.auto() # Jorge from https://arxiv.org/pdf/2310.12298
    CBSHV = enum.auto() # Chebyshev
    NDB = enum.auto() # NewtonDB

    @classmethod
    def from_string(cls, s: str):
        match s.lower():
            case 'evd':
                return cls.EVD
            case 'cn':
                return cls.CN
            case 'jorge':
                return cls.JORGE
            case 'cbshv':
                return cls.CBSHV
            case 'ndb':
                return cls.NDB
            case _:
                raise ValueError(f'Unsupported inverse root method type {s.lower()}')

@enum.unique
class DashGraftingType(enum.Enum):
    ADAGRAD = enum.auto()
    ADAM = enum.auto()

    @classmethod
    def from_string(cls, s: str):
        match s.lower():
            case 'adagrad':
                return cls.ADAGRAD
            case 'adam':
                return cls.ADAM
            case _:
                raise ValueError(f'Unsupported grafting type {s.lower()}')

@enum.unique
class DashEVDHeuristic(enum.Enum):
    ABS = enum.auto()  # apply abs to eigenvalues
    ABS_ADD = enum.auto() # apply abs to eigenvalues and then add epsilo
    RELU = enum.auto()  # apply abs to eigenvalues
    SHAMPOO = enum.auto()  # Shampoo EigenValue Heuristic (section 3.2.1 (1))

    @classmethod
    def from_string(cls, s: str):
        match s.lower():
            case 'abs':
                return cls.ABS
            case 'abs-add':
                return cls.ABS_ADD
            case 'relu':
                return cls.RELU
            case 'shmp':
                return cls.SHAMPOO
            case _:
                raise ValueError(f'Unsupported eigen values heuristic type {s.lower()}')

@enum.unique
class DashNDBReturnType(enum.Enum):
    """
    Specifies what we should return from NewtonDB.
    """
    SQRT = enum.auto()
    INV_SQRT = enum.auto()

@enum.unique
class DashMatrixScalingType(enum.Enum):
    """
    Specifies how we should scale the matrix to ensure the condition ||I-A|| < 1
    """
    POWER_ITER = enum.auto()
    POWER_ITER_MULTI = enum.auto()
    FRO = enum.auto()

    @classmethod
    def from_string(cls, s: str):
        match s.lower():
            case 'pi':
                return cls.POWER_ITER
            case 'pim':
                return cls.POWER_ITER_MULTI
            case 'fro':
                return cls.FRO
            case _:
                raise ValueError(f'Unknown value for scaling type  {s.lower()}')

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
    evd_heuristic: DashEVDHeuristic = DashEVDHeuristic.SHAMPOO

    ### Coupled Newton (CN)
    cn_tolerance: float = 1e-6

    ### Chebyshev (CBSHV)
    cbshv_degree: int = 20
