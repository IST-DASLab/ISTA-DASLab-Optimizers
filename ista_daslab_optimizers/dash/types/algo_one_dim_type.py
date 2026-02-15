import enum


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
                raise ValueError(f'Unknown value for DashAlgoOneDim  {s.lower()}')
