import enum


@enum.unique
class DashInverseRootMethodType(enum.Enum):
    EVD = enum.auto() # EigenValueDecomposition
    CN = enum.auto() # Coupled Newton
    JORGE = enum.auto() # Jorge from https://arxiv.org/pdf/2310.12298
    CBSHV = enum.auto() # Chebyshev
    NDB = enum.auto() # NewtonDB
    LRPI = enum.auto()

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
            case 'lrpi':
                return cls.LRPI
            case _:
                raise ValueError(f'Unsupported inverse root method type {s.lower()}')
