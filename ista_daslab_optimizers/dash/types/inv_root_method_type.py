import enum


@enum.unique
class DashInverseRootMethodType(enum.Enum):
    EVD = enum.auto() # EigenValueDecomposition
    CN = enum.auto() # Coupled Newton
    CBSHV = enum.auto() # Chebyshev
    NDB = enum.auto() # NewtonDB

    @classmethod
    def from_string(cls, s: str):
        match s.lower():
            case 'evd':
                return cls.EVD
            case 'cn':
                return cls.CN
            case 'cbshv':
                return cls.CBSHV
            case 'ndb':
                return cls.NDB
            case _:
                raise ValueError(f'Unsupported inverse root method type {s.lower()}')
