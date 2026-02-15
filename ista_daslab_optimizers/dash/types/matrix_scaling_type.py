import enum


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
