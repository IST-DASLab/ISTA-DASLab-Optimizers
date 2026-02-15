import enum


@enum.unique
class DashEvdHeuristic(enum.Enum):
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
