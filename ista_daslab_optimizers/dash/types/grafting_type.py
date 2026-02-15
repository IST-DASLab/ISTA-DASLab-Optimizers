import enum


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
