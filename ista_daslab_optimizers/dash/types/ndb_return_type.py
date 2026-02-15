import enum


@enum.unique
class DashNdbReturnType(enum.Enum):
    """
    Specifies what we should return from NewtonDB.
    """
    SQRT = enum.auto()
    INV_SQRT = enum.auto()
