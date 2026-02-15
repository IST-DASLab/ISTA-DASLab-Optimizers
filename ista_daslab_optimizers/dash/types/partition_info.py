import enum

@enum.unique
class DashPartitionInfo(enum.Enum):
    REGULAR_BLOCK = enum.auto() # for 2D gradient
    NO_REST = enum.auto() # for LR without rest
    REST_L = enum.auto() # for LR
    REST_R = enum.auto() # for LR
