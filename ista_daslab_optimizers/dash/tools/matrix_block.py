from dataclasses import dataclass

import torch
from torch import Tensor

from ..types import DashPartitionInfo


@dataclass
class DashMatrixBlock:
    full: Tensor = None
    rest: Tensor = None
    info: DashPartitionInfo = None

    def __init__(self, shape_full, shape_rest, info, dtype, device):
        if shape_full is not None: self.full = torch.zeros(shape_full, dtype=dtype, device=device)
        if shape_rest is not None: self.rest = torch.zeros(shape_rest, dtype=dtype, device=device)
        if info is not None: self.info = info

    @property
    def has_rest(self):
        return (self.rest is not None) and (isinstance(self.rest, Tensor))

    @classmethod
    def like(cls, block):
        return cls(shape_full=block.shape_full_1d,
                   shape_rest=block.shape_rest_1d,
                   info=block.info,
                   dtype=block.dtype,
                   device=block.device)
