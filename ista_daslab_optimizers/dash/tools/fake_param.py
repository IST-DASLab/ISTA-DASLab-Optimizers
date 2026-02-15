from dataclasses import dataclass
import torch


@dataclass
class DashFakeParam:
    """
    This class is designed to simulate a torch.Tensor type with value p and gradient grad to be used in partitioners the normalization
    layers when updating them with Shampoo (AdaGrad). This should be compatible with DashLayerProcessor and DashLayerwisePartitioner
    """
    shape: None
    dtype: None
    device: None
    ndim: None
    p: None
    grad: None

    def __init__(self, shape, dtype, device):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.ndim = 2

        self.p = torch.zeros(shape, dtype=dtype, device=device, requires_grad=False)
        # self.grad = torch.eros # I wrote this while thinking about buying Eros Pista haha
        self.grad = torch.zeros_like(self.p)
