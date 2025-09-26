import torch
import numpy as np

class Quantizer4bit:
    def __init__(self, shape, device, dtype, bucket_size):
        assert np.prod(shape) % bucket_size == 0
        self.shape = shape
        self.device = device
        self.bucket_size = bucket_size
        self.numel = np.prod(shape)

        self.n_buckets = self.numel // self.bucket_size

        self.xq = torch.zeros(self.numel // 2, dtype=torch.uint8, device=self.device)
        self.x_min = torch.zeros(self.n_buckets, 1, dtype=dtype, device=self.device)
        self.x_max = torch.zeros(self.n_buckets, 1, dtype=dtype, device=self.device)

    def quantize(self, x):
        N, B = self.n_buckets, self.bucket_size
        N = self.numel // B
        self.x_min.copy_(x.view(N, B).min(dim=1).values.view(-1, 1))
        self.x_max.copy_(x.view(N, B).max(dim=1).values.view(-1, 1))
        u = (self.x_max - self.x_min) / 15.
        xq = ((x.view(N, B) - self.x_min) / u + 0.5).floor().to(torch.uint8).view(-1, 2)
        byte_left = xq[:, 0] << 4
        byte_right = xq[:, 1]
        self.xq.copy_(byte_left | byte_right)

    def quantize_inv(self):
        N, B = self.n_buckets, self.bucket_size
        u = (self.x_max - self.x_min) / 15.
        byte_left = (self.xq & 0xF0) >> 4
        byte_right = self.xq & 0x0F
        xq = torch.hstack(
            (
                byte_left.view(-1),
                byte_right.view(-1)
            )
        ).view(N, B) # intercalate byte_left and byte_right
        x = xq * u + self.x_min
        return x.view(*self.shape)

class Quantizer8bit:
    def __init__(self, shape, device, dtype, bucket_size):
        assert np.prod(shape) % bucket_size == 0
        self.shape = shape
        self.device = device
        self.bucket_size = bucket_size
        self.numel = np.prod(shape)

        self.n_buckets = self.numel // self.bucket_size

        self.xq = torch.zeros(self.numel, dtype=torch.uint8, device=self.device)
        self.x_min = torch.zeros(self.n_buckets, 1, dtype=dtype, device=self.device)
        self.x_max = torch.zeros(self.n_buckets, 1, dtype=dtype, device=self.device)

    def quantize(self, x):
        N, B = self.n_buckets, self.bucket_size
        N = self.numel // B
        self.x_min.copy_(x.view(N, B).min(dim=1).values.view(-1, 1))
        self.x_max.copy_(x.view(N, B).max(dim=1).values.view(-1, 1))
        u = (self.x_max - self.x_min) / 15.
        xq = ((x.view(N, B) - self.x_min) / u + 0.5).floor().to(torch.uint8)
        self.xq.copy_(xq.view(-1))
        del xq, u

    def quantize_inv(self):
        N, B = self.n_buckets, self.bucket_size
        u = (self.x_max - self.x_min) / 15.
        x = self.xq.view(N, B) * u + self.x_min
        return x.view(*self.shape)
