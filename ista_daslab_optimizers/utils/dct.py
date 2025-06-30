import torch
import math

def dct3_matrix(n):
    """
        This function returns the orthogonal transformation for Discrete Cosine Transform (DCT-3).
    """
    lin = torch.arange(n)
    I = lin.repeat(n, 1).to(torch.float)
    Q = math.sqrt(2 / n) * torch.cos(torch.pi * (I.t() * (2. * I + 1.)) / (2. * n))
    del lin, I
    Q[0, :] *= math.sqrt(0.5)
    return Q
