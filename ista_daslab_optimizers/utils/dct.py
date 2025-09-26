import torch
import math

from utils.global_cache import GlobalCache

def dct3_matrix(n, dtype, device):
    """
        This function returns the orthogonal transformation for Discrete Cosine Transform (DCT-3).
    """
    lin = torch.arange(n)
    I = lin.repeat(n, 1).to(torch.float)
    Q = math.sqrt(2 / n) * torch.cos(torch.pi * (I.t() * (2. * I + 1.)) / (2. * n))
    del lin, I
    Q[0, :] *= math.sqrt(0.5)
    return Q.to(device=device, dtype=dtype)

def dct_type2_makhoul(X):
    N = X.shape[1]

    if GlobalCache.contains(category='perm', key=N):
        perm = GlobalCache.get(category='perm', key=N)
    else:
        even_idx = torch.arange(0, N, 2)  # 0, 2, 4, ...
        odd_idx = torch.arange(1, N, 2).flip(0)  # last odd → first odd
        perm = torch.cat([even_idx, odd_idx]).to(X.device)

        GlobalCache.add(category='perm', key=N, item=perm)
    #
    # X_input = X[:, perm]
    # if X_input.dtype != torch.float:
    #     X_input = X_input.to(torch.float)
    # X_fft = torch.fft.fft(X_input, dim=1)

    X_fft = torch.fft.fft(X[:, perm].contiguous(), dim=1)

    if GlobalCache.contains(category='twiddle', key=N):
        W = GlobalCache.get(category='twiddle', key=N)
    else:
        W = 2 * torch.exp((-1j * torch.pi * torch.arange(N, device=X.device) / (2 * N)))
        W[0] /= math.sqrt(4 * N)
        W[1:] /= math.sqrt(2 * N)

        GlobalCache.add(category='twiddle', key=N, item=W.reshape(1, N))

    return (X_fft * W).real