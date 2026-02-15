import os; os.system('clear')
import torch
from torch import bmm, Tensor
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from functools import partial
torch.set_printoptions(threshold=10_000)

from experimental_optimizers.utils.triton_kernels import update_dash_preconditioner

def max_eigenvalue_power_iteration_single_multi(A: Tensor, num_iters: int = 10, num_vecs: int = 16):
    """
    Performs Power-Iteration with `num_vecs` in parallel to minimize the chances
    of converging to an eigen-vector that has a corresponding eigen-value smaller
    than the largest eigen-value.
    """
    B, _ = A.shape

    # random initial vector
    v = torch.randn(B, num_vecs, device=A.device, dtype=A.dtype)
    v = v / v.norm(p=2, dim=1, keepdim=True)

    Av = torch.empty_like(v)

    for it in range(num_iters):
        torch.mm(A, v, out=Av) # (B, B) @ (B, num_vecs) => (B, num_vecs)
        v = Av / Av.norm(p=2, dim=0, keepdim=True)
    # end for it

    torch.matmul(A, v, out=Av)
    eig_vals_all = (v * Av).sum(dim=0) # (B, num_vecs).sum(dim=1) => (num_vecs, )

    max_vals, max_indices = eig_vals_all.max(dim=0) # (num_vecs,).max(dim=0) => (1,)

    # idx_expanded = max_indices.view(N, 1, 1).expand(N, B, 1) # (N, B, 1)
    # best_vals = max_vals.view(N, 1, 1)
    # best_vecs = v.gather(2, idx_expanded) # (N, B, 1)

    best_vals = max_vals
    best_vecs = v[:, max_indices]

    best_vals = best_vals.to(torch.float32)

    return best_vals, best_vecs

def main():
    from .partitioners.layer_partitioner import DashLayerPartitioner
    # A = torch.tensor([
    #     [1, 1, 2, 2, 3, 3, 4,],
    #     [1, 1, 2, 2, 3, 3, 4,],
    #     [5, 5, 6, 6, 7, 7, 8,],
    #     [5, 5, 6, 6, 7, 7, 8,]
    # ], dtype=torch.float)

    # A = torch.tensor([
    #     [1, 1, 2, 2, 3, 3],
    #     [1, 1, 2, 2, 3, 3],
    #     [5, 5, 6, 6, 7, 7],
    #     # [5, 5, 6, 6, 7, 7],
    # ], dtype=torch.float)

    # A = torch.tensor([
    #     [1, 1, 2, 2, 3, 3],
    #     [1, 1, 2, 2, 3, 3],
    #     [5, 5, 6, 6, 7, 7],
    #     [5, 5, 6, 6, 7, 7],
    # ], dtype=torch.float)

    A = torch.zeros(2048, 32_000, dtype=torch.int8)
    # A = torch.zeros(32_000, 2048, dtype=torch.int8)

    partitioner = DashLayerPartitioner(param=A, B=1024)
    blocks_full, blocks_rest = partitioner.split_into_blocks(A) # blocks_rest can be None
    # breakpoint()
    print(f'A: {tuple(A.shape)}')
    print(f'blocks: {tuple(blocks_full.shape)}')
    if blocks_rest is not None:
        print(f'rest: {tuple(blocks_rest.shape)}')
    # print(blocks_full)
    # print(blocks_rest)

    blocks_full += 10
    blocks_rest += 10

    A_reconstructed = partitioner.reconstruct_from_blocks(blocks_full, blocks_rest)
    # print(f'A_reconstructed: {A_reconstructed}')

    # print(A)

    print(f'shape full: {partitioner.shape_full}')
    print(f'shape rest: {partitioner.shape_rest}')

def test_upper_bounds_on_spectral_norm():
    N = 2048
    A = torch.randn(N, N, dtype=torch.float)
    # A = A.T @ A

    _, S, _ = torch.linalg.svd(A)
    # L, _ = torch.linalg.eigh(A)
    maxS = S.max().item()
    # maxL = L.max().item()

    norm_1 = A.abs().sum(dim=0).max().item() # sum over all rows (per column)
    norm_inf = A.abs().sum(dim=1).max().item() # sum over all columns (per row)

    sqrt_norm_1_inf = math.sqrt(norm_1 * norm_inf)
    fro = A.norm(p='fro')

    # print(f'       maxL: {maxL}')
    print(f'       maxS: {maxS}')
    print(f' sqrt-1-inf: {sqrt_norm_1_inf}')
    print(f'        fro: {fro}')
    print(f'sqrt(N)*fro: {math.sqrt(N) * maxS}')

def test_reusing_eigval_estimation_only_once_for_ndb():
    def max_eigval_power_iter(A: Tensor, num_iters: int):
        N, B, _ = A.shape

        # random initial vector
        v = torch.randn(N, B, 1, device=A.device, dtype=A.dtype)
        v = v / v.norm(p=2, dim=(1, 2), keepdim=True)

        Av = torch.empty_like(v)

        for it in range(num_iters):
            bmm(A, v, out=Av)
            v = Av / Av.norm(p=2, dim=(1, 2), keepdim=True)

        eigvals = bmm(v.transpose(1, 2), bmm(A, v))
        return eigvals.item()
    # end max_eigval_power_iter

    pi_iters = 5
    B = 1024
    X = torch.randn(B, B, dtype=torch.float32)
    A = X @ X.T

    L, Q = torch.linalg.eigh(A)
    ev_evd = L.max().item()
    ev_pi = max_eigval_power_iter(A.unsqueeze(0), pi_iters)

    Asqrt = Q @ torch.diag(L.sqrt()) @ Q.T

    sqrt_ev_evd = math.sqrt(ev_evd)
    sqrt_ev_pi = max_eigval_power_iter(Asqrt.unsqueeze(0), pi_iters)

    print(f'ev_evd: {ev_evd}')
    print(f'ev_pi: {ev_pi}')
    print(f'2*ev_pi: {2*ev_pi}')

    print(f'sqrt_ev_evd: {sqrt_ev_evd}')
    print(f'sqrt_ev_pi: {sqrt_ev_pi}')
    print(f'sqrt(ev_pi): {math.sqrt(ev_pi)}')
    print(f'sqrt(2*ev_pi): {math.sqrt(2*ev_pi)}')
    print(f'2*sqrt(ev_pi): {2 * math.sqrt(ev_pi)}')

    print(f'fro: {A.norm(p="fro")}')
    print(f'fro-sqrt: {Asqrt.norm(p="fro")}')

def test_shampoo_for_1d_params():
    from .partitioners.layer_partitioner import DashLayerPartitioner, DashFakeParam

    # func = torch.zeros
    func = DashFakeParam
    p = func((33, 2048), dtype=torch.float, device='cpu')
    bp = DashLayerPartitioner(param=p, B=1024, is_norm_layer_stack=True)

    if func == DashFakeParam:
        g = torch.randn_like(p.p)
    else:
        g = torch.randn_like(p)
    print(g)

    G = bp.get_regular_gradient_block()
    print(f'{bp.num_blocks_full=}')

    LR = bp.get_preconditioner_blocks_efficiently_grouped()
    bp.populate_gradient_block_partition(g, G)
    out = bp.reconstruct_from_blocks(block=G, out=None)
    print(out.shape)

def test_power_iter_multi():
    from .scalers import MatrixScaling
    pi_iters = 10
    pi_vecs = 16
    B = 1024
    X = torch.randn(3, B, B, dtype=torch.float32)
    A = X @ X.transpose(1, 2)
    A16 = A.half()

    L, Q = torch.linalg.eigh(A)
    vals_evd = L.max().item()
    vals_pi_single, vecs_pi_single           = MatrixScaling.max_eigval_power_iter      (A  , num_iters=pi_iters)
    vals_pi_single_fp16, vecs_pi_single_fp16 = MatrixScaling.max_eigval_power_iter      (A16, num_iters=pi_iters)
    vals_pi_multi, vecs_pi_multi             = MatrixScaling.max_eigval_power_iter_multi(A  , num_iters=pi_iters, num_vecs=pi_vecs)
    vals_pi_multi_fp16, vecs_pi_multi_fp16   = MatrixScaling.max_eigval_power_iter_multi(A16, num_iters=pi_iters, num_vecs=pi_vecs)
    vals_pi_single_multi, vecs_pi_single_multi = max_eigenvalue_power_iteration_single_multi(A[0], num_iters=pi_iters, num_vecs=1)

    print(f'{pi_iters=}, {pi_vecs=}')
    print(f'Max EV EVD: {vals_evd}')
    print(f'Max EV PI Single (FP32): {vals_pi_single.view(-1).cpu().numpy().astype(np.int32)}')
    print(f'Max EV PI Single (FP16): {vals_pi_single_fp16.view(-1).cpu().numpy().astype(np.int32)}')
    print(f'Max EV PI Multi  (FP32): {vals_pi_multi.view(-1).cpu().numpy().astype(np.int32)}')
    print(f'Max EV PI Multi  (FP16): {vals_pi_multi_fp16.view(-1).cpu().numpy().astype(np.int32)}')
    print(f'Max EV PI Single Multi (BF16): {vals_pi_single_multi.view(-1).cpu().numpy().astype(np.int32)}')
    print(torch.linalg.matrix_rank(A))

def test_eig_vals_zeros_relu_heuristic():
    B = 1024
    X = torch.randn(5, B, B, dtype=torch.float32)
    A = X @ X.transpose(1, 2)

    L, Q = torch.linalg.eigh(A)
    print(f'{L.shape=}')
    torch.nn.functional.relu(L, inplace=True)
    mask = L > 0
    ranks = mask.sum(dim=1)
    print(f'ranks={ranks}')

def test_dash_functionality():
    from .partitioners.gpu_partitioner import (
        get_stacked_shapes_for_merged_norm_layers,
        get_stacked_shapes_per_single_linear_layer,
        DashGpuPartitioner
    )

    def bucket_func(ndim):
        params = [
            ( 0, None, None, torch.empty(2048, dtype=torch.float)),
            ( 1, None, None, torch.empty(2048, dtype=torch.float)),
            ( 2, None, None, torch.empty(2048, dtype=torch.float)),
            ( 3, None, None, torch.empty(2048, dtype=torch.float)),
            ( 4, None, None, torch.empty(2048, dtype=torch.float)),
            ( 5, None, None, torch.empty(32_000, 2048, dtype=torch.float)),
            ( 6, None, None, torch.empty(2048, 5632, dtype=torch.float)),
            ( 7, None, None, torch.empty(2048, 2048, dtype=torch.float)),
            ( 8, None, None, torch.empty(5632, 2048, dtype=torch.float)),
            ( 9, None, None, torch.empty(5632, 2048, dtype=torch.float)),
            (10, None, None, torch.empty(2048, 2048, dtype=torch.float)),
            (11, None, None, torch.empty(2048, 5632, dtype=torch.float)),
        ]
        for (index, group, state, p) in params:
            if p.ndim == ndim:
                yield (index, group, state, p)

    # shape_1d = get_stacked_shapes_for_merged_norm_layers(shape=(33, 2048), B=1024)
    # # shape_2d = get_stacked_shapes_per_single_linear_layer(shape=(4096, 2048), B=1024)
    # print(shape_1d)
    # # print(shape_2d)

    B = 1024
    # dbp_1d = DashDashLayerwisePartitioner(partial(bucket_func, ndim=1), B, is_norm_layer_stack=True)
    dbp_2d = DashGpuPartitioner(partial(bucket_func, ndim=2), B, is_norm_layer_stack=False)
    print(dbp_2d.dash_shape_2d_full)
    print(dbp_2d.dash_shape_2d_rest)

def test_dash_kernel():
    M = 512
    N = 640
    beta = 0.95

    dtype = torch.float32
    device = 'cuda:0'
    batches = 12
    G = torch.randn(batches, M, N, dtype=dtype, device=device, requires_grad=False)
    L = torch.zeros(batches, M, M, dtype=dtype, device=device, requires_grad=False)
    R = torch.zeros(batches, N, N, dtype=dtype, device=device, requires_grad=False)

    G_T = G.transpose(1, 2)

    Lresult = beta * L + (1 - beta) * torch.bmm(G, G_T)
    Rresult = beta * R + (1 - beta) * torch.bmm(G_T, G)

    # warmup
    update_dash_preconditioner(X=L.clone(), G=G, beta=beta, compute_left=1)
    update_dash_preconditioner(X=R.clone(), G=G, beta=beta, compute_left=0)
    L.zero_()
    R.zero_()
    update_dash_preconditioner(X=L, G=G, beta=beta, compute_left=1)
    update_dash_preconditioner(X=R, G=G, beta=beta, compute_left=0)

    abs_L = (L - Lresult).abs()
    abs_R = (R - Rresult).abs()
    abs_max_L = abs_L.max()
    abs_max_R = abs_R.max()
    abs_min_L = abs_L.min()
    abs_min_R = abs_R.min()

    print(f'M = {M}')
    print(f'N = {N}')
    print(f'abs max for L: {abs_max_L.item()}')
    print(f'abs max for R: {abs_max_R.item()}')
    print(f'abs min for L: {abs_min_L.item()}')
    print(f'abs min for R: {abs_min_R.item()}')

    # breakpoint()
    # print('Program ended.')
    """
    
    L[0, :6, :6]
    Lresult[0, :6, :6]
    
    L[0, -6:, -6:]
    
    R[0, :6, :6]
    Rresult[0, :6, :6]
    
    
    """


if __name__ == '__main__':
    # main()
    # test_upper_bounds_on_spectral_norm()
    # test_reusing_eigval_estimation_only_once_for_ndb()
    # test_shampoo_for_1d_params()
    # test_power_iter_multi()
    # test_eig_vals_zeros_relu_heuristic()
    # test_dash_functionality()
    test_dash_kernel()