import torch
import triton
import triton.language as tl
from torch import Tensor

def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128, 256]
        for bn in [64, 128, 256]
        for bk in [64, 128, 256]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm == bn # required for pid_to_block_triangular
    ]

@triton.jit
def _pid_to_block_triangular(
    pid,
    SIZE_M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Number of block tiles along one matrix dimension
    T = tl.cdiv(SIZE_M, BLOCK_SIZE_M)

    # Number of triangular blocks per batch
    tri_per_batch = T * (T + 1) // 2

    # Extract batch index
    batch_idx = pid // tri_per_batch
    tri_id = pid % tri_per_batch

    tri_id_fp32 = tri_id.to(tl.float32)

    # Solve triangular indexing:
    # Find i such that i(i+1)/2 <= tri_id < (i+1)(i+2)/2
    i = tl.floor(
        (
            tl.sqrt(1.0 + 8.0 * tri_id_fp32) - 1.0
        ) * 0.5
    ).to(tl.int32)

    # Starting linear index of row i
    row_start = i * (i + 1) // 2
    i = tl.where(row_start > tri_id, i - 1, i)
    row_start = tl.where(row_start > tri_id, i * (i + 1) // 2, row_start)
    next_row_start = (i + 1) * (i + 2) // 2
    i = tl.where(next_row_start <= tri_id, i + 1, i)
    row_start = tl.where(next_row_start <= tri_id, next_row_start, row_start)

    # Column index inside triangular row
    j = tri_id - row_start

    # Convert block indices to matrix offsets
    m_idx = i * BLOCK_SIZE_M
    n_idx = j * BLOCK_SIZE_N

    return batch_idx, m_idx, n_idx


@triton.autotune(
    configs=_get_autotune_configs(),
    # Key now includes output size (SIZE_M) and reduction size (SIZE_K)
    key=["SIZE_M", "SIZE_K", "g_stride_r", "g_stride_c", "x_stride_r", "x_stride_c", "compute_left"],
)
@triton.jit
def update_dash_preconditioner_kernel_triangle(
    G_ptr, X_ptr,
    SIZE_M,  # Output dimension size (Rows/Cols of X)
    SIZE_K,  # Reduction dimension size (Common dim of G)
    g_stride_b, g_stride_r, g_stride_c,
    x_stride_b, x_stride_r, x_stride_c,
    beta,
    compute_left: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    batch_idx, m_idx, n_idx = _pid_to_block_triangular(pid, SIZE_M, BLOCK_SIZE_M, BLOCK_SIZE_N)

    G_ptr += batch_idx * g_stride_b
    X_ptr += batch_idx * x_stride_b

    offs_m = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_idx + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointer Setup
    # SIZE_M is the range for m_idx and n_idx
    # SIZE_K is the range for k (loop)

    if compute_left == 0:
        # Case: X = G.T @ G
        # X is (N x N), Reduction is M.
        # Here SIZE_M maps to N, SIZE_K maps to M.

        # Operand A: G.T[m, k] -> G[k, m]. Rows=k, Cols=m
        g_ptrs = G_ptr + (offs_k[None, :] * g_stride_r + offs_m[:, None] * g_stride_c)
        g_step = BLOCK_SIZE_K * g_stride_r

        # Operand B: G[k, n]. Rows=k, Cols=n
        gt_ptrs = G_ptr + (offs_k[:, None] * g_stride_r + offs_n[None, :] * g_stride_c)
        gt_step = BLOCK_SIZE_K * g_stride_r

    else:
        # Case: X = G @ G.T
        # X is (M x M), Reduction is N.
        # Here SIZE_M maps to M, SIZE_K maps to N.

        # Operand A: G[m, k]. Rows=m, Cols=k
        g_ptrs = G_ptr + (offs_m[:, None] * g_stride_r + offs_k[None, :] * g_stride_c)
        g_step = BLOCK_SIZE_K * g_stride_c

        # Operand B: G.T[k, n] -> G[n, k]. Rows=n, Cols=k
        gt_ptrs = G_ptr + (offs_k[:, None] * g_stride_c + offs_n[None, :] * g_stride_r)
        gt_step = BLOCK_SIZE_K * g_stride_c

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    mask_mn = offs_m < SIZE_M  # offs_n < SIZE_M is identical since BM==BN

    # Main Loop (Reduction over SIZE_K)
    for k in tl.range(0, tl.cdiv(SIZE_K, BLOCK_SIZE_K)):
        k_remaining = SIZE_K - k * BLOCK_SIZE_K
        mask_k = offs_k < k_remaining

        if compute_left == 0:
            load_mask_g = mask_k[None, :] & mask_mn[:, None]
            load_mask_gt = mask_k[:, None] & mask_mn[None, :]
        else:
            load_mask_g = mask_mn[:, None] & mask_k[None, :]
            load_mask_gt = mask_k[:, None] & mask_mn[None, :]

        g = tl.load(g_ptrs, mask=load_mask_g, other=0.0)
        gt = tl.load(gt_ptrs, mask=load_mask_gt, other=0.0)

        accumulator = tl.dot(g, gt, accumulator)

        g_ptrs += g_step
        gt_ptrs += gt_step

    # Epilogue
    x_ptrs = X_ptr + (offs_m[:, None] * x_stride_r + offs_n[None, :] * x_stride_c)
    # x_mask = (offs_m[:, None] < SIZE_M) & (offs_n[None, :] < SIZE_M)
    x_mask = mask_mn[:, None] & mask_mn[None, :]

    # Read old X
    x_curr = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
    accumulator = beta * x_curr + (1.0 - beta) * accumulator
    output = accumulator.to(X_ptr.dtype.element_ty)

    # Store Lower Triangular Block
    tl.store(x_ptrs, output, mask=x_mask)

def update_dash_preconditioner(X: Tensor, G: Tensor, beta: float, compute_left: bool):
    """
    Computes update:
    compute_left=True:  X = beta * X + (1-beta) * G @ G.T
    compute_left=False: X = beta * X + (1-beta) * G.T @ G
    """
    if G.ndim > 3 or G.ndim < 2:
        raise ValueError(f"Input tensor must be 2D or 3D, but got {G.ndim}D tensor.")
    if X is None:
        raise ValueError(f"Output tensor must not be None!")

    # Determine Dimensions
    # G shape is [Batch, Row, Col]
    g_rows = G.size(-2)
    g_cols = G.size(-1)

    if compute_left:
        # X = G @ G.T -> Output (Rows x Rows), Reduction (Cols)
        size_m = g_rows
        size_k = g_cols
    else:
        # X = G.T @ G -> Output (Cols x Cols), Reduction (Rows)
        size_m = g_cols
        size_k = g_rows

    # Validation
    assert X.size(-2) == size_m and X.size(-1) == size_m, \
        f"Output shape {X.shape} mismatch. Expected {size_m}x{size_m}."

    batch_size = G.size(0) if G.ndim == 3 else 1
    g_stride_b = G.stride(0) if G.ndim == 3 else 0
    x_stride_b = X.stride(0) if X.ndim == 3 else 0

    def grid(meta):
        T = triton.cdiv(size_m, meta["BLOCK_SIZE_M"])
        return (batch_size * (T * (T + 1) // 2),)

    beta = float(beta)

    update_dash_preconditioner_kernel_triangle[grid](
        G_ptr=G,
        X_ptr=X,
        SIZE_M=size_m,
        SIZE_K=size_k,
        g_stride_b=g_stride_b,
        g_stride_r=G.stride(-2),
        g_stride_c=G.stride(-1),
        x_stride_b=x_stride_b,
        x_stride_r=X.stride(-2),
        x_stride_c=X.stride(-1),
        beta=beta,
        compute_left=int(compute_left)
    )

    # mirror outside kernel:
    # => X is lower triangular (contains non-zero values also on the upper diagonal)
    # => X.tril_(): erases the upper-diagonal part
    # => .add_(...): add the transposed part of X (X is lower-triang, X.T is upper-triang)
    #    => diagonal=1 means we skip the main diagonal
    X.tril_().add_(X.transpose(1, 2).triu(diagonal=1))

    return X