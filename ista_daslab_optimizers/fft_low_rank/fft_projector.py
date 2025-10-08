import torch
import torch.distributed as dist
from ista_daslab_optimizers.utils.dct import dct_type2_makhoul
from ista_daslab_optimizers.utils.global_cache import GlobalCache

class FFTLowRankProjector:
    def __init__(self, p, rank, proj, rotate_subspace, sim_type='matmul', ell_norm=1, use_th_sim=False):
        assert sim_type in ['matmul', 'makhoul']
        self.rank = rank
        self.proj = proj
        self.rotate_states = rotate_subspace # allocate indices_pref only if we choose to rotate the subspace
        self.sim_type = sim_type
        self.ell_norm = ell_norm
        self.use_th_sim = use_th_sim

        self.size = None
        self.indices_crt = None # the indices for the columns/rows
        self.indices_prev = None  # the indices for the columns/rows
        self.is_right_proj = None

        self.steps = 0
        self.device = f'cuda:{dist.get_rank()}' if dist.is_initialized() else 'cuda:0'

        GlobalCache.init()
        self._setup(p)

    def _setup(self, p):
        n, m = p.shape
        if n >= m:
            self.is_right_proj = True
            self.size = min(n, m)
        else:
            # fix for Llama-3-8B that has a layer of size (1024, 4096)
            # fix for Qwen2.5-7B that has a layer of size (512, 3584)
            if n in [512, 1024] and m in [3584, 4096]:
                self.is_right_proj = True
                self.size = m
            else:
                self.is_right_proj = False
                self.size = min(n, m)
        # self.is_right_proj = (n >= m) or (n < m and self.size == m)

        self.indices_crt = torch.zeros(self.rank, dtype=torch.int32, device=p.device)
        if self.rotate_states:
            self.indices_prev = torch.zeros(self.rank, dtype=torch.int32, device=p.device)

    def inc_step(self):
        self.steps += 1

    def compute_similarity_matmul(self, Q, A):
        if self.is_right_proj:
            S = A @ Q
            norms = S.norm(p=self.ell_norm, dim=0) # dim = 0 computes norm of columns (over all rows)
        else:
            S = Q.T @ A
            norms = S.norm(p=self.ell_norm, dim=1) # dim = 1 computes norm of rows (over all columns)
        return S, norms

    def compute_similarity_makhoul(self, A):
        if self.is_right_proj: # R >= C
            S = dct_type2_makhoul(A)
            norms = S.norm(p=1, dim=0)  # dim = 0 computes norm of columns (over all rows) to rank columns
        else: # R < C
            S = dct_type2_makhoul(A.T)
            S = S.T # account for the transposition in inputM because Makhoul computes DCT per rows by default
            norms = S.norm(p=1, dim=1)  # dim = 1 computes norm of rows (over all columns) to rank rows
        return S, norms

    def change_subspace(self, Q, A, col_norms, out=None):
        """
            This method computes P = A @ Q or P = Q.T @ A and then ranks the columns/rows of matrix P.
            Once we determine the most important r indices, we can simply select them directly from P
        without having to multiply again A @ Q[:, indices] or Q[indices, :] @ A.
            This way, we save some computations.
        """
        # if self.steps == 1 or self.steps % self.update_proj_gap == 0:
        if self.steps > 1:
            if self.rotate_states:
                self.indices_prev.copy_(self.indices_crt)

        if self.sim_type == 'matmul':
            S, norms = self.compute_similarity_matmul(Q, A)
        else:
            S, norms = self.compute_similarity_makhoul(A)

        if self.use_th_sim:
            norms.mul_(col_norms)

        indices = torch.topk(
            input=norms,
            k=self.rank,
            sorted=False,
        ).indices

        self.indices_crt.copy_(indices)
        del indices, norms

        # if self.sim_type == 'matmul':
        if out is None:
            if self.is_right_proj:
                return S[:, self.indices_crt]
            else:
                return S[self.indices_crt, :]
        else:
            if self.is_right_proj:
                out.copy_(S[:, self.indices_crt])
            else:
                out.copy_(S[self.indices_crt, :])
        # elif self.sim_type == 'makhoul':
        #     if out is None:
        #         if self.is_right_proj:
        #             return S[:, self.indices_crt]
        #         else:
        #             return S[:, self.indices_crt].T
        #     else:
        #         if self.is_right_proj:
        #             out.copy_(S[:, self.indices_crt])
        #         else:
        #             out.copy_(S[:, self.indices_crt].T)
        # else:
        #     raise RuntimeError(f'Unknown similarity sim_type: {self.sim_type}')

    def get_subspace_rotation_matrix(self, Q):
        assert self.rotate_states, f'The optimizer was not initialized with rotate_subspace=True'

        icrt = self.indices_crt
        iprev = self.indices_prev

        if self.is_right_proj:
            return Q[:, iprev].T @ Q[:, icrt] # (m, r).T @ (m, r) = (r, r) # with Q from MatrixStorage @ PhD #11, page 44 (same as with Qfrom optimizer state @ PhD #11, page 47)
            # return Q[iprev, :] @ Q[icrt, :].T # (r, m) @ (r, m).T = (r, r)
        else:
            # return Q[icrt, :] @ Q[iprev, :].T # (r, n) @ (r, n).T = (r, r) # with Q from MatrixStorage @ PhD #11, page 44
            return Q[:, icrt].T @ Q[:, iprev]  # (r, n) @ (r, n).T = (r, r) # with Q from optimizer state @ PhD #11, page 47
            # return Q[:, icrt].T @ Q[:, iprev] # (n, r).T @ (n, r) = (r, r)

    def rotate_subspace(self, R, w):
        assert self.rotate_states, f'The optimizer was not initialized with rotate_subspace=True'
        if self.is_right_proj:
            torch.matmul(w, R, out=w)
        else:
            torch.matmul(R, w, out=w)

    def from_higher_to_lower_dimensions(self, Q, X):
        # Q = MatrixStorage.get_matrix(self.size, self.proj, X.dtype, transpose=not self.is_right_proj)

        icrt = self.indices_crt

        if self.is_right_proj:
            return X @ Q[:, icrt] # (n, m) @ (m, r) = (n, r)
        else:
            # return Q[icrt, :] @ X # (r, n) @ (n, m) = (r, m) # with Q from MatrixStorage @ PhD #11, page 44
            return Q[:, icrt].T @ X # (n, r).T @ (n, m) = (r, m) # with Q from optimizer state @ PhD #11, page 47

    def from_lower_to_higher_dimensions(self, Q, x, out=None):
        # Q = MatrixStorage.get_matrix(self.size, self.proj, x.dtype, transpose=not self.is_right_proj)
        icrt = self.indices_crt

        if self.is_right_proj:
            # (n, r) @ (m, r).T = (n, m)
            if out is None:
                return x @ Q[:, icrt].T
            else:
                torch.matmul(x, Q[:, icrt].T, out=out)
        else:
            # (r, n).T @ (r, m) = (n, m)
            if out is None:
                # return Q[icrt, :].T @ x # with Q from MatrixStorage @ PhD #11, page 44
                return Q[:, icrt] @ x # with Q from optimizer state @ PhD #11, page 47
            else:
                # torch.matmul(Q[icrt, :].T, x, out=out) # with Q from MatrixStorage @ PhD #11, page 44
                torch.matmul(Q[:, icrt], x, out=out) # with Q from optimizer state @ PhD #11, page 47

# if self.strategy == STRATEGY_FIRST:
#     self.indices_crt.copy_(torch.arange(start=0, end=self.rank, dtype=torch.int32, device=self.device))
# elif self.strategy == STRATEGY_RANDOM:
#     self.indices_crt.copy_(torch.randperm(n=self.size, dtype=torch.int32, device=self.device)[:self.rank])
# elif self.strategy == STRATEGY_WINDOW:
#     """
#         For size=5, range2x will contain [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
#         For rank=3, the following indices will be generated:
#         step = 1: [0, 1, 2]
#         step = 2: [1, 2, 3]
#         step = 3: [2, 3, 4]
#         step = 4: [3, 4, 0]
#         step = 5: [4, 0, 1]
#         step = 6: [0, 1, 2] # here we have the same indices as for step 1 (the indices are repeated once at size steps)
#     """
#     range2x = torch.arange(self.size, dtype=torch.int32, device=self.device).repeat(1, 2).view(-1)
#     start = self.steps % self.size
#     self.indices_crt.copy_(range2x[start:start+self.rank]) # rank indices, starting at "start"
#     del range2x
