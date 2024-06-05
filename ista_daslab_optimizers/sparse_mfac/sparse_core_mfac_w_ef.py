import math
import torch
from ..tools import import_cuda_module, block_split, KernelVersionsManager

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

cuda_dense_mfac = import_cuda_module('cuda_dense_mfac')
cuda_sparse_mfac = import_cuda_module('cuda_sparse_mfac')
cuda_daslab_tools = import_cuda_module('cuda_daslab_tools')

USE_CUDA = True

class SparseCoreMFACwithEF:
    def __init__(self, m, d, k_init, dev, gpus, damp, use_bf16):
        if USE_CUDA and m % 32 != 0 or m > 1024:
            raise ValueError('CUDA implementation currently on supports $m$ < 1024 and divisible by 32.')
        self.m = m
        self.d = d
        self.k_init = k_init
        self.device = dev
        self.gpus = gpus
        self.dtype = torch.float
        self.lamda = 1. / damp
        self.damp = damp
        self.use_bf16 = use_bf16

        ##### Error Feedback & Top-K related methods
        self.error = torch.zeros(self.d, dtype=torch.bfloat16 if use_bf16 else torch.float32, device=self.device)

        self.d_block_size = cuda_daslab_tools.get_max_floats_for_shared_memory_per_thread_block()
        self.k_block_size = math.ceil(self.d_block_size * self.k_init)
        self.blocks_count, self.start_index_of_last_block = block_split(self.d, self.d_block_size)
        self.k = math.ceil(self.k_block_size * self.k_init) * self.blocks_count

        if self.start_index_of_last_block == 0:
            self.last_k = 0
        else:
            last_block_size = self.d - self.start_index_of_last_block
            self.last_k = math.ceil(last_block_size * self.k_init)
            self.k += self.last_k
            print(f'Last block has size {last_block_size} and associated k for it is {self.last_k}')

        self.log_interval = 0
        self.steps = 0
        self.wandb_data = dict()

        self.gpus_count = len(self.gpus)
        self.dtype_indices = torch.uint16
        self.dtype_values = torch.bfloat16 if use_bf16 else torch.float

        self.scalar_products = torch.zeros(self.m, dtype=torch.float, device=self.device)

        self.I = torch.zeros(self.m, self.k, dtype=self.dtype_indices, device=self.device)
        self.V = torch.zeros(self.m, self.k, dtype=self.dtype_values, device=self.device)

        self.dots = torch.zeros((self.m, self.m), device=self.device, dtype=self.dtype)  # matrix $GG^T$
        self.buffer_index = 0  # ringbuffer index
        self.giHig = None # matrix $D$
        self.denom = torch.zeros(self.m, device=self.device, dtype=self.dtype)  # $D_ii + m$
        self.coef = self.lamda * torch.eye(self.m, device=self.device, dtype=self.dtype)  # matrix $B$
        self.setup()

        self.kvm = KernelVersionsManager(version_SP=23, version_LCG=51, m=self.m, d=self.d, d_block_size=self.d_block_size)

    def setup(self):
        self.giHig = self.lamda * self.dots
        diag_m = torch.diag(torch.full([self.m], self.m, device=self.device, dtype=self.dtype))
        self.giHig = torch.lu(self.giHig + diag_m, pivot=False)[0]
        self.giHig = torch.triu(self.giHig - diag_m)
        self.denom = self.m + torch.diagonal(self.giHig)
        tmp = -self.giHig.t().contiguous() / self.denom.reshape((1, -1))

        if USE_CUDA:
            diag_lambd = torch.diag(torch.full([self.m], self.lamda, device=self.device, dtype=self.dtype))
            self.coef = cuda_dense_mfac.hinv_setup(tmp, diag_lambd)
        else:
            for i in range(max(self.buffer_index, 1), self.m):
                self.coef[i, :i] = tmp[i, :i].matmul(self.coef[:i, :i])

    def _apply_ef_then_topk(self, g):
        """
            See PhD #9 page 70 for the pseudocode
        """
        self.error.add_(g) # the error feedback is the accumulator here

        self.I[self.buffer_index, :self.start_index_of_last_block] = torch.topk(
            input=self.error[0:self.start_index_of_last_block].abs().view(self.blocks_count, self.d_block_size),
            k=self.k_block_size, # k is the same for all first n-1 blocks
            sorted=False).indices # will have 2D shape: (blocks_count, self.block_size)

        if self.start_index_of_last_block < self.d:
            self.I[self.buffer_index, :self.start_index_of_last_block] = torch.topk(
                input=self.error[self.start_index_of_last_block:].abs(),
                k=self.last_k,
                sorted=False).indices

        # copy the values from the error feedback accumulator to values V (this is the G update), e.g. V[index,:] = error[I[index, :]]
        # the large tensor (error, size d) is copied to the small tensor (V, size k)
        cuda_daslab_tools.copy_values_large_to_small(self.d,
                                                     self.k,
                                                     self.d_block_size,
                                                     self.k_block_size,
                                                     self.I[self.buffer_index, :],
                                                     self.error,
                                                     self.V[self.buffer_index, :])

        # the small tensor (V, size k) is copied to the large tensor (g, size d)
        g.zero_() # this will contain the values in V, at the right indices, but will also contain zeros
        cuda_daslab_tools.copy_values_small_to_large(self.d, # this does out[I[index]] = vector
                                                     self.k,
                                                     self.d_block_size,
                                                     self.k_block_size,
                                                     self.I[self.buffer_index, :], # indices
                                                     self.V[self.buffer_index, :], # vector
                                                     g) # out

        # zerorize error: subtract the top-k values (saved in V[index, :]), which are also present in g
        self.error.sub_(g)

    def apply_ef_then_update_buffer_then_precondition(self, g):
        """
            The function name says it all
            Returns update inv(F) * g = 1/lambda * g - linear_combination_of_gradients (tmp contains linear comb params)
            :param g: the dense gradient
            :return: `the preconditioned sparse-gradient
        """
        self.steps += 1
        self._apply_ef_then_topk(g) # after this call, g will contain the top-k values and zeros
        dots = self._integrate_gradient(topk_values_w_zeros=g)
        p = self._precondition(g, dots) # here we precondition the sparse gradient, e.g. only the top-k values, stored in the d-dim tensor g
        return p

    def _integrate_gradient(self, topk_values_w_zeros):
        tmp = self.compute_scalar_products(topk_values_w_zeros)
        tmp = tmp.squeeze() # (d, 1) becomes (d,)

        self.dots[self.buffer_index, :] = tmp
        self.dots[:, self.buffer_index] = tmp

        self.setup()

        self.buffer_index = (self.buffer_index + 1) % self.m
        return tmp

    def _precondition(self, g, dots=None):
        """
            Returns the update inv(F) * x
            The matrix M stores the coefficients of the linear combination
            x: usually the sparse gradient
        """
        # print(f'[precondition]')
        if dots is None:
            dots = self.compute_scalar_products(g)
        giHix = self.lamda * dots
        if USE_CUDA:
            giHix = cuda_dense_mfac.hinv_mul(self.m, self.giHig, giHix)
            torch.cuda.synchronize()
        else:
            for i in range(1, self.m):
                giHix[i:].sub_(self.giHig[i - 1, i:], alpha=giHix[i - 1] / self.denom[i - 1])
        M = (giHix / self.denom).matmul(self.coef) # .view(-1, 1) # view is linked to matmul_grads_sequential_batch

        partA = self.lamda * g
        partB = self.compute_linear_combination(M, out=g) # out will be returned such that partB = out
        if self.steps > 0 and self.log_interval > 0 and self.steps % self.log_interval == 0:
            self.wandb_data.update(dict(norm_partA=partA.norm(p=2), norm_partB=partB.norm(p=2)))
        return partA - partB

    def compute_scalar_products(self, g):
        self.scalar_products.zero_()

        cuda_sparse_mfac.SP(
            self.kvm.get_SP_blocks(),
            self.kvm.get_SP_threads(),
            self.kvm.version_SP,
            self.d,
            min(self.m, self.steps),
            self.k,
            g,
            self.I,
            self.V,
            self.scalar_products,
            int(self.use_bf16),
            self.d_block_size,
            self.k_block_size)

        return self.scalar_products

    def compute_linear_combination(self, M, out):
        out.zero_()

        cuda_sparse_mfac.LCG(
            self.kvm.get_LCG_blocks(),
            self.kvm.get_LCG_threads(),
            self.kvm.version_LCG,
            self.d,
            min(self.m, self.steps),
            self.k,
            M,
            self.I,
            self.V,
            out,
            int(self.use_bf16),
            self.d_block_size,
            self.k_block_size)

        if self.steps > 0 and self.log_interval > 0 and self.steps % self.log_interval == 0:
            self.wandb_data.update(dict(lin_comb_coef_norm=M.norm(p=2)))

        return out