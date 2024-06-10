import torch
import torch.nn as nn
import numpy as np

USE_CUDA = True
try:
    import ista_daslab_dense_mfac
except Exception as e:
    USE_CUDA = False
    print('\n\t[WARNING] The module "ista_daslab_dense_mfac" is not installed, using slower PyTorch implementation!\n')

class DenseCoreMFAC:
    def __init__(self, grads, dev, gpus, damp=1e-5, create_G=False):
        self.m, self.d = grads.shape
        self.dev = dev
        self.gpus = gpus
        self.dtype = grads.dtype
        self.gpus = gpus
        self.grads_count = 0
        self.wandb_data = dict()
        self.damp = None
        self.lambd = None
        self.set_damp(damp)
        self.create_G = create_G
        if self.create_G:
            self.G = grads

        if USE_CUDA and self.m % 32 != 0 or self.m > 1024:
            raise ValueError('CUDA implementation currently on supports $m$ < 1024 and divisible by 32.')

        self.dper = self.d // len(gpus) + 1
        self.grads = []  # matrix $G$ in the paper
        for idx in range(len(gpus)):
            start, end = idx * self.dper, (idx + 1) * self.dper
            self.grads.append(grads[:, start:end].to(gpus[idx]))
        self.dots = torch.zeros((self.m, self.m), device=self.dev, dtype=self.dtype)  # matrix $GG^T$
        for idx in range(len(gpus)):
            self.dots += self.grads[idx].matmul(self.grads[idx].t()).to(self.dev)

        self.last = 0  # ringbuffer index
        self.giHig = self.lambd * self.dots  # matrix $D$
        self.denom = torch.zeros(self.m, device=self.dev, dtype=self.dtype)  # $D_ii + m$
        self.coef = self.lambd * torch.eye(self.m, device=self.dev, dtype=self.dtype)  # matrix $B$

        self.setup()

    def empty_buffer(self):
        for g in self.grads:
            g.zero_()

    def set_damp(self, new_damp):
        self.damp = new_damp
        self.lambd = 1. / new_damp

    def reset_optimizer(self):
        self.grads_count = 0
        for idx in range(len(self.gpus)):
            self.grads[idx].zero_()
        self.dots.zero_()
        for idx in range(len(self.gpus)):
            self.dots += self.grads[idx].matmul(self.grads[idx].t()).to(self.dev)
        self.last = 0
        self.giHig = self.lambd * self.dots  # matrix $D$
        self.denom = torch.zeros(self.m, device=self.dev, dtype=self.dtype)  # $D_ii + m$
        self.coef = self.lambd * torch.eye(self.m, device=self.dev, dtype=self.dtype)  # matrix $B$
        self.setup()

    # Calculate $D$ / `giHig` and $B$ / `coef`
    def setup(self):
        self.giHig = self.lambd * self.dots
        diag = torch.diag(torch.full(size=[self.m], fill_value=self.m, device=self.dev, dtype=self.dtype))
        self.giHig = torch.lu(self.giHig + diag, pivot=False)[0]
        self.giHig = torch.triu(self.giHig - diag)
        self.denom = self.m + torch.diagonal(self.giHig) # here we should use min(grads_count, m)
        tmp = -self.giHig.t().contiguous() / self.denom.reshape((1, -1))

        if USE_CUDA:
            diag = torch.diag(torch.full(size=[self.m], fill_value=self.lambd, device=self.dev, dtype=self.dtype))
            self.coef = ista_daslab_dense_mfac.hinv_setup(tmp, diag)
        else:
            for i in range(max(self.last, 1), self.m):
                self.coef[i, :i] = tmp[i, :i].matmul(self.coef[:i, :i])

    # Replace oldest gradient with `g` and then calculate the IHVP with `g`
    def integrate_gradient_and_precondition(self, g, x):
        tmp = self.integrate_gradient(g)
        p = self.precondition(x, tmp)
        return p

    # Replace oldest gradient with `g`
    def integrate_gradient(self, g):
        self.set_grad(self.last, g)
        tmp = self.compute_scalar_products(g)
        self.dots[self.last, :] = tmp
        self.dots[:, self.last] = tmp
        self.setup()
        self.last = (self.last + 1) % self.m
        return tmp

    # Distributed `grads[j, :] = g`
    def set_grad(self, j, g):
        # for the eigenvalue experiment:
        if self.create_G:
            self.G[j, :] = g

        self.grads_count += 1
        def f(i):
            start, end = i * self.dper, (i + 1) * self.dper
            self.grads[i][j, :] = g[start:end]

        nn.parallel.parallel_apply(
            [f] * len(self.grads), list(range(len(self.gpus)))
        )

    # Distributed `grads.matmul(x)`
    def compute_scalar_products(self, x):
        def f(i):
            start, end = i * self.dper, (i + 1) * self.dper
            G = self.grads[i]
            return G.matmul(x[start:end].to(G.device)).to(self.dev)

        outputs = nn.parallel.parallel_apply(
            [f] * len(self.gpus), list(range(len(self.gpus)))
        )
        return sum(outputs)

    # Product with inverse of dampened empirical Fisher
    def precondition(self, x, dots=None):
        if dots is None:
            dots = self.compute_scalar_products(x)
        giHix = self.lambd * dots
        if USE_CUDA:
            giHix = ista_daslab_dense_mfac.hinv_mul(self.m, self.giHig, giHix)
        else:
            for i in range(1, self.m):
                giHix[i:].sub_(self.giHig[i - 1, i:], alpha=giHix[i - 1] / self.denom[i - 1])
        """
            giHix size: 1024
            denom size: 1024
            coef size: 1024x1024
            M size: 1024
            x size: d
        """
        M = (giHix / self.denom).matmul(self.coef)
        partA = self.lambd * x
        partB = self.compute_linear_combination(M)
        self.wandb_data.update({f'norm_partA': partA.norm(p=2), f'norm_partB': partB.norm(p=2)})
        return partA.to(self.dev) - partB.to(self.dev)

    # Distributed `x.matmul(grads)`
    def compute_linear_combination(self, x):
        def f(G):
            return (x.to(G.device).matmul(G)).to(self.dev)
        outputs = nn.parallel.parallel_apply(
            [f] * len(self.grads), self.grads
        )
        """
            x size: 1024
            grads: 1024 x d
        """
        x = x.detach().cpu().numpy()
        norm = np.linalg.norm(x)
        self.wandb_data.update({f'lin_comb_coef_norm': norm})
        return torch.cat(outputs)
