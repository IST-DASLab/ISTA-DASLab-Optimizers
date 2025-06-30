import torch
import math

from ..utils.matrix_storage import MatrixStorage, ALL_PROJ, PROJ_DCT, PROJ_HDM
from ..utils import SCION_NORM_DICT
from ..ista_optimizer import ISTAOptimizer

STATE_ADAM_M = 'adam-m'
STATE_ADAM_V = 'adam-v'
STATE_SOAP_LEFT_PREC = 'soap-left-preconditioner'
STATE_SOAP_RIGHT_PREC = 'soap-right-preconditioner'
STATE_SCION_MOMENTUM = 'scion-momentum'
STATE_SOAP_LEFT_EVS = 'soap-left-eigen-vectors'
STATE_SOAP_RIGHT_EVS = 'soap-right-eigen-vectors'

PROJ_EVD = 'evd' # Eigen-Value-Decomposition


class ScionedSOAP(ISTAOptimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8,
                 *,
                 soap_freq=10, soap_proj=PROJ_DCT, soap_rank='full', soap_beta_left=0.9, soap_beta_right=0.9,
                 scion_momentum=0.9, scion_constrained=True, max_shape=32_000):
        assert soap_proj in ALL_PROJ + [PROJ_EVD]
        super().__init__(params, lr, weight_decay=0)
        self.betas = betas
        self.eps = eps

        self.soap_freq = soap_freq
        self.soap_proj = soap_proj
        self.soap_rank = soap_rank
        self.soap_beta_left = soap_beta_left
        self.soap_beta_right = soap_beta_right

        self.scion_constrained = scion_constrained # Scion constrained
        self.scion_momentum = scion_momentum

        self.max_shape = max_shape

        self.init_optimizer_states()

    def is_large_layer(self, p):
        n, m = p.shape
        return n >= self.max_shape or m >= self.max_shape

    @torch.no_grad()
    def init_optimizer_states(self):
        for group, state, p in self.loop_params(check_grad=False):
            state[STATE_ADAM_M] = torch.zeros_like(p)
            state[STATE_ADAM_V] = torch.zeros_like(p)

            if self.scion_momentum > 0:
                state[STATE_SCION_MOMENTUM] = torch.zeros_like(p)

            if p.ndim == 2:
                if self.is_large_layer(p): # apply adamw because one dimension > max_shape (embedding or lm_head)
                    pass # do nothing because p is either embedding or lm_head
                else: # update for 2D tensors that have both dimensions < max_shape
                    n, m = p.shape
                    state[STATE_SOAP_LEFT_PREC] = torch.zeros(n, n, dtype=p.dtype, device=p.device)
                    state[STATE_SOAP_RIGHT_PREC] = torch.zeros(m, m, dtype=p.dtype, device=p.device)
                    if self.soap_proj == 'evd':
                        state[STATE_SOAP_LEFT_EVS] = torch.zeros(n, n, dtype=p.dtype, device=p.device)
                        state[STATE_SOAP_RIGHT_EVS] = torch.zeros(m, m, dtype=p.dtype, device=p.device)
                    elif self.soap_proj in ALL_PROJ:
                        MatrixStorage.add_matrix(n, self.soap_proj, p.dtype)
                        MatrixStorage.add_matrix(m, self.soap_proj, p.dtype)

    @torch.no_grad()
    def optimizer_step(self):
        ### shortcuts
        t = self.optim_steps
        beta1, beta2 = self.betas
        eps = self.eps
        bc1 = 1 - beta1 ** t
        bc2_sqrt = math.sqrt(1 - beta2 ** t)
        beta_left, beta_right = self.soap_beta_left, self.soap_beta_right

        for group, state, p in self.loop_params():
            ### group shortcuts
            lr = group['lr']
            scale = group.get('scale', 1)
            norm_backend = SCION_NORM_DICT[group['norm']](**group['norm_kwargs'])

            ### state shortcuts
            G = p.grad
            m = state[STATE_ADAM_M]
            v = state[STATE_ADAM_V]
            if self.scion_momentum > 0:
                scion_m = state[STATE_SCION_MOMENTUM]

            if p.ndim == 1 or self.is_large_layer(p):
                m.lerp_(G, 1 - beta1)
                v.mul_(beta2).addcmul_(G, G, value=1 - beta2) # after this step, G is not needed and we can reuse it
                update = G.copy_(v).sqrt_().div_(bc2_sqrt).add_(eps).div_(bc1).div_(m).reciprocal_()
            elif p.ndim == 2:
                L = state[STATE_SOAP_LEFT_PREC]
                R = state[STATE_SOAP_RIGHT_PREC]
                if self.soap_proj == PROJ_EVD:
                    QL = state[STATE_SOAP_LEFT_EVS] # left eigenvectors of EVD
                    QR = state[STATE_SOAP_RIGHT_EVS] # right eigenvectors of EVD

                ### STEP 4: update L
                L.lerp_(G @ G.T, 1 - beta_left)

                ### STEP 5: update R
                R.lerp_(G.T @ G, 1 - beta_right)

                ### STEPS 6,7,8,9: update QL and QR
                if t == 1 or t % self.soap_freq == 0: # update
                    if self.soap_proj == PROJ_EVD:
                        _, Q = torch.linalg.eigh(L)
                        QL.copy_(Q)

                        _, Q = torch.linalg.eigh(R)
                        QR.copy_(Q)

                        del Q

                ### STEP 10: rotate gradient
                Gtilde = QL.T @ G @ QR

                ### STEP 11: integrate rotated gradient into the first moment EMA
                m.lerp_(Gtilde, 1 - beta1)

                ### STEP 12: integrate rotated gradient into the second moment EMA
                v.mul_(beta2).addcmul_(Gtilde, Gtilde, value=1 - beta2) # after this step, Gtilde is not needed and we can reuse it

                ### STEP 13: compute Adam update (bias corrected)
                ##### SUB-STEP 13.1: save denominator of Adam update in G to save memory
                G.copy_(v).sqrt_().div_(bc2_sqrt).add_(eps)

                ##### SUB-STEP 13.2: compute the effective bias-corrected Adam update and save it into Gtilde
                Dtilde = Gtilde.copy_(m).div_(bc1).div_(G) # Gtilde will hold the Adam update, named now Dtilde for Delta_tilde

                ### STEP 14: project Adam update back and save it into G (named now D for Delta)
                D = G.copy_(QL @ Dtilde @ QR.T)

                ### STEP 15: compute LMO
                update = Gtilde.copy_(norm_backend.lmo(D)) # save LMO to Gtilde and name it update
            else:
                raise RuntimeError("ScionedSOAP currently supports only 1D and 2D tensors!")

            ### STEP 16: constrained / unconstrained
            if self.scion_momentum > 0:
                update = scion_m.lerp_(update, 1 - self.scion_momentum)

            if self.scion_constrained:
                p.mul_(1 - lr)
            p.add_(update, alpha=-lr / scale)

    @torch.no_grad()
    def init(self):
        for group, state, p in self.loop_params(check_grad=False):
            norm_backend = SCION_NORM_DICT[group['norm']](**group['norm_kwargs'])
            norm_backend.init(p)

            scale = group.get('scale', 1)
            p.data.mul_(scale)

    @torch.no_grad()
    def project(self):
        pass

    @torch.no_grad()
    def update_preconditioner(self):
        pass
