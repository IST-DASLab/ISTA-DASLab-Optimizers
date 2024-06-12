import wandb
import torch
from .sparse_core_mfac_w_ef import SparseCoreMFACwithEF
from ..tools import get_first_device, get_gpus, get_weights_and_gradients, update_model, get_gpu_mem_usage


class SparseMFAC(torch.optim.Optimizer):
    def __init__(self, params, lr: float, damp: float, m: int, k_init: float, weight_decay: float, use_bf16: bool):
        super(SparseMFAC, self).__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m
        self.damp = damp
        self.use_bf16 = use_bf16
        self.k_init = k_init

        self.device = get_first_device()
        self.d = sum([p.numel() for group in self.param_groups for p in group['params']])

        ##### Sparse M-FAC preconditioner
        self.core_mfac = SparseCoreMFACwithEF(
            m=self.m,
            d=self.d,
            k_init=self.k_init,
            dev=self.device,
            gpus=[self.device] if torch.distributed.is_initialized() else get_gpus(),
            damp=damp,
            use_bf16=use_bf16)

        ##### scalar variables
        self.steps = 0
        self.log_interval = 100
        self.grad_norms_sum = 0

        self.wandb_data = dict()
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ##################################################
        ########## [1] GET GRADIENT
        ##################################################
        g_dense = get_weights_and_gradients(self.param_groups, get_weights=False, get_grad=True, grad_bf16=self.use_bf16)
        norm_g_dense = g_dense.norm(p=2)
        self.grad_norms_sum += norm_g_dense

        ##################################################
        ########## [2] PRECONDITION
        ##################################################
        update = self.core_mfac.apply_ef_then_update_buffer_then_precondition(g_dense)

        ##################################################
        ########## [3] UPDATE THE MODEL
        ##################################################
        lr = self.param_groups[0]['lr']

        update_model(
            params=self.param_groups,
            update=update,
            weight_decay=self.weight_decay,
            alpha=None,
            multiply_wd_w_lr=True)

        ##################################################
        ########## LOGS
        ##################################################
        if self.log_interval > 0 and self.steps % self.log_interval == 0:
            norm_error = self.core_mfac.error.norm(p=2)
            self.wandb_data.update({
                'epoch/step': self.steps,
                'epoch/norm_g': norm_g_dense,
                'epoch/norm_error': norm_error,
                'epoch/ef_norm_div_grad_norm_sum': norm_error / self.grad_norms_sum,
                'epoch/norm_u': update.norm(p=2),
                'epoch/gpu_mem_usage': get_gpu_mem_usage(),
            })
            self.wandb_data.update(self.core_mfac.wandb_data)
            wandb.log(self.wandb_data)

        return loss
