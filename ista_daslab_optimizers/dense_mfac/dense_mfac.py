import wandb
import torch
from ..tools import get_weights_and_gradients, update_model, get_first_device, get_gpus
from .dense_core_mfac import DenseCoreMFAC

# Disable tensor cores as they can mess with precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class DenseMFAC(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr: float,
                 weight_decay: float,
                 ngrads: int,
                 damp: float,
                 create_G=False):

        super(DenseMFAC, self).__init__(params, dict(lr=lr, weight_decay=weight_decay))

        self.m = ngrads
        self.lr = lr
        self.damp = damp
        self.weight_decay = weight_decay
        self.device = get_first_device()

        self.model_size = None
        self.steps = 0
        self.wandb_data = dict()



        self.model_size = sum([p.numel() for group in self.param_groups for p in group['params']])
        print(f'Model size: {self.model_size}')

        self.hinv = DenseCoreMFAC(
            grads=torch.zeros((ngrads, self.model_size), dtype=torch.float),
            dev=self.device,
            gpus=get_gpus(),
            damp=damp,
            create_G=create_G)

    @torch.no_grad()
    def empty_buffer(self):
        self.hinv.empty_buffer()


    @torch.no_grad()
    def integrate_gradient(self, g):
        _ = self.hinv.integrate_gradient(g)

    @torch.no_grad()
    def compute_update(self, g, x):
        update_method = self.hinv.integrate_gradient_and_precondition
        # if self.use_sq_newton:
        #     update_method = self.hinv.integrate_gradient_and_precondition_twice

        update = update_method(g, x).to(self.device)
        return update

    @torch.no_grad()
    def log_data(self, update, g):
        lr = self.param_groups[0]['lr']
        self.wandb_data.update(dict(norm_upd_w_lr=lr * update.norm(p=2), norm_g=g.norm(p=2)))
        self.wandb_data.update(self.hinv.wandb_data)
        # self.wandb_data.update(quantify_preconditioning(g=g, u=update.to(g.device), return_distribution=False, use_abs=True, optim_name=self.optim_name))
        wandb.log(self.wandb_data)

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1

        g = get_weights_and_gradients(self.param_groups, get_weights=False)
        update = self.compute_update(g, g)

        update_model(params=self.param_groups, update=update, alpha=None)

        if self.steps % self.m == 0:
            self.log_data(update, g)


# def kfac_update_rescaling(self, g, u):
#     # rescaling_kfac64,  # use the alpha rescaling from K-FAC paper, section 6.4
#     T1 = torch.dot(g, u)
#     T2 = (self.hinv.grads_matmul(u).norm(p=2) ** 2).mean()
#     T3 = self.damp * (u.norm(p=2) ** 2)
#     alpha = -T1 / (T2 + T3)
#     self.wandb_data.update(dict(kfac_T1=T1, kfac_T2=T2, kfac_T3=T3, kfac_alpha=alpha))
#     return alpha
