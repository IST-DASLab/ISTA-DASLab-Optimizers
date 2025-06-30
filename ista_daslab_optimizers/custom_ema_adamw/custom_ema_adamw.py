import torch
import math

from ..ista_optimizer import ISTAOptimizer
from ..utils import ema_standard_schedule, ema_delayed_decay_schedule

STATE_M = 'mt'
STATE_V = 'vt'

EMA_STANDARD = 'standard'
EMA_DELAYED = 'delayed'

EMA_SCHEDULES = [
    EMA_STANDARD,
    EMA_DELAYED,
]

class CustomEmaAdamW(ISTAOptimizer):

    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8,
                 *,
                 ema_schedule=EMA_STANDARD, ema_decay=5, ema_alpha=0.001):
        super().__init__(params, lr, weight_decay)
        self.betas = betas
        self.eps = eps

        self.ema_decay = ema_decay
        self.ema_alpha = ema_alpha

        assert ema_schedule in EMA_SCHEDULES
        self.ema_schedule = ema_schedule
        if ema_schedule == EMA_DELAYED:
            self.beta_prev = 0

        self.init_optimizer_states()

    @torch.no_grad()
    def init_optimizer_states(self):
        for group, state, p in self.loop_params(check_grad=False):
            state[STATE_M] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state[STATE_V] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def optimizer_step(self):
        t = self.optim_steps
        beta1, beta2 = self.betas
        eps = self.eps

        # bias correction terms
        bc1 = 1 - beta1 ** t
        bc2_sqrt = math.sqrt(1 - beta2 ** t)

        for group, state, p in self.loop_params():
            lr = group['lr']
            wd = group.get('weight_decay', self.weight_decay)

            ##### shortcuts
            g = p.grad
            m = state[STATE_M]
            v = state[STATE_V]

            ##### update momentum
            ### momentum for m (modified in-place in the ema_* methods)
            if self.ema_schedule == EMA_STANDARD:
                ema_standard_schedule(m, g, beta1)
            elif self.ema_schedule == EMA_DELAYED:
                self.beta_prev = ema_delayed_decay_schedule(m, g, beta1, self.beta_prev, t, self.ema_decay, self.ema_alpha)

            ### momentum for v
            v.mul_(beta2).addcmul_(g, g, value=1-beta2)

            ##### compute denominator in the gradient buffer to save memory
            g.copy_(v).sqrt_().div_(bc2_sqrt).add_(eps)
            # denom = (v.sqrt() / bc2_sqrt).add_(eps) # this allocates new memory => slower

            ##### weight decay
            if wd > 0:
                p.mul_(1 - lr * wd)

            ##### parameter update
            p.addcdiv_(m, g, value=-lr / bc1)
