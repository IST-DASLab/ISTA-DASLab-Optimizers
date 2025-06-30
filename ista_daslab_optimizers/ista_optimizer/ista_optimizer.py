import torch

class ISTAOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.lr = lr
        self.weight_decay = weight_decay
        self.optim_steps = 0

    def loop_params(self, check_grad=True):
        for group in self.param_groups:
            for p in group['params']:
                if check_grad:
                    if p.grad is None: continue
                yield group, self.state[p], p

    @torch.no_grad()
    def init_optimizer_states(self):
        raise NotImplementedError

    @torch.no_grad()
    def optimizer_step(self):
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        self.optim_steps += 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.optimizer_step()

        return loss