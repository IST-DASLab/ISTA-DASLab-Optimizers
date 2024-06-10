import math
from enum import Enum
import torch
import wandb
from .wd_scheduler import WeightDecayScheduler

class ACDC_Action(Enum):
    FILL_MASK_WITH_ONES = 1
    SPARSIFY_MODEL_AND_UPDATE_MASK = 2
    KEEP_MASK_FIXED = 3

class ACDC_Phase(Enum):
    DENSE = 1
    SPARSE = 2

class ACDC_Scheduler:
    """
        This class will hold a list of epochs where the sparse training is performed
    """
    def __init__(self, warmup_epochs, epochs, phase_length_epochs, finetuning_epochs, zero_based=False):
        """
        Builds an AC/DC Scheduler
        :param warmup_epochs: the warm-up length (dense training)
        :param epochs: total number of epochs for training
        :param phase_length_epochs: the length of dense and sparse phases (both are equal)
        :param finetuning_epochs: the epoch when the finetuning_epochs starts and is considered sparse training
        :param zero_based: True if epochs start from zero, False if epochs start from one
        """
        # print(f'AC/DC Scheduler: {warmup_epochs=}, {phase_length_epochs=}, {finetuning_epochs=}, {epochs=}')
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.phase_length_epochs = phase_length_epochs
        self.finetuning_epochs = finetuning_epochs
        self.zero_based = zero_based

        self.sparse_epochs = []
        self._build_sparse_epochs()

    def _build_sparse_epochs(self):
        is_sparse = True
        for i, e in enumerate(range(self.warmup_epochs, self.epochs)):
            if is_sparse or e >= self.finetuning_epochs:
                self.sparse_epochs.append(e)

            if (e-self.warmup_epochs) % self.phase_length_epochs == self.phase_length_epochs - 1:
            # if e % self.phase_length_epochs == self.phase_length_epochs - 1:
                is_sparse = not is_sparse

        if not self.zero_based:
            for i in range(len(self.sparse_epochs)):
                self.sparse_epochs[i] += 1

    def is_sparse_epoch(self, epoch):
        """
            Returns True if sparse training should performed, otherwise returns False
            :param epoch: a zero-based epoch number
        """
        return epoch in self.sparse_epochs

    def is_finetuning_epoch(self, epoch):
        return epoch >= self.finetuning_epochs

    def get_action(self, epoch):
        is_crt_epoch_sparse = self.is_sparse_epoch(epoch)
        is_prev_epoch_sparse = self.is_sparse_epoch(epoch-1)
        if is_crt_epoch_sparse:
            if is_prev_epoch_sparse:
                return ACDC_Action.KEEP_MASK_FIXED # mask was updated with top-k at the first dense epoch and now do nothing
            return ACDC_Action.SPARSIFY_MODEL_AND_UPDATE_MASK # first dense epoch, update the mask using topk
        else: # dense epoch
            if epoch == int(not self.zero_based) or is_prev_epoch_sparse:
                return ACDC_Action.FILL_MASK_WITH_ONES  # fill mask with ones
            else:
                return ACDC_Action.KEEP_MASK_FIXED # do not change mask

    def get_phase(self, epoch):
        if self.is_sparse_epoch(epoch):
            return ACDC_Phase.SPARSE
        return ACDC_Phase.DENSE

class ACDC(torch.optim.Optimizer):
    """
        This class implements the Default AC/DC schedule from the original paper https://arxiv.org/abs/2106.12379.pdf:
        We use the model only to add names for the parameters in wandb logs and print the sparsity, norm and weight decay
        For example, fc.bias (1D, that requires weight decay) and ln/bn.weight/bias (1D, that do not require weight decay).
            * LN = Layer Normalization
            * BN = Batch Normalization
        Example for 100 epochs (from original AC/DC paper https://arxiv.org/pdf/2106.12379.pdf)
            - first 10 epochs warmup (dense training)
            - alternate sparse/dense training phases once at 5 epochs
            - last 10 epochs finetuning (sparse training)
            !!!!! SEE THE HUGE COMMENT AT THE END OF THIS FILE FOR A MORE DETAILED EXAMPLE !!!!!

        To use this class, make sure you call method `update_acdc_state` at the beginning of each epoch.

        The following information is required:
            - params
            - model
            - momentum
            - weight_decay
            - wd_type
            - k
            - total_epochs
            - warmup_epochs
            - phase_length_epochs
            - finetuning_epochs
    """
    def __init__(self,
                 params, model, # optimization set/model
                 lr, momentum, weight_decay, wd_type, k, # hyper-parameters
                 total_epochs, warmup_epochs, phase_length_epochs, finetuning_epochs): # acdc schedulers
        super(ACDC, self).__init__(params, defaults=dict(lr=lr, weight_decay=weight_decay, momentum=momentum, k=k))

        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.wd_type = wd_type
        self.k = k

        self.acdc_scheduler = ACDC_Scheduler(
            warmup_epochs=warmup_epochs,
            epochs=total_epochs,
            phase_length_epochs=phase_length_epochs,
            finetuning_epochs=finetuning_epochs,
            zero_based=False)

        self.phase = None
        self.is_finetuning_epoch = False
        self.update_mask_flag = None

        self.current_epoch = 0
        self.steps = 0
        self.log_interval = 250

        self._initialize_param_states()

    def _initialize_param_states(self):
        for group in self.param_groups:
            for p in group['params']:

                # this method is called before the first forward pass and all gradients will be None
                # if p.grad is None:
                #     continue

                state = self.state[p]

                # initialize the state for each individual parameter p
                if len(state) == 0:
                    # v is the momentum buffer
                    state['v'] = torch.zeros_like(p)

                    # set density to be used in top-k call (only for multi-dim tensors)
                    state['density'] = int(self.k * p.numel())

                    # 1D tensors, like:
                    # - batch/layer normalization layers
                    # - biases for other layers
                    # will always have mask=1 because they will never be pruned
                    state['mask'] = torch.ones_like(p)

                    # set the weight decay scheduler for each parameter individually
                    # all biases and batch/layer norm layers are not decayed
                    if len(p.shape) == 1:
                        state['wd_scheduler'] = WeightDecayScheduler(weight_decay=0, wd_type='const')
                    else:
                        state['wd_scheduler'] = WeightDecayScheduler(weight_decay=self.weight_decay, wd_type=self.wd_type)

    @torch.no_grad()
    def update_acdc_state(self, epoch):
        self.current_epoch = epoch
        phase = self.acdc_scheduler.get_phase(self.current_epoch)
        action = self.acdc_scheduler.get_action(self.current_epoch)

        print(f'{epoch=}, {phase=}')

        self._set_phase(phase)

        if action == ACDC_Action.FILL_MASK_WITH_ONES:
            for group in self.param_groups:
                for p in group['params']:
                    self.state[p]['mask'].fill_(1)
        elif action == ACDC_Action.SPARSIFY_MODEL_AND_UPDATE_MASK:
            # update mask and sparsify model
            for group in self.param_groups:
                for p in group['params']:
                    is_multi_dim = (len(p.shape) > 1)
                    if is_multi_dim:
                        state = self.state[p]
                        # original_shape = p.shape.clone()
                        indices = torch.topk(p.reshape(-1).abs(), k=state['density']).indices

                        # zerorize, view mask as 1D and then set 1 to specific indices. The result will have p.shape
                        state['mask'].zero_().reshape(-1)[indices] = 1.

                        # apply the mask to the parameters
                        p.mul_(state['mask'])
        elif action == ACDC_Action.KEEP_MASK_FIXED:
            pass  # do nothing

    def _zerorize_momentum_buffer(self):
        # zerorize momentum buffer only for the multi-dimensional parameters
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['v'].zero_()

    def _set_phase(self, phase):
        if phase != self.phase:
            self.phase = phase
            if self.phase == ACDC_Phase.DENSE:
                # AC/DC: zerorize momentum buffer at the transition SPARSE => DENSE
                # The following quote is copy-pasted from original ACDC paper, page 7:
                # "We reset SGD momentum at the beginning of every decompression phase."
                self._zerorize_momentum_buffer()

    @torch.no_grad()
    def _wandb_log(self):
        if self.steps % self.log_interval == 0:
            wandb_dict = dict()

            total_params = 0
            global_sparsity = 0
            global_params_norm = 0
            global_grad_norm = 0
            for name, p in self.model.named_parameters():
                total_params += p.numel()
                crt_sparsity = (p == 0).sum().item()
                norm_param = p.norm(p=2)
                norm_grad = p.grad.norm(p=2)

                wandb_dict[f'weight_sparsity_{name}'] = crt_sparsity / p.numel() * 100.
                wandb_dict[f'mask_sparsity_{name}'] = (self.state[p]['mask'] == 0).sum().item() / p.numel() * 100.
                wandb_dict[f'norm_param_{name}'] = norm_param
                wandb_dict[f'norm_grad_{name}'] = norm_grad

                if self.wd_type == 'awd':
                    wandb_dict[f'awd_{name}'] = self.state[p]['wd_scheduler'].get_wd()

                global_params_norm += norm_param ** 2
                global_grad_norm += norm_grad ** 2
                global_sparsity += crt_sparsity

            wandb_dict[f'global_params_norm'] = math.sqrt(global_params_norm)
            wandb_dict[f'global_grad_norm'] = math.sqrt(global_grad_norm)
            wandb_dict[f'global_sparsity'] = global_sparsity / total_params * 100.

            wandb_dict['optimizer_epoch'] = self.current_epoch
            wandb_dict['optimizer_step'] = self.steps

            wandb_dict['is_dense_phase'] = int(self.phase == ACDC_Phase.DENSE)
            wandb_dict['is_sparse_phase'] = int(self.phase == ACDC_Phase.SPARSE)
            wandb.log(wandb_dict)

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue

                # holds all buffers for the current parameter
                state = self.state[p]

                ### apply mask to the gradient on sparse phase
                ### do not modify gradient in place via p.grad.mul_(state['mask'])
                ### because this will affect the norm statistics in self._wandb_log
                ### this will create intermediary tensors
                # grad = p.grad
                if self.phase == ACDC_Phase.SPARSE:
                    # grad = grad * state['mask']
                    p.grad.mul_(state['mask']) # sparsify gradient

                state['v'].mul_(momentum).add_(p.grad)

                wd = state['wd_scheduler'](w=p, g=p.grad) # use sparsified gradient
                p.mul_(1 - lr * wd).sub_(other=state['v'], alpha=lr).mul_(state['mask'])

        self._wandb_log()

"""
This is what ACDC_Scheduler outputs for the default ACDC schedule, presented at page 5, Figure 1 in the paper https://arxiv.org/pdf/2106.12379.pdf

epoch	is_sparse_epoch	phase	            action
1		False			ACDC_Phase.DENSE	ACDC_Action.FILL_MASK_WITH_ONES
2		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
3		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
4		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
5		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
6		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
7		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
8		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
9		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
10		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
11		True			ACDC_Phase.SPARSE	ACDC_Action.SPARSIFY_MODEL_AND_UPDATE_MASK
12		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
13		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
14		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
15		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
16		False			ACDC_Phase.DENSE	ACDC_Action.FILL_MASK_WITH_ONES
17		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
18		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
19		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
20		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
21		True			ACDC_Phase.SPARSE	ACDC_Action.SPARSIFY_MODEL_AND_UPDATE_MASK
22		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
23		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
24		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
25		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
26		False			ACDC_Phase.DENSE	ACDC_Action.FILL_MASK_WITH_ONES
27		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
28		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
29		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
30		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
31		True			ACDC_Phase.SPARSE	ACDC_Action.SPARSIFY_MODEL_AND_UPDATE_MASK
32		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
33		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
34		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
35		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
36		False			ACDC_Phase.DENSE	ACDC_Action.FILL_MASK_WITH_ONES
37		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
38		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
39		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
40		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
41		True			ACDC_Phase.SPARSE	ACDC_Action.SPARSIFY_MODEL_AND_UPDATE_MASK
42		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
43		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
44		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
45		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
46		False			ACDC_Phase.DENSE	ACDC_Action.FILL_MASK_WITH_ONES
47		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
48		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
49		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
50		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
51		True			ACDC_Phase.SPARSE	ACDC_Action.SPARSIFY_MODEL_AND_UPDATE_MASK
52		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
53		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
54		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
55		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
56		False			ACDC_Phase.DENSE	ACDC_Action.FILL_MASK_WITH_ONES
57		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
58		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
59		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
60		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
61		True			ACDC_Phase.SPARSE	ACDC_Action.SPARSIFY_MODEL_AND_UPDATE_MASK
62		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
63		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
64		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
65		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
66		False			ACDC_Phase.DENSE	ACDC_Action.FILL_MASK_WITH_ONES
67		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
68		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
69		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
70		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
71		True			ACDC_Phase.SPARSE	ACDC_Action.SPARSIFY_MODEL_AND_UPDATE_MASK
72		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
73		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
74		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
75		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
76		False			ACDC_Phase.DENSE	ACDC_Action.FILL_MASK_WITH_ONES
77		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
78		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
79		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
80		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
81		True			ACDC_Phase.SPARSE	ACDC_Action.SPARSIFY_MODEL_AND_UPDATE_MASK
82		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
83		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
84		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
85		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
86		False			ACDC_Phase.DENSE	ACDC_Action.FILL_MASK_WITH_ONES
87		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
88		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
89		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
90		False			ACDC_Phase.DENSE	ACDC_Action.KEEP_MASK_FIXED
91		True			ACDC_Phase.SPARSE	ACDC_Action.SPARSIFY_MODEL_AND_UPDATE_MASK
92		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
93		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
94		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
95		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
96		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
97		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
98		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
99		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
100		True			ACDC_Phase.SPARSE	ACDC_Action.KEEP_MASK_FIXED
"""