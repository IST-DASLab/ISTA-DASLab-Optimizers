import torch
from torch import Tensor, bmm
import torch.distributed as dist
from functools import partial
import traceback
import wandb
import sys

from ..invertors import DashRootInvertor
from ..partitioners import DashGpuPartitioner
from ..tools import DashMultiShape, DashShapesCalculator, DashMatrixBlock, DashStackedBlocksHandler
from ..types import DashGraftingType, DashPartitionInfo
from ..dash_config import DashConfig

STATE_DASH_SHAPE = 'dash_shape'  # saved in the state

class DashGpuProcessor1D:
    """
        Saves the states of Shampoo optimizer for a all 1D layers assigned to the current GPU.

        Arguments:
            bucket (callable): a function which, once called, returns a list [(index, group, state, p)], where:
                index = index of the layer from enumerate(model.parameters)
                group = the group of the parameter
                state = the state of the parameter
                p     = the parameter itself
    """

    def __init__(self, bucket_func, cfg: DashConfig):
        self.bucket_func = bucket_func
        self.cfg = cfg
        self.name = '1d'

        self.shape_stacked: DashMultiShape = None
        self.shape_raw = None

        self.G = None
        self.L = None  # 1D layers only have L
        self.invL = None # inverse of L
        self.A = None  # grafting buffer
        self.Pshmp = None # shampoo direction
        self.P = None  # shampoo update
        self.Pgraft_fro = None  # frobenius norm of grafting update (G or tildeG) / (eps + sqrt(A))
        self.tildeG = None  # momentum for gradient
        self.M = None  # momentum for shampoo update

        self._initialize()

    @torch.no_grad()
    def _initialize(self):
        cfg = self.cfg
        assert cfg.grafting_type == DashGraftingType.ADAM
        B = cfg.block_size

        p0 = next(self.bucket_func())[-1]  # the generator bucket_func() yields (index, group, state, p)
        dtype = p0.dtype
        device = p0.device

        DSBH = partial(DashStackedBlocksHandler, block_size=B, dtype=dtype, device=device)
        DSBH_like = DashStackedBlocksHandler.like

        zeros = partial(torch.zeros, dtype=dtype, device=device, requires_grad=False)
        zeros_like = torch.zeros_like

        rank = dist.get_rank() if dist.is_initialized() else 0

        # this is the shape that results after stacking the normalization layers
        self.shape_raw: DashShape3D = DashShapesCalculator.get_param_shape_of_merged_norm_layers(self.bucket_func)  # (N, E, 1)
        # this is a multi-shape object that specifies the shapes of each (G/L/R/LR)(full/rest) construction
        self.shape_stacked: DashMultiShape = DashShapesCalculator.get_stacked_shapes_for_merged_norm_layers(self.shape_raw, B)

        print(f'[rank={rank}] shape_full_1d: '
              f'shape_raw: {self.shape_raw} '
              f'shape_stacked: {self.shape_stacked}')

        self.G = zeros(self.shape_stacked.Gfull.as_tuple())
        self.L = zeros(self.shape_stacked.Lfull.as_tuple())
        self.invL = zeros_like(self.L)
        self.A = zeros_like(self.G)
        self.Pshmp = zeros_like(self.G)
        self.P = zeros_like(self.G)
        self.Pgraft_fro = zeros((self.G.shape[0], 1, 1))
        if cfg.beta_G > 0: self.tildeG = zeros_like(self.G)
        if cfg.mu > 0: self.M = zeros_like(self.G)

    # end _initialize

    @torch.no_grad()
    def update_layer(self, t, lr):
        self._copy_gradient()
        self._update_factors()
        self._update_grafting()
        self._invert_factors(t)
        self._update_grad_ema()
        self._compute_grafting_direction(t)
        self._compute_shampoo_direction(t)
        self._apply_momentum_with_nesterov_then_update_weights(lr)

    @torch.no_grad()
    def _copy_gradient(self):
        """
        This function copies the gradients from p.grad to G (there is no rest
        Copy gradients from buckets to merged states that replaces the following
        line from layer_processor:

        self.block_partitioner.populate_gradient_block_partition(p.grad, self.G)

        GPU rank #0:
            [1D] shape_merged_norm_layers_1d: DashShape3D(b=4, m=2048, n=1)
            [1D] shape_full_1d: DashMultiShape(
                        Gfull=DashShape3D(b=8, m=1024, n=1),
                        Grest=None,
                        Lfull=DashShape3D(b=8, m=1024, n=1024),
                        Rfull=DashShape3D(b=8, m=1, n=1),
                        Lrest=None,
                        Rrest=None,
                        LRfull=DashShape3D(b=8, m=1024, n=1024),
                        LRrest=None,
                        stats=BlockStats(
                            R_full=2048,
                            C_full=None,
                            num_row_blocks=2,
                            num_col_blocks=None,
                            num_blocks_full=8,
                            row_rest=0,
                            col_rest=None),
                        info=<DashPartitionInfo.REGULAR_BLOCK: 1>)
        """
        cfg = self.cfg
        B = cfg.block_size
        # copy gradient from the p.grad parameter to Gfull/rest buffer

        N = self.shape_raw[0]  # number of 1D params
        E = self.shape_raw[1]  # embedding size

        # when N = 4 and shape_stacked.Gfull=DashShape3D(b=8, m=1024, n=1), then
        # g_view becomes DashShape3D(b=2, m=1024, n=1)
        g_view = self.shape_stacked.Gfull // N
        batch_size = g_view.b  # this will be 2
        gi = 0  # pointer in the self.G_full buffer
        for pi, group, state, p in self.bucket_func():  # gi/pi = global/param index
            self.G[gi: gi + batch_size].copy_(p.grad.view(g_view.as_tuple()))
            gi += batch_size  # advance pointer

    @torch.no_grad()
    def _update_factors(self):
        """
        We use betaLR instead of beta2 in our implementation to know exactly where beta is used.
        We perform these updates in grouped blocks (see DashLayerwisePartitioner class).

        We use efficient grouping and save L and R blocks in self.LR

        Algorithm for Shampoo:
            if betaLR < 1 then
                L_t = betaLR L_t-1 + (1 - betaLR) G_t     @ G_t ^ T
                R_t = betaLR R_t-1 + (1 - betaLR) G_t ^ T @ G_t
            else
                L_t = L_t-1 + G_t     @ G_t ^ T
                R_t = R_t-1 + G_t ^ T @ G_t
            end if
        Algorithm for Jorge:
            This function should update X_L/R from Jorge
            - L/R from Shampoo will store X_L/R from Jorge
            - barL/barR from Shampoo will store hat(L)/hat(R) from Jorge (the estimations of inverse 4-th root)

            L_t = hat(L)_t-1 ^ 4   @   G_t     @   G_t^T (this is X_L)
            R_t = hat(R)_t-1 ^ 4   @   G_t^T   @   G_t   (this is X_R)
        """
        cfg = self.cfg
        betaLR = cfg.beta_LR
        G = self.G
        L = self.L

        G_T = G.transpose(1, 2)
        GGT = bmm(G, G_T)

        if betaLR < 1:
            L.lerp_(GGT, weight=1 - betaLR)
        else:
            L.add_(GGT)

    @torch.no_grad()
    def _update_grafting(self):
        """
        Algorithm:
            A_t = UpdateGraftingState(A_t-1, G_t).
            For Adam, we keep track of the second moment of the gradient using EMA.
            For AdaGrad, we keep track of the 2nd moment of the gradient using plain summation (no EMA).
            We use raw gradient (p.grad and not the momentum tilde(G)) to update grafting.

            Algorithm for Adam grafting (page 9 in Distributed Shampoo paper):
                P_t,graft_Adam = G_t / (tilde(A)_t^1/2 + eps)
                tilde(A)_t = Adam's second moment of the gradient
        """
        grafting_type = self.cfg.grafting_type
        if grafting_type != DashGraftingType.ADAM:
            raise NotImplementedError(f'Grafting type {grafting_type} is currently not supported, use Adam grafting for now!')

        A = self.A
        G = self.G
        beta_graft = self.cfg.beta_graft

        A.mul_(beta_graft).addcmul_(G, G, value=1-beta_graft)

    @torch.no_grad()
    def _invert_factors(self, t):
        """
        ComputeMatrixRootInverse is a generic function. In our case, it can be:
            - torch.linalg.eigh, followed by some spectrum post-processing and matmuls
            - Chebyshev
            - Newton-DB
            - Jorge
        Algorithm:
            if 𝑡 ≥ start_preconditioning_step and 𝑡 % precondition_frequency = 0:
                bar(L)_t =  ComputeMatrixRootInverse(L_t , eps, t, use_bias_correction)
                bar(R)_t =  ComputeMatrixRootInverse(R_t , eps, t, use_bias_correction)
            end if
        """
        cfg = self.cfg
        start_prec_step = cfg.start_prec_step
        inv_root_freq = cfg.inv_root_freq
        if t >= start_prec_step and (t == 1 or t % inv_root_freq == 0):
            DashRootInvertor.invert(Xin=self.L, Xout=self.invL, cfg=cfg, root=2)

    @torch.no_grad()
    def _update_grad_ema(self):
        """
        We use betaG instead of beta1 to avoid confusion.
        Algorithm:
            if betaG > 0:
                tilde(G)_t = betaG tilde(G)_t-1 + (1 - betaG) G
            end if
        """
        betaG = self.cfg.beta_G
        if betaG > 0:
            self.tildeG.lerp_(self.G, weight=1-betaG)

    @torch.no_grad()
    def _compute_grafting_direction(self, t):
        """
        This function computes the frobenius norm of the grafting direction P_t,graft_Adam = ComputeGraftingDirection(tilde(G)_t, t, use_bias_correction)
        Algorithm for Adam grafting (page 9 in Distributed Shampoo paper):
            P_t,graft_Adam = tilde(G)_t / (tilde(A)_t^1/2 + eps)
            tilde(A)_t = Adam's second moment of the gradient
        """
        cfg = self.cfg
        graft_type = cfg.grafting_type

        if graft_type != DashGraftingType.ADAM:
            raise NotImplementedError(f'Grafting type {graft_type} is currently not supported, use Adam grafting for now!')

        betaG = cfg.beta_G
        beta_graft = cfg.beta_graft
        eps_graft = cfg.eps_grafting
        use_bias_correction = cfg.use_bias_correction
        start_prec_step = cfg.start_prec_step

        G = self.G
        A = self.A
        Pshmp = self.Pshmp
        Pgraft_fro = self.Pgraft_fro
        tildeG = self.tildeG

        use_momentum_G = (betaG > 0)
        chosenG = tildeG if use_momentum_G else G

        biasG = (1 - betaG ** t) if use_bias_correction and use_momentum_G else 1
        biasA = (1 - beta_graft ** t) if use_bias_correction else 1

        Pgraft = (chosenG / biasG) / (eps_graft + (A / biasA).sqrt())
        normPgraft = Pgraft.norm(p='fro', dim=(1, 2), keepdim=True)
        Pgraft_fro.copy_(normPgraft)
        if t < start_prec_step:  # anticipate the next step: compute shampoo direction here because we have access to Pgrafr_full/rest
            Pshmp.copy_(Pgraft)

    @torch.no_grad()
    def _compute_shampoo_direction(self, t):
        """
        This function computes the effective Shampoo direction.
        Algorithm:
            if t >= start_preconditioning_step:
                U_t,shmp = bar(L)_t @ tilde(G)_t @ bar(R)_t
                P_t = (||P_t,graft|| / ||U_t,shmp||) * U_t,shmp
        """
        start_prec_step = self.cfg.start_prec_step
        if t >= start_prec_step:
            betaG = self.cfg.beta_G
            use_momentum_G = (betaG > 0)

            G = self.G
            invL = self.invL
            Pshmp = self.Pshmp
            Pgraft_fro = self.Pgraft_fro
            tildeG = self.tildeG

            chosenG = tildeG if use_momentum_G else G

            # compute the unscaled shampoo update L^-1/2 @ G
            Ushmp = invL @ chosenG

            # compute the scaling
            scaling = Pgraft_fro / Ushmp.norm(p='fro', dim=(1, 2), keepdim=True)

            # rescale in-place
            Ushmp.mul_(scaling)

            # update the final direction that will be used for the model
            Pshmp.copy_(Ushmp)
        else:
            """
            Here we should use grafting direction.
            However, right here we do nothing because we do not have access to grafting directions.
            Instead, we treat this case in the function `_compute_grafting_direction` because there we explicitly compute the grafting directions.
            Therefore, check the statements `if t < start_prec_step` in function `_compute_grafting_direction`, where we copy the grafting direction to P.
            """
            pass

    @torch.no_grad()
    def _apply_momentum_with_nesterov_then_update_weights(self, lr):
        """
        Apply momentum on P_t and nesterov if it is enabled.
        Update weights given the final update P_t
        Algorithm:
            ### apply momentum
            if mu > 0:
                M_t = mu * M_t + P_t
                if use_nesterov:
                    P_t = mu * M_t + P_t
                else:
                    P_t = M_t
            end if

            ### update model
            W_t+1 = W_t - lr * P_t
        """
        mu = self.cfg.mu
        use_nesterov = self.cfg.use_nesterov

        Pshmp = self.Pshmp

        if mu > 0:
            M = self.M
            M.mul_(mu).add_(Pshmp)

            if use_nesterov:
                Pshmp.mul_(mu).add_(M)
                U = Pshmp
            else:
                U = M
            # end if-else use_nesterov
        else:
            U = Pshmp
        # end if-else mu

        # reconstruct parameter update from update U
        N = self.shape_raw[0]  # number of 1D params
        # E = self.shape_raw[1]  # embedding size

        g_view = self.shape_stacked.Gfull // N
        batch_size = g_view.b  # this will be 2

        i = 0 # indexes self.U
        for index, group, state, p in self.bucket_func():
            u = U[i : i + batch_size].view_as(p)
            i += batch_size
            p.add_(u, alpha=-lr)

    ############################################################
    ########## FUNCTIONS NOT RELATED TO SHAMPOO UPDATE
    ############################################################
    @torch.no_grad()
    def _wandbify(self, x: Tensor, op='norm'):
        match op:
            case 'norm':
                if x.ndim == 3:
                    out = x.norm(p=2, dim=(1, 2), keepdim=True).view(-1).cpu().numpy()
                else:
                    out = x.norm(p=2).view(-1).cpu().numpy()
            case 'rank':
                if x.ndim == 3:
                    out = torch.linalg.matrix_rank(x).view(-1).cpu().numpy()
                else:
                    out = torch.linalg.matrix_rank(x).cpu().numpy()
            case 'vector':
                out = x.cpu().numpy()
        return out

    @torch.no_grad()
    def log_stats(self, t):
        try:
            wandbify = self._wandbify
            name = self.name
            mu = self.cfg.mu

            G = self.G
            L = self.L
            invL = self.invL
            A = self.A
            Pgraft_fro = self.Pgraft_fro
            shapeG = tuple(G.shape)
            shapeL = tuple(L.shape)

            data = {  # add stats for FULL blocks
                'stats/t': t,
                f'stats/{name}_{shapeG}_norm_G': wandb.Histogram(wandbify(G, op='norm')),
                f'stats/{name}_{shapeL}_norm_L': wandb.Histogram(wandbify(L, op='norm')),
                f'stats/{name}_{shapeL}_norm_L_inv': wandb.Histogram(wandbify(invL, op='norm')),
                f'stats/{name}_{shapeG}_norm_A': wandb.Histogram(wandbify(A, op='norm')),
                f'stats/{name}_{shapeG}_norm_PgraftFro': wandb.Histogram(wandbify(Pgraft_fro, op='norm')),

                # f'stats/{name}_{shape}_rank_G_full': wandb.Histogram(wandbify(G.full, op='rank')),
                # f'stats/{name}_{shape}_rank_L_full': wandb.Histogram(wandbify(Lfull, op='rank')),
                # f'stats/{name}_{shape}_rank_R_full': wandb.Histogram(wandbify(Rfull, op='rank')),
            }
            if mu > 0:
                data[f'stats/{name}_{shapeG}_norm_M'] = wandb.Histogram(wandbify(self.M))

            wandb.log(data)
        except Exception as e:
            print('=' * 100)
            print(f'[ERROR][{self.name}][t={int(t)}]')
            print(f'Error: {str(e)}')
            traceback.print_exc()
            # breakpoint()
            print('=' * 100)