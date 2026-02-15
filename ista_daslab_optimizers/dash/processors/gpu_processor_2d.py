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

class DashGpuProcessor2D:
    """
        Saves the states of Shampoo optimizer for a all 2D layers assigned to the current GPU.

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
        self.name = '2d'

        # store the shapes of G and LR (inclusively invLR) for all stacked layers on the current GPU (merged)
        self.shape_G: DashStackedBlocksHandler = None
        self.shape_LR: DashShape3D = None

        self.G = None
        self.LR = None # for 1D layers it is Tensor and for 2D layers it is DashStackedFactorsHandler
        self.invLR = None
        self.A = None # grafting buffer
        self.Pshmp = None # shampoo update
        self.Pgraft_fro = None # frobenius norm of grafting update (G or tildeG) / (eps + sqrt(A))
        self.tildeG = None # momentum for gradient
        self.M = None # momentum for shampoo update

        self._initialize()

    @torch.no_grad()
    def _initialize(self):
        cfg = self.cfg
        assert cfg.grafting_type == DashGraftingType.ADAM
        B = cfg.block_size

        p0 = next(self.bucket_func())[-1] # the generator bucket_func() yields (index, group, state, p)
        dtype = p0.dtype
        device = p0.device

        DSBH = partial(DashStackedBlocksHandler, block_size=B, dtype=dtype, device=device)
        DSBH_like = DashStackedBlocksHandler.like

        zeros = partial(torch.zeros, dtype=dtype, device=device, requires_grad=False)
        zeros_like = torch.zeros_like

        rank = dist.get_rank() if dist.is_initialized() else 0

        for index, group, state, p in self.bucket_func():
            state[STATE_DASH_SHAPE]: DashMultiShape = DashShapesCalculator.get_stacked_shapes_per_single_linear_layer(shape=p.shape, B=cfg.block_size)
        # end for

        self.shape_G, self.shape_LR = DashShapesCalculator.get_stacked_shape_for_all_linear_layers(self.bucket_func, B)
        print(f'[rank={rank}] shape_G: {self.shape_G}, shape_LR: {self.shape_LR}')

        # define matrices storing both full and rests in the handler
        self.G          = DSBH(self.shape_G)
        self.LR         = DSBH(self.shape_LR)
        self.invLR      = DSBH_like(self.LR)
        self.A          = DSBH_like(self.G)
        self.Pshmp      = DSBH_like(self.G)
        self.Pgraft_fro = {shape: zeros((block.shape[0], 1, 1)) for shape, block in self.G.iter_shapes_blocks()}
        if cfg.beta_G > 0: self.tildeG = DSBH_like(self.G)
        if cfg.mu > 0: self.M = DSBH_like(self.G)
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
        Copy the gradient from model parameters to our GPU-stacked buffer G.
        """
        cfg = self.cfg
        B = cfg.block_size

        G: DashStackedBlocksHandler = self.G
        G.reset_stacking_indices() # important when we update the stacked blocks

        for pi, group, state, p in self.bucket_func():
            g = p.grad # 2D gradient (capital G is reserved for the stacked blocks
            R, C = p.shape
            blockified_shape: DashMultiShape = state[STATE_DASH_SHAPE]  # the current layer has this blockified shape

            R_full = blockified_shape.stats.R_full
            C_full = blockified_shape.stats.C_full
            num_row_blocks = blockified_shape.stats.num_row_blocks
            num_col_blocks = blockified_shape.stats.num_col_blocks
            row_rest = blockified_shape.stats.row_rest
            col_rest = blockified_shape.stats.col_rest

            # if R_full > 0 and C_full > 0: # this is always True, it's commented because it is redundant
            X_full = g[:R_full, :C_full]
            view_shape = (num_row_blocks, B, num_col_blocks, B)
            blocks_full = X_full.view(view_shape).transpose(1, 2).reshape(-1, B, B)
            batch_size_full = blockified_shape.Gfull.b

            G.stack_block(block=blocks_full, action='copy')

            blocks_rest = None
            if col_rest > 0:
                right = g[:R_full, C_full:]  # shape (R_full, col_rest)
                view_shape = (num_row_blocks, B, col_rest)
                blocks_rest = right.view(view_shape)
            elif row_rest > 0:
                bottom = g[R_full:, :C_full]  # (row_rest, C_full)
                view_shape = (row_rest, num_col_blocks, B)
                blocks_rest = bottom.view(view_shape).transpose(0, 1)  # after ranspose it becomes (num_col_blocks, row_rest, B)
            # end if-elif
            G.stack_block(block=blocks_rest, action='copy')
        # end for loop

    @torch.no_grad()
    def _update_factors(self):
        """
        For Llama-953M, the first GPU has the following shapes:
            shape_G_2d: {
                (1024, 1024): DashShape3D(b=110, m=1024, n=1024),
                (256, 1024): DashShape3D(b=2, m=256, n=1024),
                (1024, 512): DashShape3D(b=4, m=1024, n=512),
                (512, 1024): DashShape3D(b=4, m=512, n=1024)
            }
            shape_LR_2d: {
                (1024, 1024): DashShape3D(b=230, m=1024, n=1024),
                (256, 256): DashShape3D(b=2, m=256, n=256),
                (512, 512): DashShape3D(b=8, m=512, n=512)
            }
        """
        cfg = self.cfg
        betaLR = cfg.beta_LR
        B = cfg.block_size

        G: DashStackedBlocksHandler = self.G
        LR: DashStackedBlocksHandler = self.LR
        LR.reset_stacking_indices() # important when we update the stacked blocks

        # ### Update L and R preconditioners using efficient triton kernel
        # for blockG in G.iter_blocks():
        #     LR.stack_grad_product(G=blockG, beta=betaLR)

        ## Update L and R preconditioners using bmm (with redundant matmuls)
        is_ema = (betaLR < 1)
        action    = 'lerp'       if is_ema else 'add'
        ema_decay = (1 - betaLR) if is_ema else None

        for blockG in G.iter_blocks():
            blockG_T = blockG.transpose(1, 2)
            LR.stack_block(block=bmm(blockG, blockG_T), action=action, lerp_weight=ema_decay)
            LR.stack_block(block=bmm(blockG_T, blockG), action=action, lerp_weight=ema_decay)

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

        beta_graft = self.cfg.beta_graft
        for blockA, blockG in zip(self.A.iter_blocks(), self.G.iter_blocks()):
            assert tuple(blockA.shape) == tuple(blockG.shape)
            blockA.mul_(beta_graft).addcmul_(blockG, blockG, value=1-beta_graft)

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
            for blockLR, blockInvLR in zip(self.LR.iter_blocks(), self.invLR.iter_blocks()):
                assert tuple(blockLR.shape) == tuple(blockInvLR.shape)
                DashRootInvertor.invert(Xin=blockLR, Xout=blockInvLR, cfg=cfg, root=4)

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
            for blockG, blockTildeG in zip(self.G.iter_blocks(), self.tildeG.iter_blocks()):
                assert tuple(blockG.shape) == tuple(blockTildeG.shape)
                blockTildeG.lerp_(blockG, weight=1-betaG)

    @torch.no_grad()
    def _compute_grafting_direction(self, t):
        """
        This function computes the frobenius norm of the grafting direction P_t,graft_Adam = ComputeGraftingDirection(tilde(G)_t, t, use_bias_correction)
        Algorithm for Adam grafting (page 9 in Distributed Shampoo paper):
            P_t,graft_Adam = (G_t OR tilde(G)_t) / (tilde(A)_t^1/2 + eps)
            tilde(G)_t = Adam's second moment of the gradient
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

        G: DashStackedBlocksHandler = self.G
        A: DashStackedBlocksHandler = self.A
        Pshmp: DashStackedBlocksHandler = self.Pshmp
        Pgraft_fro: dict = self.Pgraft_fro

        use_momentum_G = (betaG > 0)
        chosenG: DashStackedBlocksHandler = self.tildeG if use_momentum_G else self.G

        # do bias correction for G only when momentum for G is enabled
        biasG = (1 - betaG ** t) if use_bias_correction and use_momentum_G else 1
        biasA = (1 - beta_graft ** t) if use_bias_correction else 1

        for (shapeChosenG, blockChosenG), (shapeA, blockA) in zip(chosenG.iter_shapes_blocks(), A.iter_shapes_blocks()):
            assert shapeChosenG == shapeA # make sure we iterate through chosenG and A in the same order
            Pgraft = (blockChosenG / biasG) / (eps_graft + (blockA / biasA).sqrt())
            normPgraft = Pgraft.norm(p='fro', dim=(1, 2), keepdim=True)
            Pgraft_fro[shapeA].copy_(normPgraft)
            if t < start_prec_step: # anticipate the next step: compute shampoo direction here because we have access to Pgrafr_full/rest
                Pshmp.copy_block(Pgraft)

    @torch.no_grad()
    def _compute_shampoo_direction(self, t):
        """
        This function computes the effective Shampoo direction.
        Algorithm:
            if t >= start_preconditioning_step:
                U_t,shmp = bar(L)_t @ (G_t OR tilde(G)_t) @ bar(R)_t
                P_t = (||P_t,graft|| / ||U_t,shmp||) * U_t,shmp
        """
        start_prec_step = self.cfg.start_prec_step
        if t >= start_prec_step:
            betaG = self.cfg.beta_G
            use_momentum_G = (betaG > 0)

            invLR: DashStackedBlocksHandler = self.invLR
            Pshmp: DashStackedBlocksHandler = self.Pshmp
            Pgraft_fro = self.Pgraft_fro
            chosenG: DashStackedBlocksHandler = self.tildeG if use_momentum_G else self.G

            invLR.reset_unstacking_indices() # important when we update the UNstacked blocks

            for shapeG, blockChosenG in chosenG.iter_shapes_blocks():
                b = blockChosenG.shape[0]
                r, c = shapeG

                blockLinv = invLR.unstack_block(shape=(r,r), length=b)
                blockRinv = invLR.unstack_block(shape=(c,c), length=b)

                # compute the unscaled shampoo update L^-1/4 @ G @ R^-1/4
                Ushmp = blockLinv @ blockChosenG @ blockRinv # this is unscaled

                # compute the scaling
                scaling = Pgraft_fro[shapeG] / Ushmp.norm(p='fro', dim=(1, 2), keepdim=True)

                # rescale in-place
                Ushmp.mul_(scaling)

                # update the final direction that will be used for the model
                Pshmp.copy_block(Ushmp)
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
        cfg = self.cfg
        mu = cfg.mu
        use_nesterov = cfg.use_nesterov

        Pshmp: DashStackedBlocksHandler = self.Pshmp

        if mu > 0:
            M: DashStackedBlocksHandler = self.M

            for blockM, blockPshmp in zip(M.iter_blocks(), Pshmp.iter_blocks()):
                assert tuple(blockM.shape) == tuple(blockPshmp.shape)
                blockM.mul_(mu).add_(blockPshmp)
            # end for

            if use_nesterov:
                for blockM, blockPshmp in zip(M.iter_blocks(), Pshmp.iter_blocks()):
                    assert tuple(blockM.shape) == tuple(blockPshmp.shape)
                    blockPshmp.mul_(mu).add_(blockM)
                # end for
                U = Pshmp
            else:
                U = M
            # end if-else
        else:
            U = Pshmp
        # end if-else

        # update weights using update U
        U.reset_unstacking_indices() # important when we update the UNstacked blocks

        for index, group, state, p in self.bucket_func():
            multi_shape: DashMultiShape = state[STATE_DASH_SHAPE]

            Nfull, Rfull, Cfull = multi_shape.Gfull
            blockGfull = U.unstack_block(shape=(Rfull, Cfull), length=Nfull)

            if multi_shape.Grest is None:
                blockGrest = None
            else:
                Nrest, Rrest, Crest = multi_shape.Grest
                blockGrest = U.unstack_block(shape=(Rrest, Crest), length=Nrest)

            DashGpuPartitioner.reconstruct_from_blocks_for_2d(full_block=blockGfull,
                                                              rest_block=blockGrest,
                                                              block_size=cfg.block_size,
                                                              stats=multi_shape.stats,
                                                              out=p.grad) # save the reconstructed block in p.grad
            p.add_(p.grad, alpha=-lr) # weight decay is applied in optimizer step


    ############################################################
    ########## BELOW WE HAVE FUNCTIONS THAT LOG STATS TO WANDB
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

            LR: DashStackedBlocksHandler = self.LR
            invLR: DashStackedBlocksHandler = self.invLR
            A: DashStackedBlocksHandler = self.A
            G: DashStackedBlocksHandler = self.G
            Pgraft_fro: dict = self.Pgraft_fro

            data = { 'stats/t': t }

            for shapeG in G.iter_shapes(): # shapeG can be used for G, A, Pgraft_fro
                blockG = G._state[shapeG]
                blockA = A._state[shapeG]
                data[f'stats/{name}_{shapeG}_norm_G'] = wandb.Histogram(wandbify(blockG, op='norm'))
                data[f'stats/{name}_{shapeG}_norm_A'] = wandb.Histogram(wandbify(blockA, op='norm'))
                data[f'stats/{name}_{shapeG}_norm_PgraftFro'] = wandb.Histogram(wandbify(Pgraft_fro[shapeG], op='norm'))

                # f'stats/{name}_{shape}_rank_G_full': wandb.Histogram(wandbify(G.full, op='rank'))
                # f'stats/{name}_{shape}_rank_L_full': wandb.Histogram(wandbify(Lfull, op='rank'))
                # f'stats/{name}_{shape}_rank_R_full': wandb.Histogram(wandbify(Rfull, op='rank'))

                if mu > 0:
                    blockM = M._state[shapeG]
                    data[f'stats/{shapeG}_norm_M'] = wandb.Histogram(wandbify(blockM, op='norm'))
            # end for G

            # add stats for LR and invLR
            for (shapeLR, blockLR), (shapeLRinv, blockLRinv) in zip(LR.iter_shapes_blocks(), invLR.iter_shapes_blocks()):
                data[f'stats/{name}_{shapeLR}_norm_LR'] = wandb.Histogram(wandbify(blockLR, op='norm'))
                data[f'stats/{name}_{shapeLRinv}_norm_LR_inv'] = wandb.Histogram(wandbify(blockLRinv, op='norm'))
            # end for LR, invLR

            wandb.log(data)
        except Exception as e:
            print('=' * 100)
            print(f'[ERROR][{self.name}][t={int(t)}]')
            print(f'Error: {str(e)}')
            traceback.print_exc()
            # breakpoint()
            print('=' * 100)