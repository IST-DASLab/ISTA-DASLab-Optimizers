import torch
from torch import Tensor
import math
import wandb
import traceback
from typing import Union

from .dash_layerwise_block_partitioner import *
from .dash_configs import *
from .dash_block_root_invertor import *

class DashLayerwiseProcessor:
    """
        Saves the states of Shampoo optimizer for a single layer.
        The states L, R, Linv4, Rinv4 and A (grafting state) are saved in a BlockPartitioner object because
        we want to compute statistics per block.
    """
    def __init__(self, param: Union[Tensor, DashFakeTensorWithGrad], cfg: DashConfig, name: str, is_norm_layer_stack: bool):
        self.param = param
        self.cfg = cfg
        self.name = name
        self.is_norm_layer_stack = is_norm_layer_stack

        # self.is_jorge = (cfg.inv_root_method == InverseRootMethodType.JORGE)

        self._initialize()

    @torch.no_grad()
    def _initialize(self):
        p = self.param # .p if self.is_norm_layer_stack else self.param
        ndim = p.ndim

        # if ndim not in [1, 2]:
        #     raise NotImplementedError(f'EfficientShampoo is implemented only for 1D and 2D tensors!')

        cfg = self.cfg
        algo1d = cfg.algo_one_dim
        dtype, device = p.dtype, p.device

        if not self.is_norm_layer_stack:
            if ndim == 1:
                if algo1d == DashAlgoOneDim.ADAMW:
                    print('Running adamw for 1D layers!')
                    self.m = torch.zeros_like(p)
                    self.v = torch.zeros_like(p)
                    return
                else:
                    print('Running Shampoo for 1D layers!')

        # if 2D or (1D and Shampoo)
        bp: DashLayerwiseBlockPartitioner = DashLayerwiseBlockPartitioner(param=self.param, B=cfg.block_size, is_norm_layer_stack=self.is_norm_layer_stack)
        self.block_partitioner = bp

        self.G: DashMatrixBlock = bp.get_regular_gradient_block()
        self.Pshmp: DashMatrixBlock = bp.get_regular_gradient_block()

        has_rest = self.G.has_rest

        # efficient grouping of L and R into a single MatrixBlock
        self.LR: DashMatrixBlock = bp.get_preconditioner_blocks_efficiently_grouped()
        self.barLR: DashMatrixBlock = bp.get_preconditioner_blocks_efficiently_grouped()

        # if self.is_jorge: # initialize barLR only for Jorge
        #     eps_pow = cfg.eps_inv_root ** (-0.25)
        #     self.barLR.full.diagonal(dim1=-2, dim2=-1).add_(eps_pow)
        #     if has_rest:
        #         self.barLR.rest.diagonal(dim1=-2, dim2=-1).add_(eps_pow)

        if cfg.grafting_type in [DashGraftingType.ADAGRAD, DashGraftingType.ADAM]:
            self.A: DashMatrixBlock = bp.get_regular_gradient_block() # this is A_t^(i) in Algorithm 2

            N_full = self.A.full.shape[0]

            if has_rest:
                N_rest = self.A.rest.shape[0]

            self.Pgraft_fro = DashMatrixBlock(
                shape_full=(N_full, 1, 1),
                shape_rest=(N_rest, 1, 1) if has_rest else None,
                info=DashBlockInfo.REGULAR_BLOCK,
                dtype=dtype,
                device=device)

        if cfg.beta_G > 0:
            self.tildeG: DashMatrixBlock = bp.get_regular_gradient_block() # this is tilde(G)_t^(i) in Algorithm 2

        if cfg.mu > 0:
            self.M = bp.get_regular_gradient_block() # this is M_t^(i) in Algorithm 2

    @torch.no_grad()
    def update_layer(self, t, lr):
        """
            Runs a step from Algorithm 2 in https://arxiv.org/pdf/2309.06497#page=12

            We run the following steps and skip the exponent "(i)":

            _update_factors: the paper has a typo: it uses beta2 in if, but updates L_t and R_t using beta1 (it should be beta2)
                We use betaLR instead of beta2 in our implementation to know exactly where beta is used.
                We perform these updates in grouped blocks (see BlockPartitioner class)
                Algorithm:
                    if betaLR < 1 then
                        L_t = betaLR L_t-1 + (1 - betaLR) G_t     @ G_t ^ T
                        R_t = betaLR R_t-1 + (1 - betaLR) G_t ^ T @ G_t
                    else
                        L_t = L_t-1 + G_t     @ G_t ^ T
                        R_t = R_t-1 + G_t ^ T @ G_t
                    end if

            _update_grafting:
                Algorithm:
                    A_t = UpdateGraftingState(A_t-1, G_t)
                    For Adam, we keep track of the second moment of the gradient using EMA
                    For AdaGrad, we keep track of the 2nd moment of the gradient using plain summation (no EMA)

            _invert_factors:
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

            _update_grad_ema:
                We use betaG instead of beta1 to avoid confusion.
                Algorithm:
                    if betaG > 0:
                        tilde(G)_t = betaG tilde(G)_t-1 + (1 - betaG) G
                    end if

            _compute_grafting_direction:
                This function computes the grafting direction
                P_t,graft = ComputeGraftingDirection(tilde(G)_t, t, use_bias_correction)

            _compute_shampoo_direction:
                This function computes the effective Shampoo direction.
                Algorithm:
                    if t >= start_preconditioning_step:
                        U_t,shmp = bar(L)_t @ tilde(G)_t @ bar(R)_t
                        P_t = (||P_t,graft|| / ||U_t,shmp||) * U_t,shmp

            _apply_momentum_and_nesterov:
                Apply momentum on P_t and nesterov if it is enabled.
                Algorithm:
                    if mu > 0:
                        M_t = mu * M_t + P_t
                        if use_nesterov:
                            P_t = mu * M_t + P_t
                        else:
                            P_t = M_t

            _update_weights:
                Update weights given the final update P_t
                Algorithm:
                    W_t+1 = W_t - lr * P_t
        """
        p = self.param # .p if self.is_norm_layer_stack else self.param
        ndim = p.ndim

        # if ndim not in [1, 2]: raise NotImplementedError(f'EfficientShampoo is implemented only for 1D and 2D tensors!')

        cfg = self.cfg
        algo1d = cfg.algo_one_dim

        if not self.is_norm_layer_stack:
            if ndim == 1 and algo1d == DashAlgoOneDim.ADAMW:
                self._adamw_step(t, lr)
                return

        self.block_partitioner.populate_gradient_block_partition(p.grad, self.G)

        self._update_factors()
        self._update_grafting()
        self._invert_factors(t)
        self._update_grad_ema()
        self._compute_grafting_direction(t)
        self._compute_shampoo_direction(t)
        self._apply_momentum_with_nesterov_then_update_weights(lr)

    @torch.no_grad()
    def _update_factors(self):
        """
        We use betaLR instead of beta2 in our implementation to know exactly where beta is used.
        We perform these updates in grouped blocks (see BlockPartitioner class).

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
        # is_jorge = self.is_jorge
        G = self.G
        Nfull = self.block_partitioner.num_blocks_full
        has_rest = G.has_rest
        LR = self.LR # efficient grouping for L and R
        barLR = self.barLR # also efficient grouping
        betaLR = self.cfg.beta_LR
        info = LR.info
        # is_2d = (self.param.ndim == 2) # for 1D: compute only L, for 2D: compute both L and R
        is_2d = not self.is_norm_layer_stack

        Gfull = G.full
        Gfull_T = Gfull.transpose(1, 2) # call contiguous for Dion kernel
        Lfull = torch.bmm(Gfull, Gfull_T) # G @ G.T # TODO: add Dion kernel here
        if is_2d:
            Rfull = torch.bmm(Gfull_T, Gfull) # G.T @ G# TODO: add Dion kernel here

        if has_rest:
            N_rest = self.block_partitioner.num_blocks_rest # if gradient has a rest block, then L and R will also have
            Grest = G.rest
            Grest_T = Grest.transpose(1, 2) # call contiguous for Dion kernel
            Lrest = torch.bmm(Grest, Grest_T) # G @ G.T # TODO: add Dion kernel here
            if is_2d:
                Rrest = torch.bmm(Grest_T, Grest) # G.T @ G # TODO: add Dion kernel here

        # unpacking by simple indexing: these two slices will always be the same size
        slice_Lfull = LR.full[0: Nfull]
        if is_2d:
            slice_Rfull = LR.full[Nfull: 2 * Nfull]
        # if is_jorge:
        #     slice_barLfull = barLR.full[0: Nfull]
        #     slice_barRfull = barLR.full[Nfull: 2 * Nfull]

        if info == DashBlockInfo.REGULAR_BLOCK:
            # do nothing
            # for 1D, self.LR contains only the L preconditioner, which is a regular block
            pass
        elif info == DashBlockInfo.EFFICIENT_BLOCK_GROUPING_FULL_LR_NO_REST:
            # do nothing because we have no rest
            # slice_Lrest = None
            # slice_Rrest = None
            pass
        elif info == DashBlockInfo.EFFICIENT_BLOCK_GROUPING_FULL_LR_REST_R_AND_REST_L:
            # full contains L_full, R_full, R_rest
            # rest contains L_rest
            slice_Rrest = LR.full[2 * Nfull:] # packed next to the full blocks
            if has_rest:
                slice_Lrest = LR.rest

            # if is_jorge:
            #     slice_barRrest = barLR.full[2 * Nfull:] # packed next to the full blocks
            #     if has_rest:
            #         slice_barLrest = barLR.rest
        elif info == DashBlockInfo.EFFICIENT_BLOCK_GROUPING_FULL_LR_REST_L_AND_REST_R:
            # full contains L_full, R_full, L_rest
            # rest contains R_rest
            if has_rest:
                slice_Rrest = LR.rest
            slice_Lrest = LR.full[2 * Nfull:]  # packed next to the full blocks

            # if is_jorge:
            #     if has_rest:
            #         slice_barRrest = barLR.rest
            #     slice_barLrest = barLR.full[2 * Nfull:]  # packed next to the full blocks

        if False: # self.is_jorge:
            pass # code for Jorge
            # pow2_slice_barLfull = bmm(slice_barLfull, slice_barLfull)
            # pow4_slice_barLfull = bmm(pow2_slice_barLfull, pow2_slice_barLfull)
            # slice_Lfull.copy_(bmm(pow4_slice_barLfull, Lfull))
            #
            # pow2_slice_barRfull = bmm(slice_barRfull, slice_barRfull)
            # pow4_slice_barRfull = bmm(pow2_slice_barRfull, pow2_slice_barRfull)
            # slice_Rfull.copy_(bmm(pow4_slice_barRfull, Rfull))
            #
            # del pow2_slice_barLfull, pow4_slice_barLfull, pow2_slice_barRfull, pow4_slice_barRfull
            #
            # if has_rest:
            #     pow2_slice_barLrest = bmm(slice_barLrest, slice_barLrest)
            #     pow4_slice_barLrest = bmm(pow2_slice_barLrest, pow2_slice_barLrest)
            #     slice_Lrest.copy_(bmm(pow4_slice_barLrest, Lrest))
            #
            #     pow2_slice_barRrest = bmm(slice_barRrest, slice_barRrest)
            #     pow4_slice_barRrest = bmm(pow2_slice_barRrest, pow2_slice_barRrest)
            #     slice_Rrest.copy_(bmm(pow4_slice_barRrest, Rrest))
            #
            #     del pow2_slice_barLrest, pow4_slice_barLrest, pow2_slice_barRrest, pow4_slice_barRrest
        else: # shampoo
            if betaLR < 1:
                one_minus_betaLR = 1 - betaLR

                slice_Lfull.lerp_(Lfull, weight=one_minus_betaLR)
                if has_rest:
                    slice_Lrest.lerp_(Lrest, weight=one_minus_betaLR)

                if is_2d:
                    slice_Rfull.lerp_(Rfull, weight=one_minus_betaLR)
                    if has_rest:
                        slice_Rrest.lerp_(Rrest, weight=one_minus_betaLR)
            else:
                slice_Lfull.add_(Lfull)
                if has_rest:
                    slice_Lrest.add_(Lrest)

                if is_2d:
                    slice_Rfull.add_(Rfull)
                    if has_rest:
                        slice_Rrest.add_(Rrest)

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
        beta_graft = self.cfg.beta_graft

        match grafting_type:
            case DashGraftingType.ADAM:
                A, G = self.A, self.G
                A.full.mul_(beta_graft).addcmul_(G.full, G.full, value=1-beta_graft)
                if G.has_rest:
                    A.rest.mul_(beta_graft).addcmul_(G.rest, G.rest, value=1-beta_graft)
            case _:
                raise NotImplementedError(f'Grafting type {grafting_type} is currently not supported, use Adam grafting for now!')

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
            ret = DashBlockRootInvertor.invert(Xin=self.LR, Xout=self.barLR, cfg=cfg, root=2 if self.is_norm_layer_stack else 4)
            # if ret is not None:
            #     name = self.name
            #     shape = tuple(self.param.shape)
            #     wandb.log({ # TODO: THIS AFFECTS RUNNING TIME OF OPTIMIZER STEP, FIND A WAY TO LOG THIS INFO IN log_stats
            #         't': t,
            #         f'stats/{name}_{shape}_ranks_eig_vals_full': wandb.Histogram(self._wandbify(ret, op='vector')),
            #         f'stats/{name}_{shape}_ranks_eig_vals_full_min': ret.min().item(),
            #         f'stats/{name}_{shape}_ranks_eig_vals_full_max': ret.max().item(),
            #     })

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
            G = self.G
            tildeG = self.tildeG
            one_minus_betaG = 1 - betaG

            tildeG.full.lerp_(G.full, weight=one_minus_betaG)
            if G.has_rest:
                tildeG.rest.lerp_(G.rest, weight=one_minus_betaG)

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
        betaG = cfg.beta_G
        beta_graft = cfg.beta_graft
        eps_graft = cfg.eps_grafting
        use_bias_correction = cfg.use_bias_correction
        G = self.G
        tildeG = self.tildeG
        start_prec_step = cfg.start_prec_step
        Pshmp = self.Pshmp

        if graft_type == DashGraftingType.ADAM:
            if betaG > 0: # we compute EMA for gradient and therefore use tilde(G)_t for grafting
                GorTildeG = self.tildeG
            else: # otherwise use plain gradient
                GorTildeG = self.G

            A = self.A
            Pgraft_fro = self.Pgraft_fro

            if betaG > 0: # this is redundant because we have GorTildeG
                if use_bias_correction:
                    biasG = 1 - betaG ** t
                    biasA = 1 - beta_graft ** t
                    Pgraft_full = (tildeG.full / biasG) / (eps_graft + (A.full / biasA).sqrt()) # new memory allocation
                else:
                    Pgraft_full = tildeG.full / (eps_graft + A.full.sqrt()) # new memory allocation
            else:
                if use_bias_correction: # apply bias correction only for A
                    biasA = 1 - beta_graft ** t
                    Pgraft_full = G.full / (eps_graft + (A.full / biasA).sqrt()) # new memory allocation
                else:
                    Pgraft_full = G.full / (eps_graft + A.full.sqrt()) # new memory allocation
            # end if-else
            Pgraft_fro.full.copy_(Pgraft_full.norm(p='fro', dim=(1, 2), keepdim=True))

            if t < start_prec_step: # anticipate the next step: compute shampoo direction here because we have access to Pgrafr_full/rest
                Pshmp.full.copy_(Pgraft_full)

            if A.has_rest:
                if betaG > 0:
                    if use_bias_correction:
                        Pgraft_rest = (tildeG.rest / biasG) / (eps_graft + (A.rest / biasA).sqrt()) # new memory allocation
                    else:
                        Pgraft_rest = tildeG.rest / (eps_graft + A.rest.sqrt()) # new memory allocation
                else:
                    if use_bias_correction: # apply bias correction only for A
                        Pgraft_rest = G.rest / (eps_graft + (A.rest / biasA).sqrt()) # new memory allocation
                    else:
                        Pgraft_rest = G.rest / (eps_graft + A.rest.sqrt()) # new memory allocation
                # end if-else
                Pgraft_fro.rest.copy_(Pgraft_rest.norm(p='fro', dim=(1, 2), keepdim=True))

                if t < start_prec_step:  # anticipate the next step: compute shampoo direction here because we have access to Pgrafr_full/rest
                    Pshmp.rest.copy_(Pgraft_rest)
        else:
            raise NotImplementedError(f'Grafting type {graft_type} is currently not supported, use Adam grafting for now!')

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
            Nfull = self.block_partitioner.num_blocks_full
            barLR = self.barLR
            Pshmp = self.Pshmp
            Pgraft_fro = self.Pgraft_fro
            betaG = self.cfg.beta_G
            # is_2d = (self.param.ndim == 2) # for 1D: compute only L, for 2D: compute both L and R
            is_2d = not self.is_norm_layer_stack

            if betaG > 0: # we compute EMA for gradient and therefore use tilde(G)_t for grafting
                GorTildeG = self.tildeG
            else: # otherwise use plain gradient
                GorTildeG = self.G

            Linv_full = barLR.full[    0 :     Nfull]
            if is_2d:
                Rinv_full = barLR.full[Nfull : 2 * Nfull]

            ### apply grafting per block
            if is_2d:
                Pshmp.full.copy_(Linv_full @ GorTildeG.full @ Rinv_full) # here is where we do shampoo update L^-1/4 @ G @ R^-1/4
            else:
                Pshmp.full.copy_(Linv_full @ GorTildeG.full)  # here is where we do shampoo update L^-1/2 @ G @
            scaling_full = Pgraft_fro.full / Pshmp.full.norm(p='fro', dim=(1, 2), keepdim=True)
            Pshmp.full.mul_(scaling_full)

            if GorTildeG.has_rest:
                info = barLR.info

                if info == DashBlockInfo.EFFICIENT_BLOCK_GROUPING_FULL_LR_REST_R_AND_REST_L:
                    # full contains L_full, R_full, R_rest
                    # rest contains L_rest
                    Rinv_rest = self.barLR.full[2 * Nfull:]  # packed next to the full blocks
                    Linv_rest = self.barLR.rest
                elif info == DashBlockInfo.EFFICIENT_BLOCK_GROUPING_FULL_LR_REST_L_AND_REST_R:
                    # full contains L_full, R_full, L_rest
                    # rest contains R_rest
                    Rinv_rest = self.barLR.rest
                    Linv_rest = self.barLR.full[2 * Nfull:]  # packed next to the full blocks

                ### apply grafting per block
                if is_2d:
                    Pshmp.rest.copy_(Linv_rest @ GorTildeG.rest @ Rinv_rest) # here is where we do shampoo update L^-1/4 @ G @ R^-1/4
                else:
                    Pshmp.rest.copy_(Linv_rest @ GorTildeG.rest)  # here is where we do shampoo update L^-1/2 @ G
                scaling_rest = Pgraft_fro.rest / Pshmp.rest.norm(p='fro', dim=(1, 2), keepdim=True)
                Pshmp.rest.mul_(scaling_rest)
        else:
            """
            Here we should use grafting direction.
            However, right here we do nothing because we do not have access to grafting directions.
            Instead, we treat this case in the function `_compute_grafting_direction` because there we explicitly compute the grafting directions.
            Therefore, check the statements `if t < start_prec_step` in function `_compute_grafting_direction`, where we copy the grafting direction to Pshmp.
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
        Pshmp = self.Pshmp
        use_nesterov = self.cfg.use_nesterov

        if mu > 0:
            M = self.M
            M.full.mul_(mu).add_(Pshmp.full)
            if M.has_rest:
                M.rest.mul_(mu).add_(Pshmp.rest)

            if use_nesterov:
                Pshmp.full.add_(M.full, alpha=mu)
                if M.has_rest:
                    Pshmp.rest.add_(M.rest, alpha=mu)

                U = Pshmp
            else:
                U = M
        else:
            U = Pshmp

        if self.is_norm_layer_stack:
            W = self.param.p
            G = self.param.grad
        else:
            W = self.param
            G = self.param.grad

        # update weights using update U
        bp = self.block_partitioner
        bp.reconstruct_from_blocks(block=U, out=G)
        W.add_(G, alpha=-lr) # weight decay is applied in optimizer step


    ############################################################
    ########## FUNCTIONS NOT RELATED TO SHAMPOO UPDATE
    ############################################################
    @torch.no_grad()
    def _adamw_step(self, t, lr):
        cfg = self.cfg
        p = self.param
        assert p.ndim == 1
        g = p.grad
        m, v = self.m, self.v

        beta1 = cfg.adamw_beta1
        beta2 = cfg.adamw_beta2
        eps = cfg.adamw_eps

        m.lerp_(g, weight=1 - beta1)
        v.lerp_(g.square(), weight=1 - beta2)

        m_unbiased = m / (1 - beta1 ** t)
        v_unbiased = v / (1 - beta2 ** t)

        u = m_unbiased / (eps + v_unbiased.sqrt())

        # weight decay is applied in optimizer step
        p.add_(u, alpha=-lr)

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
            p = self.param # .p if self.is_norm_layer_stack else self.param
            ndim = p.ndim
            name = self.name
            algo1d = self.cfg.algo_one_dim

            if ndim == 1 and algo1d == DashAlgoOneDim.ADAMW:
                shape = p.shape[0]
                wandb.log({
                    't': t,
                    f'stats/{name}_{shape}_norm_m': wandbify(self.m),
                    f'stats/{name}_{shape}_norm_v': wandbify(self.v),
                })
            else:
                is_2d = not self.is_norm_layer_stack
                shape = tuple(p.shape)

                Nfull = self.block_partitioner.num_blocks_full
                mu = self.cfg.mu

                LR = self.LR
                barLR = self.barLR
                A = self.A
                G = self.G
                Pgraft_fro = self.Pgraft_fro
                g = self.param.grad

                Lfull = LR.full[0:     Nfull]

                data = {  # add stats for FULL blocks
                    't': t,
                    f'stats/{name}_{shape}_norm_g': (wandbify(g, op='norm')),

                    f'stats/{name}_{shape}_norm_L_full': wandb.Histogram(wandbify(Lfull, op='norm')),

                    f'stats/{name}_{shape}_norm_Linv_full': wandb.Histogram(wandbify(barLR.full[0:     Nfull], op='norm')),

                    f'stats/{name}_{shape}_norm_G_full': wandb.Histogram(wandbify(G.full, op='norm')),
                    f'stats/{name}_{shape}_norm_A_full': wandb.Histogram(wandbify(A.full, op='norm')),
                    f'stats/{name}_{shape}_norm_PgraftFro_full': wandb.Histogram(wandbify(Pgraft_fro.full, op='norm')),

                    # f'stats/{name}_{shape}_rank_G_full': wandb.Histogram(wandbify(G.full, op='rank')),
                    # f'stats/{name}_{shape}_rank_L_full': wandb.Histogram(wandbify(Lfull, op='rank')),
                    # f'stats/{name}_{shape}_rank_R_full': wandb.Histogram(wandbify(Rfull, op='rank')),
                }

                if is_2d: # regular 2D parameter, that has R-preconditioner
                    Rfull = LR.full[Nfull: 2 * Nfull]
                    data[f'stats/{name}_{shape}_norm_R_full'] = wandb.Histogram(wandbify(Rfull, op='norm'))
                    data[f'stats/{name}_{shape}_norm_Rinv_full'] = wandb.Histogram(wandbify(barLR.full[Nfull: 2 * Nfull], op='norm'))

                if mu > 0:
                    data[f'stats/{name}_{shape}_norm_M_full'] = wandb.Histogram(wandbify(self.M.full))

                if G.has_rest:  # add stats for REST blocks
                    info = LR.info

                    if info == DashBlockInfo.EFFICIENT_BLOCK_GROUPING_FULL_LR_REST_R_AND_REST_L:
                        # full contains L_full, R_full, R_rest
                        # rest contains L_rest

                        if is_2d:
                            slice_R_rest = LR.full[2 * Nfull:]  # packed next to the full blocks
                        slice_L_rest = LR.rest

                        if is_2d:
                            slice_Rinv_rest = barLR.full[2 * Nfull:]  # packed next to the full blocks
                        slice_Linv_rest = barLR.rest
                    elif info == DashBlockInfo.EFFICIENT_BLOCK_GROUPING_FULL_LR_REST_L_AND_REST_R:
                        # full contains L_full, R_full, L_rest
                        # rest contains R_rest

                        if is_2d:
                            slice_R_rest = LR.rest
                        slice_L_rest = LR.full[2 * Nfull:]  # packed next to the full blocks

                        if is_2d:
                            slice_Rinv_rest = barLR.rest
                        slice_Linv_rest = barLR.full[2 * Nfull:]  # packed next to the full blocks

                    data[f'stats/{name}_{shape}_norm_L_rest'] = wandb.Histogram(wandbify(slice_L_rest))
                    data[f'stats/{name}_{shape}_norm_Linv_rest'] = wandb.Histogram(wandbify(slice_Linv_rest))
                    if is_2d:
                        data[f'stats/{name}_{shape}_norm_R_rest'] = wandb.Histogram(wandbify(slice_R_rest))
                        data[f'stats/{name}_{shape}_norm_Rinv_rest'] = wandb.Histogram(wandbify(slice_Rinv_rest))

                    data[f'stats/{name}_{shape}_norm_G_rest'] = wandb.Histogram(wandbify(G.rest))
                    data[f'stats/{name}_{shape}_norm_A_rest'] = wandb.Histogram(wandbify(A.rest))
                    data[f'stats/{name}_{shape}_norm_PgraftFro_rest'] = wandb.Histogram(wandbify(Pgraft_fro.rest))

                    if mu > 0:
                        data[f'stats/{name}_{shape}_norm_M_rest'] = wandb.Histogram(wandbify(self.M.rest))
                # end if G.has_rest
                wandb.log(data)
        except Exception as e:
            print('=' * 100)
            print(f'[ERROR][{self.name}][t={int(t)}][shape={tuple(p.shape)}]')
            print(f'Error: {str(e)}')
            traceback.print_exc()
            print('=' * 100)