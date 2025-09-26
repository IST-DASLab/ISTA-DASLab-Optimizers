import torch
import torch.distributed as dist
import math
import wandb

from ista_daslab_optimizers.utils.dct import dct3_matrix, dct_type2_makhoul
from ista_daslab_optimizers.utils.global_cache import GlobalCache
from ista_daslab_optimizers.utils.newton_schulz_triton import newton_schulz_triton

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)

def zeropower_via_newtonschulz5(G, steps: int):
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def trion_update(G, M, rank, ns_type, mu, use_makhoul, ns_steps, out_ortho, out_indices):
    # formerly called dct_dion_low_rank_muon_update
    M.add_(G)

    R, C = G.shape
    is_right_proj = (R >= C)
    # DCT = get_dct_matrix(size=min(R, C), key=(min(R, C), rank), device=G.device, dtype=G.dtype)

    size = min(R, C)
    key = (size, rank)
    if GlobalCache.contains(category='ortho', key=key):
        DCT = GlobalCache.get(category='ortho', key=key)
    else:
        DCT = dct3_matrix(min(R, C), device=device, dtype=dtype)
        GlobalCache.add(category='ortho', key=key, item=DCT.T if use_makhoul else DCT)

    if use_makhoul:
        if is_right_proj: # R >= C, I have to flip it for Makhoul
            inputM = M
        else:
            inputM = M.T
        # force the input to have more columns than rows for Makhoul
        # fatM = M if R <= C else M.T # input fat/wide matrix to Makhoul: R < C
        S = dct_type2_makhoul(inputM)

        if is_right_proj:
            norms = S.norm(p=1, dim=0)  # dim = 0 computes norm of columns (over all rows) to rank columns
        else:
            ### case 1: transpose S to be able to use dim=1
            S = S.T # account for the transposition in inputM because Makhoul computes DCT per rows by default
            norms = S.norm(p=1, dim=1)  # dim = 1 computes norm of rows (over all columns) to rank rows

            ### case 2: to avoid transposing S, use dim=0 instead of dim=1 and it should be the same
            # norms = S.norm(p=1, dim=0)  # dim = 0 computes norm of columns (over all rows) to rank columns
    else: # use matmul
        # ranking: compute similarities
        if is_right_proj:
            S = M @ DCT # (R, C) @ (C, C) = (R, C)
            norms = S.norm(p=1, dim=0)  # dim = 0 computes norm of columns (over all rows)
        else:
            S = DCT @ M # (R, R) @ (R, C) = (R, C)
            norms = S.norm(p=1, dim=1)

    # ranking: determine indices of most significant rows/columns
    indices = torch.topk(input=norms, k=rank, sorted=False).indices

    # create Q_r
    if is_right_proj:
        Q = DCT[:, indices] # (C, r)
        m = S[:, indices] # (R, r)
        M.add_(m @ Q.T, alpha=-(1 - mu))
    else:
        Q = DCT[indices, :] # (r, R)
        m = S[indices, :] # (r, C)
        M.add_(Q.T @ m, alpha=-(1 - mu))

    if ns_type == 'torch':
        ortho_m = zeropower_via_newtonschulz5(m, steps=ns_steps).to(dtype=M.dtype)
    elif ns_type == 'triton':
        ortho_m = newton_schulz_triton(m).to(dtype=M.dtype)
    else:
        raise RuntimeError(f'Unknown ns_type: {ns_type}')

    out_ortho.copy_(ortho_m)
    out_indices.copy_(indices)

class Trion(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:

                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["step"] = 0
                # assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                # assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_param_index = 0
        for group in self.param_groups:
            if group["use_muon"]:
                group["step"] += 1

                params = group["params"]

                if ('lowrank_updates' not in group) and ('lowrank_indices' not in group):
                    group["lowrank_updates"] = []
                    group["lowrank_indices"] = []
                    for p in params:
                        R, C = p.shape
                        is_right_proj = (R >= C)
                        o_shape = (R, group["rank"]) if is_right_proj else (group["rank"], C)
                        group["lowrank_updates"].append(torch.zeros(o_shape, dtype=p.dtype, device=p.device))
                        group["lowrank_indices"].append(torch.zeros(group["rank"], dtype=torch.int32, device=p.device))
                lowrank_updates = group["lowrank_updates"]
                lowrank_indices = group["lowrank_indices"]

                pad_size = dist.get_world_size() - len(params) % dist.get_world_size()
                # params_pad = params + [torch.empty_like(params[-1])] * pad_size
                lowrank_updates_pad = lowrank_updates + [torch.empty_like(lowrank_updates[-1])] * pad_size
                lowrank_indices_pad = lowrank_indices + [torch.empty_like(lowrank_indices[-1])] * pad_size

                ##### compute low-rank updates only on one GPU
                for pi in range(len(params))[::dist.get_world_size()]:
                    idx = pi + dist.get_rank()
                    if idx < len(params):
                        p = params[idx]
                        lowrank_u = lowrank_updates[idx] # low-rank update
                        lowrank_idx = lowrank_indices[idx] # row/column indices

                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization

                        state = self.state[p]

                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["param_id"] = muon_param_index
                            muon_param_index += 1

                        trion_update(
                            G=p.grad,
                            M=state["momentum_buffer"],
                            rank=group["rank"],
                            ns_type=group["ns_type"],
                            mu=group["momentum"],
                            use_makhoul=group.get("use_makhoul", False),
                            ns_steps=5,
                            out_ortho=lowrank_u,
                            out_indices=lowrank_idx)
                    # end if

                    # all-gather for low-rank updates
                    dist.all_gather(
                        tensor_list=lowrank_updates_pad[pi:pi + dist.get_world_size()],
                        tensor=lowrank_updates_pad[idx])

                    # all-gather for row/column indices
                    dist.all_gather(
                        tensor_list=lowrank_indices_pad[pi:pi + dist.get_world_size()],
                        tensor=lowrank_indices_pad[idx])
                # end for pi

                for pi in range(len(params)):
                    p = params[pi]
                    R, C = p.shape
                    indices = lowrank_indices[pi]
                    ot = lowrank_updates[pi]
                    DCT = get_dct_matrix(size=min(R, C), key=(min(R, C), group["rank"]), device=p.device, dtype=p.dtype)

                    is_right_proj = (R >= C)
                    # print(f'R: {R}, C:{C}, Q: {tuple(Q.shape)}, ot: {ot.shape}')
                    if is_right_proj:
                        Q = DCT[:, indices] # (C, r)
                        update = ot @ Q.T  # (R, r) @ (r, C) = (R, C)
                    else:
                        Q = DCT[indices, :]  # (R, r)
                        update = Q.T @ ot  # (R, r) @ (r, C) = (R, C)

                    # R, C = p.shape
                    scaling_type = group["scaling_type"]
                    if scaling_type == 'kj':
                        scaling = max(1, R / C) ** 0.5
                    elif scaling_type == 'none':
                        scaling = 1
                    elif scaling_type == 'kimi':
                        scaling = 0.2 * math.sqrt(max(R, C))
                    elif scaling_type == 'dion':
                        scaling = (R / C) ** 0.5

                    p.mul_(1 - group["lr"] * group["weight_decay"]).add_(update.reshape(p.shape), alpha=-group["lr"] * scaling)
                # end for pi
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad,state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
