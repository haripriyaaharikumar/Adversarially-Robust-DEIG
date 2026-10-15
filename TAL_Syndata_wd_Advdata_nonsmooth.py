from __future__ import annotations

import math
from typing import Tuple, Dict, Any, Optional, Sequence

import numpy as np
import torch
import gpytorch

import config1 as config  

def gen_decision_torch(
    Ntrain: int,
    Nquery: int,
    Ndec: int,
    Ntest: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    torch.manual_seed(config.seed)
    Ntot = Ntrain + Nquery + Ntest

    alpha = torch.ones((Ndec,), dtype=dtype, device=device)
    mix = torch.distributions.Dirichlet(alpha).sample()

    p = (0.65 / Ndec) * torch.ones((Ndec,), dtype=dtype, device=device) + 0.35 * mix
    p = p / p.sum()

    return torch.multinomial(p, num_samples=Ntot, replacement=True)


def _build_kernel_ard(
    dim: int,
    lengthscale_vec: torch.Tensor,     
    variance_scalar: torch.Tensor,     
    kernel_type: str = "matern",
    matern_nu: float = 1.5,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> gpytorch.kernels.Kernel:
    if kernel_type == "rbf":
        base = gpytorch.kernels.RBFKernel(ard_num_dims=dim).to(device=device, dtype=dtype)
    elif kernel_type == "matern":
        base = gpytorch.kernels.MaternKernel(nu=matern_nu, ard_num_dims=dim).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown kernel_type={kernel_type}. Use 'rbf' or 'matern'.")

    base.lengthscale = lengthscale_vec.view(1, 1, -1)
    kernel = gpytorch.kernels.ScaleKernel(base).to(device=device, dtype=dtype)
    kernel.outputscale = variance_scalar
    return kernel


def sample_gp_joint_at_X(
    X_joint: torch.Tensor,            
    lengthscale_vec: torch.Tensor,     
    variance_scalar: torch.Tensor,     
    jitter: float,
    generator: torch.Generator,
    kernel_type: str,
    matern_nu: float,
) -> torch.Tensor:
    dim = X_joint.shape[1]
    kernel = _build_kernel_ard(
        dim=dim,
        lengthscale_vec=lengthscale_vec,
        variance_scalar=variance_scalar,
        kernel_type=kernel_type,
        matern_nu=matern_nu,
        device=X_joint.device,
        dtype=X_joint.dtype,
    )

    K = kernel(X_joint).evaluate()
    K = K + jitter * torch.eye(X_joint.shape[0], dtype=X_joint.dtype, device=X_joint.device)
    L = torch.linalg.cholesky(K)
    z = torch.randn((X_joint.shape[0],), dtype=X_joint.dtype, device=X_joint.device, generator=generator)
    return L @ z


def make_adv_eval_points(
    x_test: torch.Tensor,                
    adv_dims: Optional[Sequence[int]] = None,
    eps_adv: float = 0.1,
    n_eval: int = 1,
    seed_split: int = 1,
) -> torch.Tensor:
    adv_dims = [int(d) for d in (adv_dims or [])]
    dim = x_test.shape[1]
    for d in adv_dims:
        if d < 0 or d >= dim:
            raise ValueError(f"adv_dims contains invalid dim {d} for input dim={dim}")

    if len(adv_dims) == 0:
        return x_test.repeat_interleave(n_eval, dim=0)

    g = torch.Generator(device=x_test.device).manual_seed(seed_split)

    deltas = (2.0 * torch.rand(
        (x_test.shape[0], n_eval, len(adv_dims)),
        dtype=x_test.dtype, device=x_test.device, generator=g
    ) - 1.0) * eps_adv

    X_rep = x_test[:, None, :].repeat(1, n_eval, 1).contiguous()
    for j, d in enumerate(adv_dims):
        X_rep[:, :, d] = X_rep[:, :, d] + deltas[:, :, j]

    return X_rep.view(-1, dim)


def generation_gpytorch_like_gpy(
    dim: int,
    std: float,
    Ndec: int,
    Ntrain: int,
    Nquery: int,
    Ntest: int,
    seed_split: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    add_adv_eval: bool = True,
    jitter: float = 1e-6,
    n_adv_eval: int = 10,
    n_grid: int = 400,
    grid_range: tuple[float, float] = (-2.0, 2.0),
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
 

    torch.manual_seed(config.seed)
    gen_data = torch.Generator(device=device).manual_seed(config.seed)

    Ntot = Ntrain + Nquery + Ntest

    X = torch.randn((Ntot, dim), device=device, dtype=dtype, generator=gen_data)

    D = gen_decision_torch(Ntrain=Ntrain, Nquery=Nquery, Ndec=Ndec, Ntest=Ntest, device=device, dtype=dtype)

    l = math.sqrt(dim) * (0.25 + 0.75 * torch.rand((Ndec, dim), generator=gen_data, device=device, dtype=dtype))
    v = 0.5 + 2.0 * torch.rand((Ndec,), generator=gen_data, device=device, dtype=dtype)

    gen_split = torch.Generator(device=device).manual_seed(seed_split)

    idx_tot = torch.arange(Ntot, device=device)
    if Ntrain < Ndec:
        raise ValueError(f"Ntrain ({Ntrain}) must be >= Ndec ({Ndec}) to pick one per decision.")

    idx_tr = torch.empty((Ntrain,), dtype=torch.long, device=device)
    for i in range(Ndec):
        idx_d = torch.nonzero(D == i, as_tuple=False).squeeze(-1)
        if idx_d.numel() == 0:
            raise RuntimeError(f"Decision {i} has zero samples. Increase Ntot or adjust decision sampling.")
        pick = idx_d[torch.randint(low=0, high=idx_d.numel(), size=(1,), generator=gen_split)]
        idx_tr[i] = pick

    idx_first = torch.unique(idx_tr[:Ndec])
    if idx_first.numel() != Ndec:
        used = set()
        for i in range(Ndec):
            idx_d = torch.nonzero(D == i, as_tuple=False).squeeze(-1)
            while True:
                pick = int(idx_d[torch.randint(0, idx_d.numel(), (1,), generator=gen_split)].item())
                if pick not in used:
                    used.add(pick)
                    idx_tr[i] = pick
                    break
        idx_first = idx_tr[:Ndec].clone()

    mask_keep = torch.ones((Ntot,), dtype=torch.bool, device=device)
    mask_keep[idx_first] = False
    idx_rem = idx_tot[mask_keep]

    n_extra = Ntrain - Ndec
    perm = torch.randperm(idx_rem.numel(), generator=gen_split, device=device)
    idx_extra = idx_rem[perm[:n_extra]]
    idx_tr[Ndec:] = idx_extra

    mask_keep2 = torch.ones((idx_rem.numel(),), dtype=torch.bool, device=device)
    mask_keep2[perm[:n_extra]] = False
    idx_left = idx_rem[mask_keep2]

    perm2 = torch.randperm(idx_left.numel(), generator=gen_split, device=device)
    idx_left = idx_left[perm2]

    idx_qu = idx_left[:Nquery]
    idx_te = idx_left[Nquery:Nquery + Ntest]  

    adv_dims = config.adv_dims
    eps_adv = float(getattr(config, "eps_adv", 0.1))

    if add_adv_eval:
        x_test_for_adv = X[idx_te, :]
        X_adv_eval = make_adv_eval_points(
            x_test_for_adv,
            adv_dims=adv_dims,
            eps_adv=eps_adv,
            n_eval=config.Nadv_test,
            seed_split=seed_split,
        )
    else:
        X_adv_eval = torch.empty((0, dim), dtype=dtype, device=device)

    if dim == 1:
        X_grid = torch.linspace(grid_range[0], grid_range[1], n_grid, dtype=dtype, device=device).view(-1, 1)
    else:
        X_grid = torch.empty((0, dim), dtype=dtype, device=device)

    X_all = torch.cat([X, X_adv_eval], dim=0)
    X_joint = torch.cat([X_all, X_grid], dim=0)
    N_all = X_all.shape[0]

    Y_all = torch.zeros((N_all, Ndec), dtype=dtype, device=device)
    F_grid = torch.zeros((X_grid.shape[0], Ndec), dtype=dtype, device=device) if dim == 1 else None

    smooth_arm = 0
    rough_arms = [i for i in range(Ndec) if i != smooth_arm]

    nu_smooth = 2.5
    nu_rough = 0.5

  
    # l_smooth = 2.0 * eps_adv
    # l_rough = 0.6 * eps_adv #0.5*eps_adv

    l_smooth = 2.0 * 0.3 
    l_rough = 0.6 * 0.3 

    v[smooth_arm] *= 1.0
    for i in rough_arms:
        v[i] *= 1.2

    adv_dims_int = [int(d) for d in (adv_dims or [])]
    if len(adv_dims_int) == 0:
        adv_dims_int = list(range(dim))

    for d in adv_dims_int:
        l[smooth_arm, d] = l_smooth
    for i in rough_arms:
        for d in adv_dims_int:
            l[i, d] = l_rough

    for i in range(Ndec):
        kernel_type = "matern"
        nu_i = nu_smooth if i == smooth_arm else nu_rough

        f_joint = sample_gp_joint_at_X(
            X_joint=X_joint,
            lengthscale_vec=l[i, :],
            variance_scalar=v[i],
            jitter=jitter,
            generator=gen_data,
            kernel_type=kernel_type,
            matern_nu=nu_i,
        )

        f_all_i = f_joint[:N_all]
        f_grid_i = f_joint[N_all:]  

        f_all_i_noisy = f_all_i + std * torch.randn((N_all,), dtype=dtype, device=device, generator=gen_data)

        Y_all[:, i] = f_all_i_noisy
        if dim == 1:
            F_grid[:, i] = f_grid_i  

    train_x = X[idx_tr, :]
    train_d = D[idx_tr]
    train_y = Y_all[idx_tr, train_d].reshape(Ntrain, 1)

    query_x = X[idx_qu, :]
    query_d = D[idx_qu]
    query_y = Y_all[idx_qu, query_d].reshape(Nquery, 1)

    test_x = X[idx_te, :]
    test_Y = Y_all[idx_te, :]

    adv_x = X_adv_eval
    adv_Y = Y_all[Ntot:Ntot + X_adv_eval.shape[0], :]

    train = {"x": train_x.detach().cpu().numpy(),
             "y": train_y.detach().cpu().numpy(),
             "d": train_d.detach().cpu().numpy()}

    query = {"x": query_x.detach().cpu().numpy(),
             "y": query_y.detach().cpu().numpy(),
             "d": query_d.detach().cpu().numpy()}

    test = {"x": test_x.detach().cpu().numpy(),
            "Y": test_Y.detach().cpu().numpy()}

    adv_test = {"x": adv_x.detach().cpu().numpy(),
                "Y": adv_Y.detach().cpu().numpy()}

    meta: Dict[str, Any] = {
        "X": X.detach().cpu().numpy(),
        "D": D.detach().cpu().numpy(),
        "Y": Y_all[:Ntot].detach().cpu().numpy(),
        "l": l.detach().cpu().numpy(),
        "v": v.detach().cpu().numpy(),
        "idx_tr": idx_tr.detach().cpu().numpy(),
        "idx_qu": idx_qu.detach().cpu().numpy(),
        "idx_te": idx_te.detach().cpu().numpy(),
        "seed_split": seed_split,
        "std": std,
        "jitter": jitter,
        "smooth_arm": int(smooth_arm),
        "rough_arms": [int(a) for a in rough_arms],
        "adv_dims": [int(d) for d in (adv_dims or [])],
        "eps_adv": eps_adv,
        "n_eval": int(config.Nadv_test),
        "nu_smooth": float(nu_smooth),
        "nu_rough": float(nu_rough),
        "l_smooth": float(l_smooth),
        "l_rough": float(l_rough),
    }

    if dim == 1:
        meta["X_grid"] = X_grid.detach().cpu().numpy()
        meta["F_grid"] = F_grid.detach().cpu().numpy()

    return train, query, test, adv_test, meta

