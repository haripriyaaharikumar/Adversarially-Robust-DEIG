from __future__ import annotations

import math
from typing import Tuple, Dict, Any, Optional, Sequence

import numpy as np
import torch
import gpytorch

import config1 as config  # expects: seed, seed_split, Ndec, adv_dims, eps_adv


# -----------------------------
# Decision sampling
# -----------------------------
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

    # 65% uniform + 35% random mixture
    p = (0.65 / Ndec) * torch.ones((Ndec,), dtype=dtype, device=device) + 0.35 * mix
    p = p / p.sum()

    return torch.multinomial(p, num_samples=Ntot, replacement=True)


# -----------------------------
# GP sampler (ARD + choice of kernel)
# -----------------------------
def _build_kernel_ard(
    dim: int,
    lengthscale_vec: torch.Tensor,     # (dim,)
    variance_scalar: torch.Tensor,     # scalar
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
    X_joint: torch.Tensor,             # (N_joint, dim)
    lengthscale_vec: torch.Tensor,     # (dim,)
    variance_scalar: torch.Tensor,     # scalar
    jitter: float,
    generator: torch.Generator,
    kernel_type: str,
    matern_nu: float,
) -> torch.Tensor:
    """Sample one GP function draw jointly at X_joint."""
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


# -----------------------------
# Adversarial eval set builder
# -----------------------------
def make_adv_eval_points(
    x_test: torch.Tensor,                 # (Ntest, dim)
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


# -----------------------------
# Main generator
# -----------------------------
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
    # plotting grid (only used when dim==1)
    n_grid: int = 400,
    grid_range: tuple[float, float] = (-2.0, 2.0),
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Returns (train, query, test, adv_test, meta)

    meta additionally includes (when dim==1):
      meta['X_grid']   shape (n_grid, 1)
      meta['F_grid']   shape (n_grid, Ndec)  # latent GP draw on grid (no obs noise)
    """

    torch.manual_seed(config.seed)
    gen_data = torch.Generator(device=device).manual_seed(config.seed)

    Ntot = Ntrain + Nquery + Ntest

    # Covariates
    X = torch.randn((Ntot, dim), device=device, dtype=dtype, generator=gen_data)

    # Decisions
    D = gen_decision_torch(Ntrain=Ntrain, Nquery=Nquery, Ndec=Ndec, Ntest=Ntest, device=device, dtype=dtype)

    # Kernel hyperparams per arm (initialize, then override attacked dims below)
    l = math.sqrt(dim) * (0.25 + 0.75 * torch.rand((Ndec, dim), generator=gen_data, device=device, dtype=dtype))
    v = 0.5 + 2.0 * torch.rand((Ndec,), generator=gen_data, device=device, dtype=dtype)

    # Split RNG
    gen_split = torch.Generator(device=device).manual_seed(seed_split)

    idx_tot = torch.arange(Ntot, device=device)
    if Ntrain < Ndec:
        raise ValueError(f"Ntrain ({Ntrain}) must be >= Ndec ({Ndec}) to pick one per decision.")

    # pick one per decision
    idx_tr = torch.empty((Ntrain,), dtype=torch.long, device=device)
    for i in range(Ndec):
        idx_d = torch.nonzero(D == i, as_tuple=False).squeeze(-1)
        if idx_d.numel() == 0:
            raise RuntimeError(f"Decision {i} has zero samples. Increase Ntot or adjust decision sampling.")
        pick = idx_d[torch.randint(low=0, high=idx_d.numel(), size=(1,), generator=gen_split)]
        idx_tr[i] = pick

    # ensure unique for first Ndec
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

    # remainder -> query/test
    mask_keep2 = torch.ones((idx_rem.numel(),), dtype=torch.bool, device=device)
    mask_keep2[perm[:n_extra]] = False
    idx_left = idx_rem[mask_keep2]

    perm2 = torch.randperm(idx_left.numel(), generator=gen_split, device=device)
    idx_left = idx_left[perm2]

    idx_qu = idx_left[:Nquery]
    idx_te = idx_left[Nquery:Nquery + Ntest]  # length Ntest (explicit)

    # adversarial eval points
    adv_dims = config.adv_dims
    eps_adv = float(getattr(config, "eps_adv", 0.1)) #get value from the config file

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

    # plotting grid (dim==1 only)
    if dim == 1:
        X_grid = torch.linspace(grid_range[0], grid_range[1], n_grid, dtype=dtype, device=device).view(-1, 1)
    else:
        X_grid = torch.empty((0, dim), dtype=dtype, device=device)

    # Build joint inputs for consistent GP draw on (data + adv + grid)
    X_all = torch.cat([X, X_adv_eval], dim=0)
    X_joint = torch.cat([X_all, X_grid], dim=0)
    N_all = X_all.shape[0]

    # Outputs
    Y_all = torch.zeros((N_all, Ndec), dtype=dtype, device=device)
    F_grid = torch.zeros((X_grid.shape[0], Ndec), dtype=dtype, device=device) if dim == 1 else None

    # -----------------------------
    # Robustness geometry (KEY)
    # -----------------------------
    smooth_arm = 0
    rough_arms = [i for i in range(Ndec) if i != smooth_arm]

    # Kernel smoothness:
    nu_smooth = 2.5
    nu_rough = 0.5

    # LENGTHSCALES TIED TO EPS (KEY):
    # smooth: changes gently across [-eps, eps]
    # rough : can vary a lot inside [-eps, eps]
    # l_smooth = 2.0 * eps_adv
    # l_rough = 0.6 * eps_adv #0.5*eps_adv

    l_smooth = 2.0 * 0.3 ##  curves have fixed shape
    l_rough = 0.6 * 0.3 

    # Amplitude: make rough arms slightly "tempting" (higher amplitude), smooth arm stable but not dominating
    v[smooth_arm] *= 1.0
    for i in rough_arms:
        v[i] *= 1.2

    adv_dims_int = [int(d) for d in (adv_dims or [])]
    if len(adv_dims_int) == 0:
        adv_dims_int = list(range(dim))

    # Set attacked-dim lengthscales explicitly
    for d in adv_dims_int:
        l[smooth_arm, d] = l_smooth
    for i in rough_arms:
        for d in adv_dims_int:
            l[i, d] = l_rough

    # For non-attacked dims, keep the random initial l[i, :]
    # (this makes the effect "directional": only adversarial dims induce instability)

    # -----------------------------
    # Sample per-arm jointly (data+adv+grid)
    # -----------------------------
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
        f_grid_i = f_joint[N_all:]  # (n_grid,) if dim==1 else empty

        # add observation noise to data+adv only
        f_all_i_noisy = f_all_i + std * torch.randn((N_all,), dtype=dtype, device=device, generator=gen_data)

        Y_all[:, i] = f_all_i_noisy
        if dim == 1:
            F_grid[:, i] = f_grid_i  # latent (no noise) for clean curves

    # build dicts
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


# -----------------------------
# One-figure plot: curves + points
# -----------------------------
# def plot_everything_1d(train: dict, query: dict, test: dict, meta: dict,
#                        filename: str | None = None,
#                        show_eps: bool = True):
#     X_grid = meta.get("X_grid", None)
#     F_grid = meta.get("F_grid", None)
#     if X_grid is None or F_grid is None:
#         raise ValueError("meta must contain 'X_grid' and 'F_grid' (run generation with dim=1).")

#     Xg = np.asarray(X_grid).reshape(-1)
#     Fg = np.asarray(F_grid)
#     Ndec = Fg.shape[1]
#     colors = plt.cm.tab10.colors

#     Xtr = np.asarray(train["x"]).reshape(-1)
#     ytr = np.asarray(train["y"]).reshape(-1)
#     Dtr = np.asarray(train["d"]).astype(int).reshape(-1)

#     Xqu = np.asarray(query["x"]).reshape(-1)
#     yqu = np.asarray(query["y"]).reshape(-1)
#     Dqu = np.asarray(query["d"]).astype(int).reshape(-1)

#     Xte = np.asarray(test["x"]).reshape(-1)
#     Yte = np.asarray(test["Y"])

#     plt.figure(figsize=(11, 4.8))

#     # curves
#     for k in range(Ndec):
#         plt.plot(Xg, Fg[:, k], color=colors[k % 10], label=f"GP curve (arm {k})")

#     # epsilon region around first test point (for display)
#     if show_eps and len(Xte) > 0:
#         eps = float(meta.get("eps_adv", 0.0))
#         x0 = float(Xte[0])
#         if eps > 0:
#             plt.axvline(x0 - eps, linestyle="--", linewidth=1)
#             plt.axvline(x0 + eps, linestyle="--", linewidth=1)

#     # train points
#     for k in range(Ndec):
#         m = (Dtr == k)
#         if np.any(m):
#             plt.scatter(Xtr[m], ytr[m], color=colors[k % 10], s=28, alpha=0.85,
#                         marker="o", edgecolors="k", linewidths=0.25, label=f"train (d={k})")

#     # query points
#     for k in range(Ndec):
#         m = (Dqu == k)
#         if np.any(m):
#             plt.scatter(Xqu[m], yqu[m], facecolors="none", edgecolors=colors[k % 10],
#                         s=55, alpha=0.9, marker="^", linewidths=1.2, label=f"query (d={k})")

#     # test outcomes
#     for t in range(len(Xte)):
#         for k in range(Ndec):
#             plt.scatter([Xte[t]], [Yte[t, k]], color=colors[k % 10], s=140, marker="X",
#                         edgecolors="k", linewidths=0.5,
#                         label=("test Y (per-arm)" if (t == 0 and k == 0) else None))

#     plt.xlabel("x (1D)")
#     plt.ylabel("reward")
#     plt.title("GP reward curves + train/query/test points (one figure)")
#     plt.grid(alpha=0.25)

#     # de-duplicate legend
#     handles, labels = plt.gca().get_legend_handles_labels()
#     uniq = {}
#     for h, l in zip(handles, labels):
#         if l and l not in uniq:
#             uniq[l] = h
#     plt.legend(uniq.values(), uniq.keys(), ncol=3, frameon=True, fontsize=9)

#     plt.tight_layout()
#     if filename is not None:
#         plt.savefig(filename, dpi=220)
#     plt.show()


# if __name__ == "__main__":
#     train, query, test, adv, meta = generation_gpytorch_like_gpy(
#         dim=1,
#         std=0.05,
#         Ndec=config.Ndec,
#         Ntrain=200,
#         Nquery=50,
#         Ntest=1,
#         seed_split=config.seed_split,
#         device="cpu",
#         dtype=torch.float64,
#         add_adv_eval=True,
#         jitter=1e-6,
#         n_adv_eval=10,
#         n_grid=500,
#         grid_range=(-2.0, 2.0),
#     )

    # plot_everything_1d(
    #     train, query, test, meta,
    #     filename=f"Nonsmooth_Syndata_all_{config.Ndec}.png",
    #     show_eps=True,
    # )
