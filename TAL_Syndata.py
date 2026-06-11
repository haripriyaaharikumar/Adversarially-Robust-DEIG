import numpy as np
import torch
import gpytorch
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import math
import config
# -----------------------------
# GPytorch-based GP sampling 
# -----------------------------
def sample_gp_at_X_gpytorch(
    X: torch.Tensor,
    lengthscale: float,
    variance: float,
    jitter: float = 1e-6,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    base = gpytorch.kernels.RBFKernel()
    base.lengthscale = torch.tensor(lengthscale, dtype=X.dtype, device=X.device)

    kernel = gpytorch.kernels.ScaleKernel(base)
    kernel.outputscale = torch.tensor(variance, dtype=X.dtype, device=X.device)

    K = kernel(X).evaluate()
    K = K + jitter * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)

    L = torch.linalg.cholesky(K)
    z = torch.randn((X.shape[0],), dtype=X.dtype, device=X.device, generator=generator)
    return L @ z


def gen_decision_torch(
    Ntrain: int,
    Nquery: int,
    Ndec: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:

    torch.manual_seed(config.seed)

    Ntot = Ntrain + Nquery + 1

    alpha = torch.ones((Ndec,), dtype=dtype, device=device)
    dirich = torch.distributions.Dirichlet(alpha)
    mix = dirich.sample()  # (Ndec,)  <- no generator kwarg

    p = (0.65 / Ndec) * torch.ones((Ndec,), dtype=dtype, device=device) + 0.35 * mix
    p = p / p.sum()

    # Sample Ntot decisions with replacement
    D = torch.multinomial(p, num_samples=Ntot, replacement=True)  # (Ntot,)
    return D


def generation_gpytorch_like_gpy(
    dim: int,
    std: float,
    Ndec: int,
    Ntrain: int,
    Nquery: int,
    Ntest: int,
    seed_split: int = 1,     # corresponds to seed_split in the GPy code
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    jitter: float = 1e-6,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:

    # --- Seeds / generators (separate seeds for data vs split, like your code)
    torch.manual_seed(config.seed)
    gen_data = torch.Generator(device=device)
    gen_data.manual_seed(config.seed)

    Ntot = Ntrain + Nquery + Ntest #Ntest is 1

    # --- Covariates
    X = torch.randn((Ntot, dim), device=device, dtype=dtype, generator=gen_data)

    # --- Decision assignments with same distributional construction
    D = gen_decision_torch(Ntrain, Nquery, Ndec, device=device, dtype=dtype)

    # --- Sample GP outcomes per decision at all X
    Y = torch.zeros((Ntot, Ndec), dtype=dtype, device=device)

    # ARD lengthscales and variances to match GPy code
    # l: (Ndec, dim), v: (Ndec,)
    l = math.sqrt(dim) * (0.25 + 0.75 * torch.rand((Ndec, dim), generator=gen_data, device=device, dtype=dtype))
    v = 0.5 + 2.0 * torch.rand((Ndec,), generator=gen_data, device=device, dtype=dtype)

    # IMPORTANT: your sampler currently accepts scalar lengthscale/variance.
    # To match ARD, we implement an ARD-aware sampler here.
    def sample_gp_at_X_gpytorch_ard(
        X_: torch.Tensor,
        lengthscale_vec: torch.Tensor,  # (dim,)
        variance_scalar: torch.Tensor,  # scalar
        jitter_: float,
        generator_: torch.Generator,
    ) -> torch.Tensor:
        base = gpytorch.kernels.RBFKernel(ard_num_dims=X_.shape[1]).to(device=X_.device, dtype=X_.dtype)
        base.lengthscale = lengthscale_vec.view(1, 1, -1)  # gpytorch expects (1,1,d) for ARD

        kernel = gpytorch.kernels.ScaleKernel(base).to(device=X_.device, dtype=X_.dtype)
        kernel.outputscale = variance_scalar

        K_ = kernel(X_).evaluate()
        K_ = K_ + jitter_ * torch.eye(X_.shape[0], dtype=X_.dtype, device=X_.device)

        L_ = torch.linalg.cholesky(K_)
        z_ = torch.randn((X_.shape[0],), dtype=X_.dtype, device=X_.device, generator=generator_)
        return L_ @ z_

    for i in range(Ndec):
        f_i = sample_gp_at_X_gpytorch_ard(
            X_=X,
            lengthscale_vec=l[i, :],
            variance_scalar=v[i],
            jitter_=jitter,
            generator_=gen_data,
        )
        # add observation noise like: + std*np.random.randn(Ntot)
        f_i = f_i + std * torch.randn((Ntot,), dtype=dtype, device=device, generator=gen_data)
        Y[:, i] = f_i

    # --- Split (matches your numpy logic)
    torch.manual_seed(seed_split)
    gen_split = torch.Generator(device=device)
    gen_split.manual_seed(seed_split)

    idx_tot = torch.arange(Ntot, device=device)

    # pick one training index per decision (without replacement within each decision group)
    idx_tr = torch.empty((Ntrain,), dtype=torch.long, device=device)

    for i in range(Ndec):
        idx_d = torch.nonzero(D == i, as_tuple=False).squeeze(-1)
        # In rare cases a decision may not appear in D; guard by resampling until it does.
        # (GPy code assumes there is at least one.)
        if idx_d.numel() == 0:
            raise RuntimeError(
                f"Decision {i} has zero samples in D. Increase Ntot or adjust decision sampling."
            )
        pick = idx_d[torch.randint(low=0, high=idx_d.numel(), size=(1,), generator=gen_split)]
        idx_tr[i] = pick

    # Remove those Ndec indices from idx_tot
    # (need to remove unique indices; enforce uniqueness in idx_tr[:Ndec] just in case)
    idx_first = torch.unique(idx_tr[:Ndec])
    # if duplicates happened (unlikely), fill them by re-drawing for that decision
    if idx_first.numel() != Ndec:
        # brute-force fix: re-draw until unique (small Ndec)
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
    idx_rem = idx_tot[mask_keep]  # remaining indices after mandatory per-decision picks

    # Choose remaining training indices uniformly from the remainder
    n_extra = Ntrain - Ndec
    perm = torch.randperm(idx_rem.numel(), generator=gen_split, device=device)
    idx_extra = idx_rem[perm[:n_extra]]
    idx_tr[Ndec:] = idx_extra

    # Remove extra training indices from remainder
    mask_keep2 = torch.ones((idx_rem.numel(),), dtype=torch.bool, device=device)
    mask_keep2[perm[:n_extra]] = False
    idx_left = idx_rem[mask_keep2]

    # Permute remaining, then split into query/test
    perm2 = torch.randperm(idx_left.numel(), generator=gen_split, device=device)
    idx_left = idx_left[perm2]

    idx_qu = idx_left[:Nquery]
    idx_te = idx_left[Nquery:]  # should be length 1

    # --- Build dicts exactly like GPy code
    train_x = X[idx_tr, :]
    train_d = D[idx_tr]
    # observed outcome is Y[row, decision(row)]
    train_y = Y[idx_tr, train_d].reshape(Ntrain, 1)

    query_x = X[idx_qu, :]
    query_d = D[idx_qu]
    query_y = Y[idx_qu, query_d].reshape(Nquery, 1)

    test_x = X[idx_te, :]
    test_Y = Y[idx_te, :]  # keep all decisions at test points, like your code

    train = {"x": train_x.detach().cpu().numpy(),
             "y": train_y.detach().cpu().numpy(),
             "d": train_d.detach().cpu().numpy()}
    query = {"x": query_x.detach().cpu().numpy(),
             "y": query_y.detach().cpu().numpy(),
             "d": query_d.detach().cpu().numpy()}
    test  = {"x": test_x.detach().cpu().numpy(),
             "Y": test_Y.detach().cpu().numpy()}

    meta = {
        "X": X.detach().cpu().numpy(),
        "D": D.detach().cpu().numpy(),
        "Y": Y.detach().cpu().numpy(),
        "l": l.detach().cpu().numpy(),
        "v": v.detach().cpu().numpy(),
        "idx_tr": idx_tr.detach().cpu().numpy(),
        "idx_qu": idx_qu.detach().cpu().numpy(),
        "idx_te": idx_te.detach().cpu().numpy(),
    }
    ### plotting only possible for 1 dimension
    # plot_train_query_test_1d(train, query, test, Ndec=config.Ndec, filename=f"Syndata_all_{str(config.Ndec)}.png")

    return train, query, test, meta


