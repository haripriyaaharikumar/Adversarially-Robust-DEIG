import numpy as np
import torch
import gpytorch
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import math
import config

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
    mix = dirich.sample()  

    p = (0.65 / Ndec) * torch.ones((Ndec,), dtype=dtype, device=device) + 0.35 * mix
    p = p / p.sum()

    D = torch.multinomial(p, num_samples=Ntot, replacement=True)  
    return D


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
    jitter: float = 1e-6,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:

    torch.manual_seed(config.seed)
    gen_data = torch.Generator(device=device)
    gen_data.manual_seed(config.seed)

    Ntot = Ntrain + Nquery + Ntest #Ntest is 1

    X = torch.randn((Ntot, dim), device=device, dtype=dtype, generator=gen_data)

    D = gen_decision_torch(Ntrain, Nquery, Ndec, device=device, dtype=dtype)

    Y = torch.zeros((Ntot, Ndec), dtype=dtype, device=device)

    l = math.sqrt(dim) * (0.25 + 0.75 * torch.rand((Ndec, dim), generator=gen_data, device=device, dtype=dtype))
    v = 0.5 + 2.0 * torch.rand((Ndec,), generator=gen_data, device=device, dtype=dtype)

    def sample_gp_at_X_gpytorch_ard(
        X_: torch.Tensor,
        lengthscale_vec: torch.Tensor,  
        variance_scalar: torch.Tensor,  
        jitter_: float,
        generator_: torch.Generator,
    ) -> torch.Tensor:
        base = gpytorch.kernels.RBFKernel(ard_num_dims=X_.shape[1]).to(device=X_.device, dtype=X_.dtype)
        base.lengthscale = lengthscale_vec.view(1, 1, -1)  

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
        f_i = f_i + std * torch.randn((Ntot,), dtype=dtype, device=device, generator=gen_data)
        Y[:, i] = f_i

    torch.manual_seed(seed_split)
    gen_split = torch.Generator(device=device)
    gen_split.manual_seed(seed_split)

    idx_tot = torch.arange(Ntot, device=device)

    idx_tr = torch.empty((Ntrain,), dtype=torch.long, device=device)

    for i in range(Ndec):
        idx_d = torch.nonzero(D == i, as_tuple=False).squeeze(-1)
        if idx_d.numel() == 0:
            raise RuntimeError(
                f"Decision {i} has zero samples in D. Increase Ntot or adjust decision sampling."
            )
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
    idx_te = idx_left[Nquery:] 

    train_x = X[idx_tr, :]
    train_d = D[idx_tr]
    train_y = Y[idx_tr, train_d].reshape(Ntrain, 1)

    query_x = X[idx_qu, :]
    query_d = D[idx_qu]
    query_y = Y[idx_qu, query_d].reshape(Nquery, 1)

    test_x = X[idx_te, :]
    test_Y = Y[idx_te, :]  

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

    return train, query, test, meta


