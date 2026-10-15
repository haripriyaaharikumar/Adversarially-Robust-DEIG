
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import gpytorch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Sequence

from models import _to_2d  
import config as config
import copy


@torch.no_grad()
def _pred_mean_var_y(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Observation-space predictive mean/variance for y:
      p(y|X,D) from likelihood(model(X))
    """
    model.eval()
    likelihood.eval()
    with gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X))
        return pred.mean, pred.variance.clamp_min(1e-12)


@torch.no_grad()
def _pred_mean_covar_f(
    model: gpytorch.models.ExactGP,
    X: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Latent predictive mean/covariance for f:
      p(f|X,D) from model(X)
    """
    model.eval()
    with gpytorch.settings.fast_pred_var():
        post = model(X)
        return post.mean, post.covariance_matrix


def _gaussian_entropy_diag_from_var(var: torch.Tensor) -> torch.Tensor:
    """
    Differential entropy of independent Gaussians (diagonal approx):
      H ≈ Σ_t 0.5*log(2πe v_t)
    """
    var = var.squeeze(-1) if var.ndim == 2 and var.shape[-1] == 1 else var
    return torch.log(torch.sqrt(2 * torch.pi * torch.e * var))


def _gaussian_entropy_full_from_covar(covar: torch.Tensor) -> torch.Tensor:
    """
    Differential entropy of multivariate Gaussian with full covariance Σ:
      H = 0.5 * [ T*log(2πe) + log det(Σ) ]
    """
    T = covar.shape[0]
    jitter = 1e-8 * torch.eye(T, dtype=covar.dtype, device=covar.device)
    sign, logabsdet = torch.linalg.slogdet(covar + jitter)
    return 0.5 * (T * torch.log(2 * torch.pi * torch.e) + logabsdet)

@torch.no_grad()
def decision_entropy(
    M,
    X: np.ndarray,
    *,
    device,
    dtype: torch.dtype = torch.float64,
) -> np.ndarray:
    K = len(M)
    X_t = torch.as_tensor(X, dtype=dtype, device=device)  # [N,D]
    N = X_t.size(0)

    mu_list, var_list = [], []
    with gpytorch.settings.fast_pred_var():
        for model, _lh in M:
            model.eval()
            post = model(X_t) ## assume this is noiseless, retunrs f(X_t) - latent function
            mu_list.append(post.mean)
            var_list.append(post.variance.clamp_min(1e-12))

    mu = torch.stack(mu_list, dim=1)   # [N,K]
    var = torch.stack(var_list, dim=1) # [N,K]

    eps = torch.randn((config.Ns, N, K), dtype=dtype, device=device) ##Ns is 10000 in config, which is pretty good
    samples = mu.unsqueeze(0) + torch.sqrt(var).unsqueeze(0) * eps

    dbest = torch.argmax(samples, dim=2)  # [Ns, N]

    one_hot = F.one_hot(dbest.to(torch.int64), num_classes=K).to(dtype)  # [Ns,N,K]
    p = one_hot.mean(dim=0).clamp_min(torch.finfo(dtype).eps)            # [N,K]

    H = -(p * p.log()).sum(dim=1)  # [N] - size of the input which is one - for a single test data
    return H

# -----------------------------------------------------------------------------
# Adversarially Robust Decision-EIG for GPyTorch
# -----------------------------------------------------------------------------

def _validate_adv_dims(D: int, adv_dims: Optional[Sequence[int]]) -> List[int]:
    if adv_dims is None:
        return []
    return [int(a) for a in adv_dims if 0 <= int(a) < D]


def make_adv_points_torch(x: torch.Tensor, adv_dims: Optional[Sequence[int]], eps: float | Sequence[float], n_adv: int, *, device: Optional[str] = None, dtype: Optional[torch.dtype] = None, rng: Optional[np.random.Generator] = None) -> torch.Tensor:

    # assert x.ndim == 2 and x.shape[0] == 1, "x must be shape (1, D)"
    D = x.shape[1]
    adv_dims = _validate_adv_dims(D, adv_dims)

    dev = device or str(x.device)
    dt = dtype or x.dtype
    if len(adv_dims) == 0:
        return x.clone()

    if np.isscalar(eps):
        eps_vec = np.full(len(adv_dims), float(eps))
    else:
        eps_vec = np.asarray(eps, dtype=float)
        assert len(eps_vec) == len(adv_dims)

    rng = rng or np.random.default_rng()
    deltas_np = rng.uniform(-1.0, 1.0, size=(n_adv, len(adv_dims))) * eps_vec[None, :]
    X_adv = x.repeat(n_adv, 1).to(device=dev, dtype=dt)
    for j, d in enumerate(adv_dims):
        X_adv[:, d] += torch.from_numpy(deltas_np[:, j]).to(device=dev, dtype=dt)
    return X_adv



@torch.no_grad()
def estimate_robust_decision_entropy_mc(
    M, x_test,
    n_post=200,
    adv_dims=None, eps_adv= 0.1, n_adv= 200, seed= 0,
    device=None, dtype: torch.dtype = torch.float64,
):
    if x_test.ndim == 1:
        x_test = x_test.unsqueeze(0)

    dev = device or x_test.device
    dt = dtype or x_test.dtype
    rng = np.random.default_rng(seed)

    X_adv = make_adv_points_torch(
        x_test, adv_dims, eps_adv, n_adv,
        device=str(dev), dtype=dt, rng=rng
    )

    K = len(M)
    posts = []
    with gpytorch.settings.fast_pred_var():
        for model, _lh in M:
            model.to(dev)
            (model.double() if dt == torch.float64 else model.float())
            model.eval()
            posts.append(model(X_adv))  # joint MVN over f(X_adv): noiseless, same as decision

    # sample jointly for each arm: [n_post, K, n_adv]
    samps = torch.stack(
        [posts[k].rsample(sample_shape=torch.Size([n_post])).squeeze(-1) for k in range(K)],
        dim=1
    )

    robust_scores = samps.min(dim=2).values   # [n_post, K]
    dbest = robust_scores.argmax(dim=1)       # [n_post]

    counts = torch.bincount(dbest, minlength=K).to(dtype=dt)
    p = (counts / counts.sum().clamp_min(1.0)).clamp_min(torch.finfo(dt).eps)
    H = -(p * p.log()).sum()
    return H


###########################################################################
# Acquisition functions
###########################################################################


## Adversarially Robust Decision EIG - min over decision
@torch.no_grad()
def arobust_deig(
    models,
    Xtrain: np.ndarray,  # kept for signature compatibility; not used in your code
    Xcand: np.ndarray,
    Xtest: np.ndarray,
    n_gh: int = 11,
    normalize_gh: bool = True,
    entropy_mode: str = "diag",   # "diag" or "full"
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    adv_dims=[0],
    eps_adv=0.1, 
    n_adv=64, 
    n_post=200,
    seed=10):

    Xcand_t = torch.from_numpy(_to_2d(Xcand["x"])).to(device=device, dtype=dtype)
    Xtest_t = torch.from_numpy(_to_2d(Xtest["x"])).to(device=device, dtype=dtype)

    z_np, w_np = np.polynomial.hermite.hermgauss(n_gh)
    z = torch.from_numpy(z_np).to(device=device, dtype=dtype)
    w = torch.from_numpy(w_np).to(device=device, dtype=dtype)

    gh_norm = (1.0 / math.sqrt(math.pi)) if normalize_gh else 1.0
    scores = torch.empty(Xcand_t.shape[0], dtype=dtype, device=device)

    for i in range(Xcand_t.shape[0]):
        xq = _to_2d(Xcand_t[i])
        dq = int(Xcand["d"][i])

        mdl, lh = models[dq]
        # print (len(models))
        mu_q, var_q = _pred_mean_var_y(mdl, lh, xq)
        mu_q = mu_q.squeeze()
        var_q = var_q.squeeze()

        y_gh = mu_q + torch.sqrt(2.0 * var_q) * z
        exp_H_after = torch.zeros((), dtype=dtype, device=device)

        for j in range(n_gh):
            models_cp = [
                [copy.deepcopy(mdl), lh]   # list, not tuple
                for (mdl, lh) in models
            ]
            mdl_cp, lh_cp = models_cp[dq]
            mu_q, var_q = _pred_mean_var_y(mdl_cp, lh_cp, xq)
            yq = y_gh[j : j + 1]
            fant_model = mdl_cp.get_fantasy_model(xq, yq)
            models_cp[dq] = (fant_model, lh)
            _, cov = _pred_mean_covar_f(fant_model, Xtest_t)
            H_rob = estimate_robust_decision_entropy_mc(models_cp, Xtest_t, n_post, 
                                                        adv_dims=adv_dims, eps_adv=eps_adv, 
                                                        n_adv=n_adv, seed=seed+i*13+j*17,
                                                          device=device, dtype=dtype)
            exp_H_after = exp_H_after + w[j] * H_rob

        # scores[i] = gh_norm * exp_H_after
        scores[i] = exp_H_after

    return np.argmin(scores.detach().cpu().numpy())
