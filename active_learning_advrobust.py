from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from utils_adv import targeted_oeig, decision_eig, arobust_deig, arobust_deig_max_xa, random_sampling, peig, us_deig, _pred_mean_var_y

from models import _to_2d  
import gpytorch
import config as config
import copy
import time

@torch.no_grad()
def estimate_entropy(
    M,
    X: np.ndarray,
    Ns: int,
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
            post = model(X_t) ## assume this is noiseless
            mu_list.append(post.mean)
            var_list.append(post.variance.clamp_min(1e-12))

    mu = torch.stack(mu_list, dim=1)   # [N,K]
    var = torch.stack(var_list, dim=1) # [N,K]

    eps = torch.randn((Ns, N, K), dtype=dtype, device=device)
    samples = mu.unsqueeze(0) + torch.sqrt(var).unsqueeze(0) * eps

    dbest = torch.argmax(samples, dim=2)  # [Ns, N]

    one_hot = F.one_hot(dbest.to(torch.int64), num_classes=K).to(dtype)  # [Ns,N,K]
    p = one_hot.mean(dim=0).clamp_min(torch.finfo(dtype).eps)            # [N,K]

    H = -(p * p.log()).sum(dim=1)  # [N]
    # print (H)
    return H.detach().cpu().numpy()[0]

@torch.no_grad()
def compute_acc(
    M,
    Ndec: int,
    X_test: np.ndarray,
    d_true: np.ndarray,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> float:
    X = torch.as_tensor(_to_2d(X_test), dtype=dtype, device=device)
    means = []
    for k in range(Ndec):
        model_k, lh_k = M[k]
        mu_k, _ = _pred_mean_var_y(model_k, lh_k, X) ### noisy prediction
        means.append(mu_k.reshape(-1, 1))

    U = torch.cat(means, dim=1)
    d_pred = torch.argmax(U, dim=1).detach().cpu().numpy()
    d_true = d_true.reshape(-1)
    return float((d_pred == d_true).mean()), float(d_pred[0])

@torch.no_grad()
def compute_acc_adv(
    d_true: np.ndarray,
    d_pred: np.ndarray,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> float:
    d_true = d_true.reshape(-1)
    # print (d_true)
    # print (d_pred)
    # print ( float((d_pred == d_true).mean()))
    return float((d_pred == d_true).mean())

def activelearning(
    M,
    train,
    query,
    test,
    adv_test,
    Ndec: int,
    Nacq: int,
    Ngh: int,
    criterion: str,
    *,
    device,
    dtype: torch.dtype = torch.float64,
    iters_retrain: int = 250,
    lr: float = 0.05,
    seed: int,
    entropy_mode: str = "diag",
) -> Tuple[np.ndarray, np.ndarray]:
    acc = np.zeros(Nacq + 1, dtype=float)
    dpred = np.zeros(Nacq + 1, dtype=float)
    P = np.zeros(Nacq + 1, dtype=float)
    acc_advs = np.zeros((Nacq + 1,config.Nadv_test), dtype=float)
   
    d_true = np.argmax(test["Y"], axis=1)
    y_true = np.max(test["Y"], axis=1)
    data = []

    M0 = [[copy.deepcopy(mdl), lh]  for (mdl, lh) in M]
    acc[0],dpred[0] = compute_acc(M0, config.Ndec, test["x"], d_true, device=device, dtype=dtype)
  
    P[0]   = estimate_entropy(M0, test["x"],Ns=config.Ns, device=device, dtype=dtype)
  

    for i in range(Nacq):
        # Select index of next query
        t1 = time.time()
        if criterion == "random_sampling":
            i_star = random_sampling(query)

        if criterion == "uncertainty_random_sampling":
            i_star = random_sampling(query)

        elif criterion == "us_deig":
            i_star = us_deig(M0,query,device=device,dtype=dtype)

        elif criterion == "peig":
            i_star = peig(M0,query,
                n_gh=Ngh,normalize_gh=True,entropy_mode=entropy_mode,
                device=device,dtype=dtype,
            )
        elif criterion == "targeted_oeig":
            i_star = targeted_oeig(M0,query,test,
                n_gh=Ngh,normalize_gh=True,entropy_mode=entropy_mode,
                device=device,dtype=dtype,
            )
        elif criterion == "decision_eig":
            i_star = decision_eig(M0,train,query,test,n_gh=Ngh,
                normalize_gh=True,entropy_mode=entropy_mode,
                device=device,dtype=dtype,
            )
        elif criterion == "arobust_deig":
            i_star = arobust_deig(M0,train,query,test,n_gh=Ngh,
                normalize_gh=True,entropy_mode=entropy_mode,
                device=device,dtype=dtype,adv_dims=config.adv_dims,
                eps_adv=config.eps_adv, n_adv=config.Nadv,
                n_post=config.n_post, seed=seed)
        elif criterion == "arobust_deig_maxxa":
            i_star = arobust_deig_max_xa(M0,train,query,test,n_gh=Ngh,
                normalize_gh=True,entropy_mode=entropy_mode,
                device=device,dtype=dtype,adv_dims=config.adv_dims,
                eps_adv=config.eps_adv, n_adv=config.Nadv, n_post=config.n_post, seed=seed)
        else:
            raise ValueError(f"Unknown acquisition criterion '{criterion}'")
        t2 = time.time()
        print('Data point selected... Took ' + str(t2-t1) + ' sec.')
        dq = int(query["d"][i_star])
        print(f"Acq number {i+1} selected decision {dq} at ")

        # Move point query -> train
        x_new = query["x"][i_star : i_star + 1, :]
        y_new = query["y"][i_star : i_star + 1, :]
        train["x"] = np.vstack((train["x"], x_new))
        train["y"] = np.vstack((train["y"], y_new))
        train["d"] = np.hstack((train["d"], np.array([dq], dtype=train["d"].dtype)))
        ### append it to the data list

        query["x"] = np.delete(query["x"], i_star, axis=0)
        query["y"] = np.delete(query["y"], i_star, axis=0)
        query["d"] = np.delete(query["d"], i_star, axis=0)

        # Retrain only the GP for decision dq
        idx = np.where(train["d"] == dq)[0]
        X_dq = torch.as_tensor(train["x"][idx, :], dtype=dtype, device=device)
        y_dq = torch.as_tensor(train["y"][idx, :].reshape(-1), dtype=dtype, device=device)

        model, lik = M0[dq]
        model.set_train_data(inputs=X_dq, targets=y_dq, strict=False)

        model.train()
        lik.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

        for _ in range(iters_retrain):
            opt.zero_grad(set_to_none=True)
            out = model(X_dq)
            loss = -mll(out, y_dq)
            loss.backward()
            opt.step()

        model.eval()
        lik.eval()
        M0[dq] = (model, lik)

        acc[i + 1],dpred[i+1] = compute_acc(M0, Ndec, test["x"], d_true, device=device, dtype=dtype)
        # dpred_advs = np.full((config.Nadv_test,), dpred[i+1])

        P[i + 1]   = estimate_entropy(M0, test["x"], Ns=config.Ns,device=device, dtype=dtype)
        # acc_advs[i+1] = compute_acc_adv(d_true_advs, dpred_advs, device=device, dtype=dtype)
    
    return acc, dpred, P
