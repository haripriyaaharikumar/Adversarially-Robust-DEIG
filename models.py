from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import gpytorch
import os
import random

def set_seed(seed: int = 1432, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dtype: torch.dtype = torch.float64):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = torch.tensor(1.0, dtype=dtype)
        self.covar_module.outputscale = torch.tensor(1.0, dtype=dtype)

    def forward(self, x_):
        mean_x = self.mean_module(x_)
        covar_x = self.covar_module(x_)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _to_2d(x: np.ndarray) -> np.ndarray:
    return x.reshape(1, -1) if x.ndim == 1 else x


def train_exact_gp(
    X: torch.Tensor,
    y: torch.Tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    iters: int = 250,
    lr: float = 0.05,
    noise_init: Optional[float] = None,
) -> Tuple[ExactGPModel, gpytorch.likelihoods.GaussianLikelihood]:
    X = X.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=dtype)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
    if noise_init is not None:
        likelihood.noise = torch.as_tensor(noise_init, dtype=dtype, device=device)

    model = ExactGPModel(X, y, likelihood, dtype=dtype).to(device)
    model.train()
    likelihood.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(iters):
        opt.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        opt.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


def initialize_GPs(
    x_train: dict,
    K_decisions: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    iters: int = 250,
    lr: float = 0.05,
) -> List[Tuple[ExactGPModel, gpytorch.likelihoods.GaussianLikelihood]]:

    set_seed(1432, deterministic=True)
    X_all = torch.from_numpy(_to_2d(x_train["x"])).to(device=device, dtype=dtype)

    y_all = torch.from_numpy(x_train["y"])
    if y_all.ndim > 1:
        y_all = y_all.squeeze(-1)
    y_all = y_all.to(device=device, dtype=dtype)

    models_per_arm: List[Tuple[ExactGPModel, gpytorch.likelihoods.GaussianLikelihood]] = []

    for k in range(K_decisions):
        mask = (x_train["d"] == k)
        if mask.sum() == 0:
            Xk = torch.zeros((1, X_all.shape[1]), dtype=dtype, device=device)
            yk = torch.zeros((1,), dtype=dtype, device=device)
        else:
            Xk = X_all[mask]
            yk = y_all[mask]

        mdl, lh = train_exact_gp(Xk, yk, device=device, dtype=dtype, iters=iters, lr=lr)
        models_per_arm.append((mdl, lh))

    return models_per_arm
