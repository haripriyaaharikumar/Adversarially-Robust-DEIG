import torch

from TAL_Syndata_wd_Advdata_nonsmooth import generation_gpytorch_like_gpy

from models import initialize_GPs
from active_learning_advrobust import activelearning


import config as config
import numpy as np
import argparse
import sys
import os 

if __name__ == "__main__":
    seed = int(sys.argv[1])
    device = 'cpu'

    train, query, test, adv_test, meta = generation_gpytorch_like_gpy(
        dim=config.dim,
        std=config.std,
        Ndec=config.Ndec,
        Ntrain=config.Ntrain,
        Nquery=config.Nquery,
        Ntest=config.Ntest,
        seed_split=seed,
        device="cpu",
        dtype=torch.float64,
        add_adv_eval=True,
        jitter=1e-6,
        n_adv_eval=10,
        n_grid=500,
        grid_range=(-2.0, 2.0),
    )
 
    

    M = initialize_GPs(
        x_train=train,
        K_decisions=config.Ndec,
        device=device,
        dtype=torch.float64,
        iters=250,
        lr=0.05,
    )
 
    altype = "arobust_deig"
   

    acc, dpred, H = activelearning(
        M=M,
        train=train,
        query=query,
        test=test,
        adv_test=adv_test,
        Ndec=config.Ndec,
        Nacq=config.Nacq,
        Ngh=config.N_GH,
        criterion=altype,
        device=device,
        dtype=torch.float64,
        iters_retrain=250,
        lr=0.05,
        seed=config.seed,
        entropy_mode="diag",
    )
    
    path = f'./..../{config.dim}d{config.Ndec}dec_Nadv{config.Nadv_test}_eps{config.eps_adv}_{len(config.adv_dims)}dadv_Nacq{config.Nacq}_Npost{config.n_post}/'

    root_name = f'{path}'
    if not os.path.exists(f'{path}'):
        os.mkdir(f'{path}')

    root_name = f'{path}{config.seed}'
    if not os.path.exists(f'{path}{config.seed}'):
        os.mkdir(f'{path}{config.seed}')

    np.save(root_name + f'/{altype}_acc_' +str(config.dim)+'deci'+str(config.Ndec)+'seeds_'+str(seed) + '.npy', acc)
    np.save(root_name + f'/{altype}_H_' +str(config.dim)+'deci'+str(config.Ndec)+'seeds_'+str(seed) + '.npy', H)
    np.save(root_name + f'/{altype}_dpred_' +str(config.dim)+'deci'+str(config.Ndec)+'seeds_'+str(seed) + '.npy', dpred)

