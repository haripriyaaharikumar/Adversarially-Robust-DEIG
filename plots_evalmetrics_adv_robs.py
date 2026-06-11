import numpy as np
import matplotlib.pyplot as plt
import os

al_types = ["random_sampling", "us_deig", "peig", "targeted_oeig", "decision_eig", "arobust_deig"]

altype_label = {
    "random_sampling": "RS",   
    "us_deig": "US",          
    "peig": "PEIG",              
    "targeted_oeig": "TEIG",     
    "decision_eig": "DEIG",      
    "arobust_deig": "AR-DEIG",     
}

color_map = {
    "random_sampling": "#1f77b4",   
    "us_deig": "#ff7f0e",           
    "peig": "#9467bd",              
    "targeted_oeig": "#d62728",     
    "decision_eig": "#ff4da6",      
    "arobust_deig": "#2ca02c",      
}



run_end = 500

run_ids = range(1, run_end)

endval = 5000          # number of adversarial samples you evaluate on

alpha = 0.1           # CVaR lower tail fraction (10%)

plt.figure(figsize=(8, 6))
Nacq = 20
eps = 0.3
deci = 3
dim = 1


base_path1 = f"./results/1d3dec_Nadv5000_eps0.3/"
base_path2 = f"./results/1d3dec_Nadv5000_eps0.3/"

base_path3 = base_path2

def cvar_lower(x: np.ndarray, alpha: float = 0.1) -> float:
    """Average of the worst alpha-fraction of x."""
    x = np.asarray(x).reshape(-1)
    k = max(1, int(np.ceil(alpha * x.size)))
    return np.sort(x)[:k].mean()

metrics_to_plot = ["mean", "worst", "cvar"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
for altype in al_types:
    # store per-seed metric curves; each element becomes (A,) after aggregation over adv samples
    curves_mean = []
    curves_worst = []
    curves_cvar = []

    base_path = base_path2 if altype == "arobust_deig" else base_path1

    for r in run_ids:
        dpred_path = os.path.join(base_path, f"{altype}_dpred_{dim}deci{deci}seeds_{r}.npy")

        d_pred = np.load(dpred_path)  # (A,)

        Y_adv_path = os.path.join(base_path2, f"Y_adv_{altype}{dim}deci{deci}seeds_{r}.npy")
   
        Y_all_adv = np.load(Y_adv_path)  # shape: (Nadv_eval, Ndec) or similar
    
        A = len(d_pred) # number of acquisitions
        m = np.zeros(A)
        w = np.zeros(A)
        c = np.zeros(A)

        for i in range(A):
            id_pred = int(d_pred[i])
            y = Y_all_adv[:endval, id_pred]   # (endval,)
            # print (y.shape)
            
            m[i] = y.mean()
            w[i] = y.min()
            c[i] = cvar_lower(y, alpha=alpha)

        curves_mean.append(m)
        curves_worst.append(w)
        curves_cvar.append(c)
    
    print (altype)

    curves_mean = np.vstack(curves_mean)   # (S, A)
    curves_worst = np.vstack(curves_worst)
    curves_cvar = np.vstack(curves_cvar)

    values = np.min(curves_worst, axis=1)

    mean_mu = curves_mean.mean(axis=0)
    mean_se = curves_mean.std(axis=0) / np.sqrt(curves_mean.shape[0])

    worst_mu = curves_worst.mean(axis=0)
    worst_se = curves_worst.std(axis=0) / np.sqrt(curves_worst.shape[0])

    cvar_mu = curves_cvar.mean(axis=0)
    cvar_se = curves_cvar.std(axis=0) / np.sqrt(curves_cvar.shape[0])

    steps = np.arange(len(mean_mu))
    K = Nacq+1  

    
    if altype == "arobust_deig":
        axes[0].plot(
            steps[:K], mean_mu[:K],
            marker="o",
            linestyle="-", 
            color=color_map[altype],
            label="AR-DEIG (ours)"   # custom legend name
                      # draw on top
        )
    else:
        axes[0].plot(steps[:K], mean_mu[:K], marker="o",color=color_map[altype], label=altype_label[altype])
    axes[0].fill_between(steps[:K], mean_mu[:K] - mean_se[:K], mean_mu[:K] + mean_se[:K], alpha=0.15,color=color_map[altype])

    if altype == "arobust_deig":
        axes[1].plot(steps[:K], worst_mu[:K],
            marker="o",
            linestyle="-", color=color_map[altype],
            label="AR-DEIG (ours)"   # custom legend name
            )             # draw on top
    else:
         axes[1].plot(steps[:K], worst_mu[:K], marker="o", label=altype_label[altype],color=color_map[altype])

    axes[1].fill_between(steps[:K], worst_mu[:K] - worst_se[:K], worst_mu[:K] + worst_se[:K], alpha=0.15,color=color_map[altype])

    if altype == "arobust_deig":
        axes[2].plot(steps[:K], cvar_mu[:K],
            marker="o",
            linestyle="-", color=color_map[altype],
            label="AR-DEIG (ours)")
                         # draw on top
    else:
        axes[2].plot(steps[:K], cvar_mu[:K],
            marker="o", color=color_map[altype],
            linestyle="-",label=altype_label[altype])
    axes[2].fill_between(steps[:K], cvar_mu[:K] - cvar_se[:K], cvar_mu[:K] + cvar_se[:K], alpha=0.15,color=color_map[altype])

axes[0].set_title("Mean over perturbations")
axes[1].set_title("Worst-case (min over samples)")
axes[2].set_title(f"CVaR{int(alpha*100)} (lower tail)")
for ax in axes:
    ax.set_xlabel("Number of acquisition",fontsize=15)
    ax.set_xticks(np.arange(0, K, 1))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.5)


axes[0].set_ylabel("Outcome under perturbed contexts", fontsize=15)

# axes[0].legend(framealpha=0.3)
axes[0].legend(
    loc="upper center",
    ncol=3, fontsize=12,
    frameon=False
)
plt.tight_layout()

plt.savefig(f"{dim}d{deci}_mean_worst_cvar_eps{eps}.png")
