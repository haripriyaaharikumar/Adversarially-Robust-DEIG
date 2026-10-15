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


plt.figure(figsize=(8, 6))

Nacq = 20
eps = 0.3
deci = 3
dim = 1



base_path1 = f"./results/1d3dec_Nadv5000_eps0.3/"
base_path2 = f"./results/1d3dec_Nadv5000_eps0.3/"


base_path3 = base_path2

for altype in al_types:
    flip_fraction_seeds = []   # list of (Nacq,) arrays

    base_path = base_path2 if altype == "arobust_deig" else base_path1
    all_decisions = []
    for r in run_ids:
        dpred_path = os.path.join(base_path, f"{altype}_dpred_{dim}deci{deci}seeds_{r}.npy")
        d_pred = np.load(dpred_path) 
        all_decisions.append(d_pred)
    H = np.stack(all_decisions, axis=1)
    print (H.shape)  
    flip_fraction = (H[1:] != H[:-1]).mean(axis=1)
    flip_fraction_seeds.append(flip_fraction)
    flip_fraction_seeds = np.stack(flip_fraction_seeds, axis=0)  

    mean_flip = flip_fraction_seeds.mean(axis=0)
    std_flip  = flip_fraction_seeds.std(axis=0)
    mean_flip= mean_flip[0:Nacq]
    std_flip = std_flip[0:Nacq]
    x = np.arange(1, Nacq + 1)
    if altype == "arobust_deig":
        plt.plot(
            x, mean_flip,
            marker="o",
            linestyle="-",
            label="AR-DEIG (ours)", color=color_map[altype])
    else:
            plt.plot(x, mean_flip, marker="o", color=color_map[altype],label=altype_label[altype])

    plt.fill_between(x, mean_flip - std_flip, mean_flip + std_flip, alpha=0.2)
    plt.xticks(np.arange(1, Nacq + 1),fontsize=12)  
plt.yticks(fontsize=12)
plt.xlabel("Number of acquisition",fontsize=14)
plt.ylabel("Fraction of test decisions flipped",fontsize=14)
plt.grid(True,alpha=0.5)
plt.tight_layout()
plt.legend(framealpha=0.3,fontsize=14)


plt.savefig(f"{dim}d{deci}dec_1dadv_robust_eps{eps}_Nacq_{Nacq}decision_flips.png")
