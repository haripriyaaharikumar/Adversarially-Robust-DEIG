# Adversarially-Robust-DEIG : Accepted paper in UAI 2026 Robust Bayesian Decision Making under Adversarial Uncertainty

Install the following Python libraries before running the code:

torch

torchvision

gpytorch

scikit-learn

numpy

matplotlib

pandas



Running the Code

1. Configure the experiment settings

    Modify the configuration parameters in config.py, such as:

        epsilon values (eps_adv)

        dimensions (dim)

        decisions (Ndec)

        adversarialdimensions (adv_dims)

2. Select the acquisition function

    Choose your desired acquisition function in:

        main_nonsmooth.py


3. Plotting Adversarial Robustness Metrics


To generate plots for adversarial robustness metrics (e.g., mean, worst-case, and CVaR10), run experiments with:

    1 dimension

    3 decisions

Intermediate results for ε = 0.3 used for plotting are stored in:

    results/1d3dec_Nadv5000_eps0.3/

Generate the evaluation metric plots using:

    python plots_evalmetrics_adv_robs.py

