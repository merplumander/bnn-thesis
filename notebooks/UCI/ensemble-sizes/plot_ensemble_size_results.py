# %load_ext autoreload
# %autoreload 2
import copy
import json
from collections import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from core.plotting_utils import (
    plot_uci_ensemble_size_benchmark,
    plot_uci_single_benchmark,
)

figure_dir = "figures/temp"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)


colors = sns.color_palette()
with open("config/uci-color-config.yaml") as f:
    color_mapping = yaml.full_load(f)
figsize = (18, 6)
legend_kwargs = {"bbox_to_anchor": (1.0, 1), "loc": "upper left"}

datasets = [
    "boston",
    "concrete",
    "energy",
    "kin8nm",
    # "naval",
    "power",
    "protein",
    "wine",
    "yacht",
]

# %% markdown


# %%
patience = 20
n_comb_max = 25
experiment_name = f"ensemble-sizes_l2-convergence_n-comb-{n_comb_max}_early-stop-patience-{patience}_one-hidden-layer"

for dataset in datasets:
    experiment_path = Path(
        f"uci_data/{dataset}/results/ensemble_sizes/{experiment_name}.json"
    )
    results = json.loads(experiment_path.read_text())

    e_rmses = results["Ensemble"][
        "RMSEs"
    ]  # n_ensemble_members, n_splits, n_different_ensembles
    e_nlls = results["Ensemble"]["NLLs"]
    e_mm_nlls = results["Ensemble"]["MM-NLLs"]
    llbe_rmses = results["LLB Ensemble"]["RMSEs"]
    llbe_nlls = results["LLB Ensemble"]["NLLs"]
    llbe_mm_nlls = results["LLB Ensemble"]["MM-NLLs"]
    labels = ["Ensemble MM", "Ensemble", "LLB Ensemble MM", "LLB Ensemble"]

    plot_uci_ensemble_size_benchmark(
        [e_mm_nlls, e_nlls, llbe_mm_nlls, llbe_nlls],
        labels=labels,
        title=dataset,
        x_label="# ensemble_memebers",
        y_label="Negative Log Likelihood",
        x_lim=[None, n_comb_max],
        colors=[colors[color_mapping[method]] for method in labels],
        save_path=figure_dir.joinpath(
            f"uci_ensemble-sizes_{dataset}_{experiment_name}.pdf"
        ),
    )
