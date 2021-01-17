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

from core.evaluation_utils import (
    get_y_normalization_scales,
    normalize_ll,
    normalize_rmse,
)
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
    "naval",
    "power",
    "protein",
    "wine",
    "yacht",
]


# %%
gap_data = False
n_networks = 50
n_hidden_layers = 2
hidden_layers_string = (
    "two-hidden-layers" if n_hidden_layers == 2 else "one-hidden-layer"
)


if gap_data:
    data_dir = "uci_gap_data"
    experiment_name = f"uci-gap_ensemble-size-convergence_{hidden_layers_string}"
else:
    data_dir = "uci_data"
    experiment_name = f"uci_ensemble-size-convergence_{hidden_layers_string}"


# %% markdown


# %%
_net_sizes = np.arange(n_networks) + 1.0
_net_sizes = _net_sizes[np.logical_or(_net_sizes <= 20, _net_sizes % 5 == 0)]
for dataset in datasets[2:3]:
    experiment_path = Path(
        f"{data_dir}/{dataset}/results/ensemble_sizes/{experiment_name}.json"
    )

    y_normalization_scales = get_y_normalization_scales(dataset, gap_data=gap_data).T
    y_normalization_scales.shape

    results = json.loads(experiment_path.read_text())

    e_rmses = np.array(
        results["Ensemble"]["RMSEs"]
    )  # n_ensemble_members, n_splits, n_different_ensembles
    # e_rmses = normalize_rmse(e_rmses, y_normalization_scales)
    e_nlls = np.array(results["Ensemble"]["NLLs"])
    # e_nlls = normalize_ll(e_nlls, y_normalization_scales)

    # e_mm_nlls = results["Ensemble"]["MM-NLLs"]
    llbe_rmses = np.array(results["LLB Ensemble"]["RMSEs"])
    # llbe_rmses = normalize_rmse(llbe_rmses, y_normalization_scales)
    llbe_nlls = np.array(results["LLB Ensemble"]["NLLs"])
    # llbe_nlls = normalize_ll(llbe_nlls, y_normalization_scales)
    # llbe_mm_nlls = results["LLB Ensemble"]["MM-NLLs"]
    labels = ["Ensemble", "LLB Ensemble"]

    fig, ax = plt.subplots(figsize=(15, 5))
    plot_uci_ensemble_size_benchmark(
        [e_rmses, llbe_rmses],
        x=_net_sizes,
        labels=labels,
        title=dataset,
        fig=fig,
        ax=ax,
        x_label="# ensemble_memebers",
        y_label="Negative Log Likelihood",
        colors=[colors[color_mapping[method]] for method in labels],
        # save_path=figure_dir.joinpath(
        #     f"uci_ensemble-sizes_{dataset}_{experiment_name}.pdf"
        # ),
    )


# %%
_net_sizes = np.arange(n_networks) + 1.0
_net_sizes = _net_sizes[np.logical_or(_net_sizes <= 20, _net_sizes % 5 == 0)]

fig, ax = plt.subplots(figsize=(15, 6))
pareto_point = True
method = "Ensemble"
metric = "NLLs"
for i, dataset in enumerate(datasets[:]):
    experiment_path = Path(
        f"{data_dir}/{dataset}/results/ensemble_sizes/{experiment_name}.json"
    )

    y_normalization_scales = get_y_normalization_scales(dataset, gap_data=gap_data).T
    y_normalization_scales.shape

    results = json.loads(experiment_path.read_text())
    results = np.array(
        results[method][metric]
    )  # n_ensemble_members, n_splits, n_different_ensembles

    labels = [f"{dataset}"]
    y_label = "RMSE" if metric == "RMSEs" else "Negative log likelihood"
    plot_uci_ensemble_size_benchmark(
        [results],
        normalization=True,
        x=_net_sizes,
        x_add_jitter=0.01 * i,
        errorbars=False,
        pareto_point=pareto_point,
        labels=labels,
        title=method + " " + metric,
        fig=fig,
        ax=ax,
        x_label="# ensemble_memebers",
        y_label=y_label,
        colors=[colors[i]],
        alpha=0.6,
    )
    pareto_point = False
save_path = figure_dir.joinpath(
    f"uci-gap_ensemble-sizes_{experiment_name}_{method}_{metric}.pdf"
)
# fig.savefig(save_path, bbox_inches="tight")
