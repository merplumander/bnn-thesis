# %load_ext autoreload
# %autoreload 2
import copy
import json
from collections import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from core.plotting_utils import plot_uci_single_benchmark

figure_dir = "figures/temp"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)


color_mapping = {
    "VI": 0,
    "Map": 1,
    "Last Layer Bayesian": 2,
    "Ensemble": 3,
    "LLB Ensemble": 4,
    "VI-Flat-Prior": 5,
    "Ensemble MM": 6,
    "LLB Ensemble MM": 7,
}
colors = sns.color_palette()
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

# %% markdown
# # Fixed Epochs RMSEs

# %%
experiment_name = "epochs-40_one-hidden-layer"
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]


fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_data/{dataset}/results/{experiment_name}.json")
    results = json.loads(experiment_path.read_text())
    rmses = [results[model]["RMSEs"] for model in models]
    y_label = None
    if i == 0:
        y_label = "RMSE"
    if i == len(datasets) - 1:
        legend = True
    plot_uci_single_benchmark(
        rmses,
        labels=labels,
        ax_title=dataset,
        y_label=y_label,
        legend=legend,
        legend_kwargs=legend_kwargs,
        colors=[colors[color_mapping[method]] for method in labels],
        fig=fig,
        ax=axes[i],
    )
fig.tight_layout()
save_path = figure_dir.joinpath(f"{experiment_name}_rmses.pdf")
fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Fixed Epochs NLLs

# %%
experiment_name = "epochs-40_one-hidden-layer"
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]

fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_data/{dataset}/results/{experiment_name}.json")
    results = json.loads(experiment_path.read_text())
    nlls = [results[model]["NLLs"] for model in models]
    y_label = None
    if i == 0:
        y_label = "Negative Log Likelihood"
    if i == len(datasets) - 1:
        legend = True
    plot_uci_single_benchmark(
        nlls,
        labels=labels,
        ax_title=dataset,
        y_label=y_label,
        legend=legend,
        legend_kwargs=legend_kwargs,
        colors=[colors[color_mapping[method]] for method in labels],
        fig=fig,
        ax=axes[i],
    )
fig.tight_layout()
save_path = figure_dir.joinpath(f"{experiment_name}_nlls.pdf")
fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Early Stopping RMSEs


# %%
patience = 20
experiment_name = f"early-stop-patience-{patience}_one-hidden-layer"
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]

fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_data/{dataset}/results/{experiment_name}.json")
    results = json.loads(experiment_path.read_text())
    rmses = [results[model]["RMSEs"] for model in models]
    y_label = None
    if i == 0:
        y_label = "RMSE"
    if i == len(datasets) - 1:
        legend = True
    plot_uci_single_benchmark(
        rmses,
        labels=labels,
        ax_title=dataset,
        y_label=y_label,
        legend=legend,
        legend_kwargs=legend_kwargs,
        colors=[colors[color_mapping[method]] for method in labels],
        fig=fig,
        ax=axes[i],
    )
fig.tight_layout()
save_path = figure_dir.joinpath(f"{experiment_name}_rmses.pdf")
fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Early Stopping NLLs

# %%
patience = 20
experiment_name = f"early-stop-patience-{patience}_one-hidden-layer"
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]

fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_data/{dataset}/results/{experiment_name}.json")
    results = json.loads(experiment_path.read_text())
    nlls = [results[model]["NLLs"] for model in models]
    y_label = None
    if i == 0:
        y_label = "Negative Log Likelihood"
    if i == len(datasets) - 1:
        legend = True
    plot_uci_single_benchmark(
        nlls,
        labels=labels,
        ax_title=dataset,
        y_label=y_label,
        legend=legend,
        legend_kwargs=legend_kwargs,
        colors=[colors[color_mapping[method]] for method in labels],
        fig=fig,
        ax=axes[i],
    )
fig.tight_layout()
save_path = figure_dir.joinpath(f"{experiment_name}_nlls.pdf")
fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Moment matched versus Mixture prediction

# Cocnlusion: In terms of log likelihood it does not make a big difference whether to
# use the predictive mixture distribution or the moment matched version
# (ensemble size = 5). Perhaps the moment matched version is slightly superior.

# %%
patience = 20
experiment_name = f"early-stop-patience-{patience}_one-hidden-layer"
# experiment_name = "epochs-40_one-hidden-layer"
models = ["Ensemble", "LLB Ensemble"]

fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_data/{dataset}/results/{experiment_name}.json")
    results = json.loads(experiment_path.read_text())
    ensemble_mixture_nlls, llb_ensemble_mixture_nlls = [
        results[model]["NLLs"] for model in models
    ]
    ensemble_mm_nlls, llb_ensemble_mm_nlls = [
        results[model]["MM-NLLs"] for model in models
    ]
    y_label = None
    if i == 0:
        y_label = "Negative Log Likelihood"
    if i == len(datasets) - 1:
        legend = True
    plot_uci_single_benchmark(
        [
            ensemble_mixture_nlls,
            ensemble_mm_nlls,
            llb_ensemble_mixture_nlls,
            llb_ensemble_mm_nlls,
        ],
        labels=[
            "Ensemble Mixture ",
            "Ensemble MM",
            "LLB Ensemble Mixture",
            "LLB Ensemble MM",
        ],
        ax_title=dataset,
        y_label=y_label,
        legend=legend,
        legend_kwargs=legend_kwargs,
        colors=[
            colors[color_mapping[method]]
            for method in ["Ensemble", "Ensemble MM", "LLB Ensemble", "LLB Ensemble MM"]
        ],
        fig=fig,
        ax=axes[i],
    )
fig.tight_layout()


# %% markdown
# # VI Standard Normal Prior Vs. Flat Prior

# Conclusion: The difference between the VI models with different priors is small but
# generally the model with the standard normal distribution as prior is slightly
# superior (both in terms of RMSE and NLL). Also generally, the VI model is quite a lot
# worse than the MAP model. Even in terms of NLL. Why is that? (Early Stopping with
# patience 10 and 20).

# %%
patience = 20
experiment_name = f"early-stop-patience-{patience}_one-hidden-layer"
models = ["VI-Prior", "VI-Flat-Prior", "Map"]
labels = ["VI", "VI-Flat-Prior", "Map"]
fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_data/{dataset}/results/{experiment_name}.json")
    results = json.loads(experiment_path.read_text())
    nlls = [results[model]["NLLs"] for model in models]
    y_label = None
    if i == 0:
        y_label = "Negative Log Likelihood"
    if i == len(datasets) - 1:
        legend = True
    plot_uci_single_benchmark(
        nlls,
        labels=labels,
        ax_title=dataset,
        y_label=y_label,
        legend=legend,
        legend_kwargs=legend_kwargs,
        colors=[colors[color_mapping[method]] for method in labels],
        fig=fig,
        ax=axes[i],
    )
fig.tight_layout()

# %% markdown
# # Training Times

# Conclusion: Adding the Bayesian linear regression on top adds less than 1% of the MAP
# training time on top. In fact it is more like 0.3%.

# %%
patience = 20
experiment_name = f"early-stop-patience-{patience}_one-hidden-layer"
models = ["Map", "Last Layer Bayesian"]
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_data/{dataset}/results/{experiment_name}.json")
    results = json.loads(experiment_path.read_text())
    training_times = [results[model]["fit_times"] for model in models]
    map_avg_time, llb_avg_additional_time = np.mean(training_times, axis=1)
    # print(f"A MAP network trained for an average of {map_avg_time} seconds.")
    # print(f"It took {llb_avg_additional_time} additional seconds to add the bayesian linear regression on top.")
    print(
        f"The training of the bayesian linear regression took {llb_avg_additional_time / map_avg_time} of the time of the training of the map network."
    )
