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

from core.plotting_utils import plot_uci_single_benchmark

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

# %% markdown
# # Fixed Epochs RMSEs

# %%
experiment_name = "epochs-40_one-hidden-layer"
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble MM", "LLB Ensemble MM"]


fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_gap_data/{dataset}/results/{experiment_name}.json")
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
save_path = figure_dir.joinpath(f"uci_gap_{experiment_name}_rmses.pdf")
fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Fixed Epochs NLLs

# %%
experiment_name = "epochs-40_one-hidden-layer"
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble MM", "LLB Ensemble MM"]

fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_gap_data/{dataset}/results/{experiment_name}.json")
    results = json.loads(experiment_path.read_text())
    nlls = [results[model]["NLLs"] for model in models[0:4]]
    nlls += [results[model]["MM-NLLs"] for model in models[4:]]
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
save_path = figure_dir.joinpath(f"uci_gap_{experiment_name}_nlls.pdf")
fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Early Stopping RMSEs


# %%
patience = 20
experiment_name = f"early-stop-patience-{patience}_one-hidden-layer"
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble MM", "LLB Ensemble MM"]

fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_gap_data/{dataset}/results/{experiment_name}.json")
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
save_path = figure_dir.joinpath(f"uci_gap_{experiment_name}_rmses.pdf")
fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Early Stopping NLLs

# %%
patience = 20
experiment_name = f"early-stop-patience-{patience}_one-hidden-layer"
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble MM", "LLB Ensemble MM"]

fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_gap_data/{dataset}/results/{experiment_name}.json")
    results = json.loads(experiment_path.read_text())
    nlls = [results[model]["NLLs"] for model in models[0:4]]
    nlls += [results[model]["MM-NLLs"] for model in models[4:]]
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
save_path = figure_dir.joinpath(f"uci_gap_{experiment_name}_nlls.pdf")
fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # L2 Convergence RMSEs


# %%
weight_prior_scale = 1
patience = 20
experiment_name = (
    f"l2-reg-prior-scale-{weight_prior_scale}-patience-{patience}_one-hidden-layer"
)
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble MM", "LLB Ensemble MM"]

fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_gap_data/{dataset}/results/{experiment_name}.json")
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
save_path = figure_dir.joinpath(f"uci_gap_{experiment_name}_rmses.pdf")
fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Early Stopping NLLs

# %%
weight_prior_scale = 1
patience = 20
experiment_name = (
    f"l2-reg-prior-scale-{weight_prior_scale}-patience-{patience}_one-hidden-layer"
)
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble MM", "LLB Ensemble MM"]

fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_gap_data/{dataset}/results/{experiment_name}.json")
    results = json.loads(experiment_path.read_text())
    nlls = [results[model]["NLLs"] for model in models[0:4]]
    nlls += [results[model]["MM-NLLs"] for model in models[4:]]
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
save_path = figure_dir.joinpath(f"uci_gap_{experiment_name}_nlls.pdf")
fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Moment matched versus Mixture prediction

# Cocnlusion: Here in the UCI Gap case it is clearer, that the MM prediction is quite
# superior to the mixture. It never hurts performance and sometimes helps a lot.

# %%
weight_prior_scale = 1
patience = 20
experiment_name = (
    f"l2-reg-prior-scale-{weight_prior_scale}-patience-{patience}_one-hidden-layer"
)
# experiment_name = f"early-stop-patience-{patience}_one-hidden-layer"
# experiment_name = "epochs-40_one-hidden-layer"
models = ["Ensemble", "LLB Ensemble"]

fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_gap_data/{dataset}/results/{experiment_name}.json")
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
save_path = figure_dir.joinpath(f"uci_gap_{experiment_name}_mm_vs_mixture.pdf")
fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # VI Standard Normal Prior Vs. Flat Prior

# Conclusion: Also in the Gap case, the difference between the VI models with different
# priors is small but
# generally the model with the standard normal distribution as prior is slightly
# superior. Here however, it shows the strngths of
# epistmic uncertainty in that VI is often quite a bit better than MAP.
# this was the other way around for the non_gap case.

# %%
patience = 20
experiment_name = f"early-stop-patience-{patience}_one-hidden-layer"
models = ["VI-Prior", "VI-Flat-Prior", "Map"]
labels = ["VI", "VI-Flat-Prior", "Map"]
fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_gap_data/{dataset}/results/{experiment_name}.json")
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
save_path = figure_dir.joinpath(
    f"uci_gap_{experiment_name}_vi_flat_vs_informative_prior.pdf"
)
fig.savefig(save_path, bbox_inches="tight")
