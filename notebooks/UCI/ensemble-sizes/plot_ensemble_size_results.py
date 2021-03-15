# %load_ext autoreload
# %autoreload 2
import copy
import json
from collections import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib as tpl
import yaml
from brokenaxes import brokenaxes

from core.evaluation_utils import (
    get_y_normalization_scales,
    normalize_ll,
    normalize_rmse,
)
from core.plotting_utils import (
    plot_uci_ensemble_size_benchmark,
    plot_uci_single_benchmark,
)

figure_dir = "figures/ensemble_size_convergence"
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
gap_data = True
n_networks = 50
n_hidden_layers = 2
hidden_layers_string = (
    "two-hidden-layers" if n_hidden_layers == 2 else "one-hidden-layer"
)

data_dir = "uci_gap_data" if gap_data else "uci_data"
gap_string = "-gap" if gap_data else ""
experiment_name = f"uci{gap_string}_ensemble-size-convergence_{hidden_layers_string}"

# %% markdown


# %%
higher_is_better = True

_net_sizes = np.arange(n_networks) + 1.0
_net_sizes = _net_sizes[np.logical_or(_net_sizes <= 20, _net_sizes % 5 == 0)]
thick_linewidth = 4
thin_linewidth = 2
for dataset in datasets[3:]:
    experiment_path = Path(
        f"{data_dir}/{dataset}/results/ensemble_sizes/{experiment_name}.json"
    )

    y_normalization_scales = get_y_normalization_scales(dataset, gap_data=gap_data).T
    y_normalization_scales.shape

    results = json.loads(experiment_path.read_text())

    e_rmses = np.array(
        results["Ensemble"]["RMSEs"]
    )  # n_ensemble_members, n_splits, n_different_ensembles
    e_rmses = normalize_rmse(e_rmses, y_normalization_scales)
    e_nlls = np.array(results["Ensemble"]["NLLs"])
    e_nlls = normalize_ll(e_nlls, y_normalization_scales)

    # e_mm_nlls = results["Ensemble"]["MM-NLLs"]
    llbe_rmses = np.array(results["LLB Ensemble"]["RMSEs"])
    llbe_rmses = normalize_rmse(llbe_rmses, y_normalization_scales)
    llbe_nlls = np.array(results["LLB Ensemble"]["NLLs"])
    llbe_nlls = normalize_ll(llbe_nlls, y_normalization_scales)
    # llbe_mm_nlls = results["LLB Ensemble"]["MM-NLLs"]
    labels = ["Ensemble", "LLB Ensemble"]
    if higher_is_better:
        e_rmses = -e_rmses
        e_nlls = -e_nlls
        llbe_rmses = -llbe_rmses
        llbe_nlls = -llbe_nlls
    fig, ax = plt.subplots(figsize=(15, 5))
    plot_uci_ensemble_size_benchmark(
        [e_nlls],  # llbe_nlls],
        x=_net_sizes,
        labels=["Ensemble"],
        fig=fig,
        ax=ax,
        linewidth=thick_linewidth,
        colors=[colors[color_mapping["Ensemble"]]],
    )
    for i in range(e_nlls.shape[1]):
        plot_uci_ensemble_size_benchmark(
            [e_nlls[:, i : i + 1]],
            errorbars=False,
            x=_net_sizes,
            labels=[None],
            fig=fig,
            ax=ax,
            linewidth=thin_linewidth,
            colors=[colors[color_mapping["Ensemble"]]],
            alpha=0.2,
        )

    # and LLB Ensemble
    plot_uci_ensemble_size_benchmark(
        [llbe_nlls],  # llbe_nlls],
        x=_net_sizes,
        labels=["LLB Ensemble"],
        fig=fig,
        ax=ax,
        linewidth=thick_linewidth,
        colors=[colors[color_mapping["LLB Ensemble"]]],
    )
    for i in range(llbe_nlls.shape[1]):
        plot_uci_ensemble_size_benchmark(
            [llbe_nlls[:, i : i + 1]],
            errorbars=False,
            x=_net_sizes,
            labels=[None],
            fig=fig,
            ax=ax,
            linewidth=thin_linewidth,
            colors=[colors[color_mapping["LLB Ensemble"]]],
            alpha=0.2,
        )
    ax.set_title(f"{dataset}")
    ax.set_xlabel("# ensemble_memebers")
    ax.set_ylabel("Negative Log Likelihood")
    # if dataset == "yacht":
    #     ax.set_ylim([0, 2.5])


# %%
_net_sizes = np.arange(n_networks) + 1.0
_net_sizes = _net_sizes[np.logical_or(_net_sizes <= 20, _net_sizes % 5 == 0)]


figsize = (4, 3.8) if gap_data else (4, 3.6)
fig, ax = plt.subplots(figsize=figsize)  # (4, 3.6)
normalized_space = True
higher_is_better = True
normalization = True
optimum_is_full_ensemble = True
plot_single_lines = False
save = True
method = "LLB Ensemble"
metric = "NLLs"
method_string = "MAP Ensemble" if method == "Ensemble" else "NL Ensemble"
metric_string = "LL" if metric == "NLLs" else "RMSE"
dataset_results = []
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
    if normalized_space:
        if metric == "NLLs":
            results = normalize_ll(results, y_normalization_scales)
        elif metric == "RMSEs":
            results = normalize_rmse(results, y_normalization_scales)
        else:
            raise ValueError()
    if higher_is_better:
        results = -results
    dataset_results.append(results)

    labels = [f"{dataset}"]
    y_label = "performance fraction"
    plot_uci_ensemble_size_benchmark(
        [results],
        normalization=normalization,
        optimum_is_full_ensemble=optimum_is_full_ensemble,
        x=_net_sizes,
        x_add_jitter=0.0 * i,
        errorbars=False,
        labels=labels,
        fig=fig,
        ax=ax,
        x_label="ensemble size",
        y_label=y_label,
        colors=[colors[i]],
        linewidth=1.8,
        alpha=0.7,
    )

    # # individual split lines
    if plot_single_lines:
        size_means = np.mean(results, axis=1)
        if normalization:
            if optimum_is_full_ensemble:
                min = size_means[0]
                max = size_means[-1]
            else:
                min = np.min(size_means)
                max = np.max(size_means)
            results = (results - min) / (max - min)
        for i_in_dataset in range(results.shape[1]):
            plot_uci_ensemble_size_benchmark(
                [results[:, i_in_dataset : i_in_dataset + 1]],
                errorbars=False,
                x=_net_sizes,
                labels=[None],
                fig=fig,
                ax=ax,
                colors=[colors[i]],
                linewidth=1,
                linestyle="--",
                alpha=0.2,
            )
size_indices = np.arange(len(_net_sizes))
actual_sizes = _net_sizes
# size_indices = [1, 4, 9, 19, -1]  # is equivalent to ensemble sizes of 2, 5, 10, 20, and 50
# actual_sizes = [2, 5, 10, 20, 50]
result_means = np.zeros((len(dataset_results), len(size_indices)))
for i, results in enumerate(dataset_results):
    size_means = np.mean(results, axis=1)
    if optimum_is_full_ensemble:
        min = size_means[0]
        max = size_means[-1]
    else:
        min = np.min(size_means)
        max = np.max(size_means)
    size_means = (size_means - min) / (max - min)
    result_means[i] = size_means[size_indices]

mean_points = np.mean(result_means, axis=0)
ax.plot(actual_sizes, mean_points, c="k", alpha=0.9, label="average", linewidth=3)
line_point_indices = [
    1,
    4,
    9,
    19,
    -1,
]  # is equivalent to ensemble sizes of 2, 5, 10, 20, and 50
ax.scatter(actual_sizes[line_point_indices], mean_points[line_point_indices], c="k")
ax.hlines(
    mean_points[line_point_indices],
    np.zeros_like(len(line_point_indices)),
    actual_sizes[line_point_indices],
    color="k",
    linestyles="dashed",
    alpha=0.3,
)
ax.set_title(f"{method_string}")  # Convergence {metric_string}")  # UCI{gap_string}
ax.set_xticks([1, 5, 10, 20, 50])
ax.set_xticklabels([1, 5, 10, 20, 50])
if n_hidden_layers == 1:
    ax.set_ylim([-0.05, 1.12])
else:
    ax.set_ylim([-0.05, 1.1])
if gap_data:
    ax.set_ylim([-0.1, 1.15])
ax.legend()
# ax.set_yticks([0.5, 0.8, 0.95, 1.0])
# ax.set_yticklabels([0.5, 0.8, 0.95, 1.0])
if plot_single_lines:
    ax.set_ylim([-1.0, 2.5])

if save:
    save_path = figure_dir.joinpath(
        f"{experiment_name}_{method}_{metric}.pdf".replace(" ", "-")
    )
    fig.savefig(save_path, bbox_inches="tight")
    tex = tpl.get_tikz_code(
        axis_width="\\figwidth",  # we want LaTeX to take care of the width
        axis_height="\\figheight",  # we want LaTeX to take care of the height
        # we want the plot to look *exactly* like here (e.g. axis limits, axis ticks, etc.)
        # strict=True,
    )
    with save_path.with_suffix(".tex").open("w", encoding="utf-8") as f:
        f.write(tex)


# %%

_net_sizes = np.arange(n_networks) + 1.0
_net_sizes = _net_sizes[np.logical_or(_net_sizes <= 20, _net_sizes % 5 == 0)]

fig, ax = plt.subplots(figsize=(18, 10))
higher_is_better = True
optimum_is_full_ensemble = True
plot_single_lines = False
method = "Ensemble"
metric = "NLLs"
dataset_results = []
for i, dataset in enumerate(datasets):
    experiment_path = Path(
        f"{data_dir}/{dataset}/results/ensemble_sizes/{experiment_name}.json"
    )

    y_normalization_scales = get_y_normalization_scales(dataset, gap_data=gap_data).T
    y_normalization_scales.shape

    results = json.loads(experiment_path.read_text())
    results = np.array(
        results[method][metric]
    )  # n_ensemble_members, n_splits, n_different_ensembles
    if higher_is_better:
        results = -results
    results.shape
    if optimum_is_full_ensemble:
        min = results[0]
        max = results[-1]
    else:
        min = np.min(results, axis=0)
        max = np.max(results, axis=0)

    results = (results - min) / (max - min)
    # size_stds = size_stds / (max - min)
    dataset_results.append(results)

    labels = [f"{dataset}"]
    y_label = "RMSE" if metric == "RMSEs" else "log likelihood"
    plot_uci_ensemble_size_benchmark(
        [results],
        x=_net_sizes,
        x_add_jitter=0.0 * i,
        errorbars=False,
        labels=labels,
        title=method + " " + metric,
        fig=fig,
        ax=ax,
        x_label="# ensemble_memebers",
        y_label=y_label,
        colors=[colors[i]],
        alpha=1,
    )

    # individual split lines
    if plot_single_lines:
        for i_in_dataset in range(results.shape[1]):
            plot_uci_ensemble_size_benchmark(
                [results[:, i_in_dataset : i_in_dataset + 1]],
                errorbars=False,
                x=_net_sizes,
                labels=[None],
                fig=fig,
                ax=ax,
                colors=[colors[i]],
                linewidth=1,
                linestyle="--",
                alpha=0.4,
            )
size_indices = [1, 4, 9, 19, -1]  # is equivalent to ensemble sizes of 5, 10, 20, and 50
actual_sizes = [2, 5, 10, 20, 50]
result_means = np.zeros((9, len(size_indices)))
for i, results in enumerate(dataset_results):
    size_means = np.mean(results, axis=1)
    # min = np.min(size_means)
    # max = np.max(size_means)
    # size_means = (size_means - min) / (max - min)
    result_means[i] = size_means[size_indices]

mean_points = np.mean(result_means, axis=0)
ax.scatter(actual_sizes, mean_points, c="k")
ax.hlines(
    mean_points,
    np.zeros_like(mean_points),
    actual_sizes,
    color="k",
    linestyles="dashed",
    alpha=0.3,
)
if plot_single_lines and optimum_is_full_ensemble:
    ax.set_ylim([-0.5, 1.8])


# %%
_net_sizes = np.arange(n_networks) + 1.0
_net_sizes = _net_sizes[np.logical_or(_net_sizes <= 20, _net_sizes % 5 == 0)]

# wspace = 0.02
# fig = plt.figure(figsize=(10, 6))
# ax = brokenaxes(xlims=((0, 21), (29, 31), (39, 41), (49, 51)), wspace=wspace, despine=False)
fig, ax = plt.subplots(figsize=(7, 4))

normalized_space = True
higher_is_better = True
normalization = True
optimum_is_full_ensemble = True
x_offset = -0.35
y_offset = 0.02

metric = "RMSEs"
y_label = "Performance Fraction"
dataset_results = []
for i_method, method in enumerate(["LLB Ensemble", "Ensemble"]):
    for i, dataset in enumerate(datasets[:]):
        experiment_path = Path(
            f"{data_dir}/{dataset}/results/ensemble_sizes/{experiment_name}.json"
        )

        y_normalization_scales = get_y_normalization_scales(
            dataset, gap_data=gap_data
        ).T
        y_normalization_scales.shape

        results = json.loads(experiment_path.read_text())
        results = np.array(
            results[method][metric]
        )  # n_ensemble_members, n_splits, n_different_ensembles
        if normalized_space:
            if metric == "NLLs":
                results = normalize_ll(results, y_normalization_scales)
            elif metric == "RMSEs":
                results = normalize_rmse(results, y_normalization_scales)
            else:
                raise ValueError()
        if higher_is_better:
            results = -results
        dataset_results.append(results)

        labels = [f"{dataset}"]
        # plot_uci_ensemble_size_benchmark(
        #     [results + i_method * y_offset],
        #     normalization=normalization,
        #     optimum_is_full_ensemble=optimum_is_full_ensemble,
        #     x=_net_sizes,
        #     x_add_jitter=i_method * x_offset,
        #     errorbars=False,
        #
        #     labels=[None],
        #     fig=fig,
        #     ax=ax,
        #
        #     colors=[colors[color_mapping[method]]],
        #     linewidth=None,
        #     alpha=0.3,
        #     legend=False,
        # )
    size_indices = np.arange(len(_net_sizes))
    actual_sizes = _net_sizes
    result_means = np.zeros((len(dataset_results), len(size_indices)))
    for i, results in enumerate(dataset_results):
        size_means = np.mean(results, axis=1)
        if optimum_is_full_ensemble:
            min = size_means[0]
            max = size_means[-1]
        else:
            min = np.min(size_means)
            max = np.max(size_means)
        size_means = (size_means - min) / (max - min)
        result_means[i] = size_means[size_indices]

    mean_points = np.mean(result_means, axis=0)
    std_error_points = np.std(result_means, axis=0) / np.sqrt(len(datasets))
    ax.plot(
        actual_sizes + i_method * x_offset,
        mean_points + i_method * y_offset,
        c=colors[color_mapping[method]],
        alpha=1,
        label=f"{method}",
    )
    ax.errorbar(
        actual_sizes + i_method * x_offset,
        mean_points + i_method * y_offset,
        std_error_points,
        c=colors[color_mapping[method]],
        alpha=1,
        # label=f"{method}",
    )
    print(f"{method}: {np.around(mean_points[line_point_indices], decimals=2)}")
ax.set_xticks([1, 5, 10, 20, 30, 40, 50])
ax.set_xticklabels([1, 5, 10, 20, 30, 40, 50])
ax.set_xlabel("# ensemble_memebers")
ax.set_ylabel(y_label)
ax.set_title(f"UCI{gap_string} Ensemble Convergence Behaviour")
ax.legend(**{"loc": "lower right"})

line_point_indices = [
    1,
    4,
    9,
    19,
    -1,
]  # is equivalent to ensemble sizes of 2, 5, 10, 20, and 50
# ax.scatter(
#     actual_sizes[line_point_indices],
#     mean_points[line_point_indices],
#     color="k",
#     alpha=1,
# )
# ax.hlines(
#     mean_points[line_point_indices],
#     np.zeros_like(len(line_point_indices)),
#     actual_sizes[line_point_indices],
#     color="k",
#     linestyles="dashed",
#     alpha=0.3,
# )


save_path = figure_dir.joinpath(f"{experiment_name}_2_{metric}.pdf")
# fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Difference of shuffling data:

# %%
gap_data = False
n_networks = 50
n_hidden_layers = 2
hidden_layers_string = (
    "two-hidden-layers" if n_hidden_layers == 2 else "one-hidden-layer"
)

data_dir = "uci_gap_data" if gap_data else "uci_data"
gap_string = "-gap" if gap_data else ""
experiment_name = f"uci{gap_string}_ensemble-size-convergence_{hidden_layers_string}"

higher_is_better = True

_net_sizes = np.arange(n_networks) + 1.0
_net_sizes = _net_sizes[np.logical_or(_net_sizes <= 20, _net_sizes % 5 == 0)]
thick_linewidth = 4
thin_linewidth = 2
dataset = "yacht"
fig, ax = plt.subplots(figsize=(15, 5))
for shuffle_string, experiment_name in zip(
    ["", "shuffled data"],
    [
        f"uci{gap_string}_ensemble-size-convergence_{hidden_layers_string}",
        f"uci{gap_string}_ensemble-size-convergence_{hidden_layers_string}_shuffle-train-data",
    ],
):
    experiment_path = Path(
        f"{data_dir}/{dataset}/results/ensemble_sizes/{experiment_name}.json"
    )

    y_normalization_scales = get_y_normalization_scales(dataset, gap_data=gap_data).T
    y_normalization_scales.shape

    results = json.loads(experiment_path.read_text())

    e_rmses = np.array(
        results["Ensemble"]["RMSEs"]
    )  # n_ensemble_members, n_splits, n_different_ensembles
    e_rmses = normalize_rmse(e_rmses, y_normalization_scales)
    e_nlls = np.array(results["Ensemble"]["NLLs"])
    e_nlls = normalize_ll(e_nlls, y_normalization_scales)

    # e_mm_nlls = results["Ensemble"]["MM-NLLs"]
    llbe_rmses = np.array(results["LLB Ensemble"]["RMSEs"])
    llbe_rmses = normalize_rmse(llbe_rmses, y_normalization_scales)
    llbe_nlls = np.array(results["LLB Ensemble"]["NLLs"])
    llbe_nlls = normalize_ll(llbe_nlls, y_normalization_scales)
    # llbe_mm_nlls = results["LLB Ensemble"]["MM-NLLs"]
    labels = ["Ensemble", "LLB Ensemble"]
    if higher_is_better:
        e_rmses = -e_rmses
        e_nlls = -e_nlls
        llbe_rmses = -llbe_rmses
        llbe_nlls = -llbe_nlls

    plot_uci_ensemble_size_benchmark(
        [e_nlls],  # llbe_nlls],
        x=_net_sizes,
        labels=[f"Ensemble {shuffle_string}"],
        fig=fig,
        ax=ax,
        linewidth=thick_linewidth,
        # colors=[colors[color_mapping["Ensemble"]]],
    )
    # for i in range(e_nlls.shape[1]):
    #     plot_uci_ensemble_size_benchmark(
    #         [e_nlls[:, i : i + 1]],
    #         errorbars=False,
    #         x=_net_sizes,
    #         labels=[None],
    #         fig=fig,
    #         ax=ax,
    #         linewidth=thin_linewidth,
    #         #colors=[colors[color_mapping["Ensemble"]]],
    #         alpha=0.2,
    #     )

    # and LLB Ensemble
    plot_uci_ensemble_size_benchmark(
        [llbe_nlls],  # llbe_nlls],
        x=_net_sizes,
        labels=[f"LLB Ensemble {shuffle_string}"],
        fig=fig,
        ax=ax,
        linewidth=thick_linewidth,
        # colors=[colors[color_mapping["LLB Ensemble"]]],
    )
    # for i in range(llbe_nlls.shape[1]):
    #     plot_uci_ensemble_size_benchmark(
    #         [llbe_nlls[:, i : i + 1]],
    #         errorbars=False,
    #         x=_net_sizes,
    #         labels=[None],
    #         fig=fig,
    #         ax=ax,
    #         linewidth=thin_linewidth,
    #         #colors=[colors[color_mapping["LLB Ensemble"]]],
    #         alpha=0.2,
    #     )
    ax.set_title(f"{dataset}")
    ax.set_xlabel("# ensemble_memebers")
    ax.set_ylabel("Negative Log Likelihood")


# %% markdown
# # Make Tables


# %%
_net_sizes = np.arange(n_networks) + 1.0
_net_sizes = _net_sizes[np.logical_or(_net_sizes <= 20, _net_sizes % 5 == 0)]

method = "Ensemble"
metric = "NLLs"
higher_is_better = True
dataset_results = []

gap_data = True
n_networks = 50
n_hidden_layers = 1
hidden_layers_string = (
    "two-hidden-layers" if n_hidden_layers == 2 else "one-hidden-layer"
)


if gap_data:
    data_dir = "uci_gap_data"
    experiment_name = f"uci-gap_ensemble-size-convergence_{hidden_layers_string}"
else:
    data_dir = "uci_data"
    experiment_name = f"uci_ensemble-size-convergence_{hidden_layers_string}"

for gap_data in [True, False]:

    for i, dataset in enumerate(datasets[:]):
        experiment_path = Path(
            f"{data_dir}/{dataset}/results/ensemble_sizes/{experiment_name}.json"
        )
        results = json.loads(experiment_path.read_text())
        results = np.array(
            results[method][metric]
        )  # n_ensemble_members, n_splits, n_different_ensembles
        if higher_is_better:
            results = -results
        size_means = np.mean(results, axis=1)
        min = np.min(size_means)
        max = np.max(size_means)
        size_means = (size_means - min) / (max - min)
        dataset_results.append(results)


data = {"col_1": [3, 2, 1, 0], "col_2": ["a", "b", "c", "d"]}
pd.DataFrame.from_dict(data)  # .to_latex()
print(pd.DataFrame.from_dict(data).to_latex())

dic = {
    ("A", "a"): [1, 2, 3, 4, 5],
    ("A", "b"): [6, 7, 8, 9, 1],
    ("B", "a"): [2, 3, 4, 5, 6],
    ("B", "b"): [7, 8, 9, 1, 2],
}

pd.DataFrame.from_dict(dic, orient="columns").T

# %%
for i, dataset in enumerate(datasets[-1:]):
    experiment_path = Path(
        f"{data_dir}/{dataset}/results/ensemble_sizes/{experiment_name}.json"
    )
    results = json.loads(experiment_path.read_text())
    results = np.array(
        results[method][metric]
    )  # n_ensemble_members, n_splits, n_different_ensembles
    results.shape
    print(results)
    np.argmax(results, axis=0)
    np.argmin(results, axis=0)
    dataset_results.append(results)
