# %load_ext autoreload
# %autoreload 2
import copy
import json
from collections import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tikzplotlib as tpl
import yaml

from core.evaluation_utils import (
    get_y_normalization_scales,
    normalize_ll,
    normalize_rmse,
)
from core.plotting_utils import plot_uci_single_benchmark

figure_dir = "figures/uci-benchmark"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)


color_formatter = "deep"
# color_formatter = "colorblind"
colors = sns.color_palette(color_formatter)
colors
with open("config/uci-color-config.yaml") as f:
    color_mapping = yaml.full_load(f)
colors[1]
# colors[color_mapping["LLB"]] = sns.blend_palette(
#     [colors[color_mapping["LLB"]], colors[color_mapping["LLB Ensemble"]]], n_colors=5
# )[2]
# colors[color_mapping["MAP"]] = sns.blend_palette(
#     [colors[color_mapping["MAP"]], colors[color_mapping["Ensemble"]]], n_colors=5
# )[2]
# sns.blend_palette(
#     [colors[color_mapping["MAP"]], colors[color_mapping["Ensemble"]]], n_colors=5
# )[2]
# colors
viridis = sns.color_palette("viridis", 3 + 6)
viridis
magma = sns.color_palette("magma", 3 + 6)
magma
figsize = (18, 6)
legend_kwargs = {
    "bbox_to_anchor": (1.36, 1.01),
    "loc": "upper left",
}  # "bbox_to_anchor": (1.4, 1)

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


def plot_uci_results(
    vi_result,
    map_results,
    llb_results,
    map_ensemble_results,
    llb_ensemble_results,
    ensemble_sizes,
    x_space_between_network_and_ensemble=0.4,
    x_space_for_ensembles=1.0,
    x_space_to_next_method=0.8,
    map_marker="^",
    llb_marker="^",
    ensemble_markersize=None,
    map_ensemble_marker="o",
    llb_ensemble_marker="o",
    ensemble_alpha=1,
    color_style="plain",
    title=None,
    ax_title=None,
    y_label=None,
    y_max_ticks=None,
    legend=False,
    legend_kwargs={},
    fig=None,
    ax=None,
):

    n_different_map = map_ensemble_results.shape[0]

    if color_style == "plain":
        base_colors = sns.color_palette("deep")
        vi_color = vi_color = base_colors[2]
        map_color = base_colors[0]
        map_ensemble_colors = [base_colors[0]] * n_different_map
        llb_color = base_colors[1]
        llb_ensemble_colors = [base_colors[1]] * n_different_map

        map_ensemble_labels = [None] * (n_different_map - 1) + ["MAP E."]
        llb_ensemble_labels = [None] * (n_different_map - 1) + ["LLB E."]
    elif color_style == "blend":
        color_formatter = "deep"  # "colorblind"
        base_colors = sns.color_palette(color_formatter)
        map_blend = sns.blend_palette([base_colors[0], sns.color_palette()[3]], 6)
        map_blend
        llb_blend = sns.blend_palette(
            [sns.color_palette("colorblind")[1], sns.color_palette()[3]], 6
        )
        llb_blend

        vi_color = vi_color = base_colors[2]
        map_color = base_colors[0]
        map_ensemble_colors = [map_blend[2]] * n_different_map
        llb_color = base_colors[1]
        llb_ensemble_colors = [llb_blend[2]] * n_different_map

        map_ensemble_labels = [None] * (n_different_map - 1) + ["MAP E."]
        llb_ensemble_labels = [None] * (n_different_map - 1) + ["LLB E."]
    elif color_style == "blend_changing":
        color_formatter = "deep"  # "colorblind"
        base_colors = sns.color_palette(color_formatter)
        map_color = base_colors[0]
        map_blend = sns.blend_palette([map_color, sns.color_palette()[3]], 8)
        map_blend
        llb_color = sns.blend_palette(
            [sns.color_palette("colorblind")[8], sns.color_palette("colorblind")[1]], 6
        )[-3]
        llb_blend = sns.blend_palette([llb_color, sns.color_palette()[3]], 8)
        llb_blend

        vi_color = vi_color = base_colors[2]
        map_ensemble_colors = map_blend[2:]
        llb_ensemble_colors = llb_blend[2:]

        map_ensemble_labels = [f"MAP-{size}" for size in ensemble_sizes]
        llb_ensemble_labels = [f"NL-{size}" for size in ensemble_sizes]
    elif color_style == "viridis":
        indices = [2] + list(4 + np.arange(n_different_map))
        viridis = np.array(sns.color_palette("viridis_r", n_different_map + 6))[indices]
        magma = np.array(sns.color_palette("magma_r", n_different_map + 6))[indices]

        vi_color = sns.color_palette("colorblind")[1]
        map_color = viridis[0]
        map_ensemble_colors = viridis[1:]
        llb_color = magma[0]
        llb_ensemble_colors = magma[1:]

        map_ensemble_labels = [f"MAP-{size}" for size in ensemble_sizes]
        llb_ensemble_labels = [f"NL-{size}" for size in ensemble_sizes]
    else:
        raise NotImplementedError()

    vi_mean = np.mean(vi_result)
    vi_std_error = np.std(vi_result) / np.sqrt(vi_result.size)
    fig.suptitle(title, fontsize=15)
    ax.set_title(ax_title, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.tick_params(bottom=False, labelbottom=False)

    # VI
    ax.errorbar(0, vi_mean, vi_std_error, marker="s", label="VI", c=vi_color, ls="")

    # MAP
    map_means = np.mean(map_results, axis=1)
    map_std_errors = np.std(map_results, axis=1) / np.sqrt(map_results.shape[1])
    ax.errorbar(
        x_space_to_next_method,
        map_means,
        map_std_errors,
        marker=map_marker,
        label="MAP",
        c=map_color,
        ls="",
    )

    map_ensemble_means = np.mean(map_ensemble_results, axis=1)
    map_ensemble_std_errors = np.std(map_ensemble_results, axis=1) / np.sqrt(
        map_ensemble_results.shape[1]
    )
    map_ensemble_x = (
        x_space_to_next_method
        + x_space_between_network_and_ensemble
        + x_space_for_ensembles
        * np.arange(map_ensemble_results.shape[0])
        / map_ensemble_results.shape[0]
    )

    for x, mean, std_error, c, label in zip(
        map_ensemble_x,
        map_ensemble_means,
        map_ensemble_std_errors,
        map_ensemble_colors,
        map_ensemble_labels,
    ):
        ax.errorbar(
            x,
            mean,
            std_error,
            c=c,
            label=label,
            marker=map_ensemble_marker,
            markersize=ensemble_markersize,
            ls="",
        )

    # LLB
    llb_means = np.mean(llb_results, axis=1)
    llb_std_errors = np.std(llb_results, axis=1) / np.sqrt(llb_results.shape[1])
    ax.errorbar(
        2 * x_space_to_next_method
        + x_space_between_network_and_ensemble
        + x_space_for_ensembles,
        llb_means,
        llb_std_errors,
        marker=llb_marker,
        label="NL",
        c=llb_color,
        ls="",
    )

    llb_ensemble_means = np.mean(llb_ensemble_results, axis=1)
    llb_ensemble_std_errors = np.std(llb_ensemble_results, axis=1) / np.sqrt(
        llb_ensemble_results.shape[1]
    )
    llb_ensemble_x = (
        2 * x_space_to_next_method
        + 2 * x_space_between_network_and_ensemble
        + x_space_for_ensembles
        + x_space_for_ensembles
        * np.arange(llb_ensemble_results.shape[0])
        / llb_ensemble_results.shape[0]
    )

    for x, mean, std_error, c, label in zip(
        llb_ensemble_x,
        llb_ensemble_means,
        llb_ensemble_std_errors,
        llb_ensemble_colors,
        llb_ensemble_labels,
    ):
        ax.errorbar(
            x,
            mean,
            std_error,
            c=c,
            label=label,
            marker=llb_ensemble_marker,
            markersize=ensemble_markersize,
            ls="",
        )

    if legend:
        # get handles
        handles, labels = ax.get_legend_handles_labels()
        # remove the errorbars
        # handles = [h[0] for h in handles]
        # use them in the legend
        ax.legend(handles, labels, **legend_kwargs, fontsize=11)
    if y_max_ticks is not None:
        ax.yaxis.set_major_locator(plt.MaxNLocator(y_max_ticks))


# %%
gap_data = False
n_hidden_layers = 2
hidden_layers_string = (
    "two-hidden-layers" if n_hidden_layers == 2 else "one-hidden-layer"
)

data_dir = "uci_gap_data" if gap_data else "uci_data"
gap_string = "-gap" if gap_data else ""
experiment_name = f"uci{gap_string}_ensemble-size-convergence_{hidden_layers_string}"


n_networks = 50
_net_sizes = np.arange(n_networks) + 1.0
_net_sizes = _net_sizes[np.logical_or(_net_sizes <= 20, _net_sizes % 5 == 0)]
_size_indices = np.arange(26)
map_net_size_to_index = dict(zip(_net_sizes, _size_indices))

vi_experiment_name = f"uci_vi_{hidden_layers_string}"


# %%
ensemble_sizes = [5, 20, 50]
size_indices = [map_net_size_to_index[size] for size in ensemble_sizes]

metric = "RMSEs"
color_style = "blend_changing"
normalize_results = False

figsize = (8, 5.5)
# fig, (axes) = plt.subplots(2, int(np.ceil(len(datasets) / 2)), figsize=figsize)
# axes = axes.flatten()
# fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
fig = plt.figure(figsize=figsize)
axes = [fig.add_subplot(2, 5, i) for i in np.arange(len(datasets)) + 1]
legend = False
wspace = 0.6
if n_hidden_layers == 2:
    if gap_data:
        y_max_tickss = (
            [4, 4, 4, 4, 4, 4, 4, 4, 4]
            if metric == "NLLs"
            else [3, 3, 3, 2, 3, 4, 3, 4, 2]
        )
    else:
        y_max_tickss = (
            [4, 3, 4, 4, 5, 4, 4, 4, 4]
            if metric == "NLLs"
            else [3, 3, 3, 2, 4, 4, 3, 4, 2]
        )

else:
    y_max_tickss = (
        [4, 4, 4, 4, 5, 4, 4, 4, 4] if metric == "NLLs" else [3, 3, 3, 2, 4, 4, 3, 2, 2]
    )
for i, dataset, y_max_ticks in zip(range(len(datasets)), datasets, y_max_tickss):
    experiment_path = Path(
        f"{data_dir}/{dataset}/results/uci_vi_initial-unconstrained-scale-{-1}_{hidden_layers_string}.json"
    )
    results = json.loads(experiment_path.read_text())
    vi_results = np.array(results["VI"][metric])
    # print(f"{dataset}\nVI mean: {np.mean(vi_results)}")
    experiment_path = Path(
        f"{data_dir}/{dataset}/results/ensemble_sizes/{experiment_name}.json"
    )
    results = json.loads(experiment_path.read_text())
    map_results = np.array(results["Ensemble"][metric])  # n_ensemble_members, n_splits
    llb_results = np.array(results["LLB Ensemble"][metric])
    # map_means = np.mean(map_results, axis=1)
    # assert np.all(map_means[0] > map_means[1:])
    # llb_means = np.mean(llb_results, axis=1)
    # assert np.all(llb_means[0] > llb_means[1:])
    if normalize_results:
        y_normalization_scales = get_y_normalization_scales(
            dataset, gap_data=gap_data
        ).T
        if metric == "NLLs":
            log_scale = np.log(y_normalization_scales)
            print(f"Mean NLL Normalizing Addition {np.mean(-log_scale)}")
            vi_results = normalize_ll(vi_results, y_normalization_scales)
            map_results = normalize_ll(map_results, y_normalization_scales)
            llb_results = normalize_ll(llb_results, y_normalization_scales)
        elif metric == "RMSEs":
            vi_results = normalize_rmse(vi_results, y_normalization_scales)
            map_results = normalize_rmse(map_results, y_normalization_scales)
            llb_results = normalize_rmse(llb_results, y_normalization_scales)
        else:
            raise ValueError()
    y_label = None
    if i == 0 or i == 5:
        y_label = "negative LL" if metric == "NLLs" else "RMSE"
    if i == len(datasets) - 1:
        legend = True
    if (
        dataset == "naval"
        and metric == "RMSEs"
        and n_hidden_layers == 1
        and not gap_data
    ):
        axes[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[i].get_yaxis().get_offset_text().set_x(-0.2)
    if dataset == "naval" and metric == "RMSEs" and not gap_data and normalize_results:
        axes[i].set_ylim([-0.005, 0.23])
    plot_uci_results(
        vi_results,
        map_results[0:1],
        llb_results[0:1],
        map_results[size_indices],
        llb_results[size_indices],
        ensemble_sizes=ensemble_sizes,
        color_style=color_style,
        ax_title=dataset,
        y_label=y_label,
        y_max_ticks=y_max_ticks,
        legend=legend,
        legend_kwargs=legend_kwargs,
        fig=fig,
        ax=axes[i],
    )

    # print()
hidden_layers_title = (
    "Two Hidden Layers" if n_hidden_layers == 2 else "One Hidden Layer"
)
gap_title_string = " Gap" if gap_data else ""
metric_string = "Negative LL" if metric == "NLLs" else "RMSE"
# fig.suptitle(f"UCI{gap_title_string} {hidden_layers_title} {metric_string} Comparison")
fig.tight_layout()
plt.subplots_adjust(wspace=wspace)
save_path = figure_dir.joinpath(f"uci{gap_string}_{hidden_layers_string}_{metric}.pdf")
fig.savefig(save_path, bbox_inches="tight")
# tpl.clean_figure()
# tex = tpl.get_tikz_code(
#     axis_width="\\figwidth",  # we want LaTeX to take care of the width
#     axis_height="\\figheight",  # we want LaTeX to take care of the height
#     # we want the plot to look *exactly* like here (e.g. axis limits, axis ticks, etc.)
#     # strict=True,
# )
#
# with save_path.with_suffix(".tex").open("w", encoding="utf-8") as f:
#     f.write(tex)


# %%
figsize = (12, 6)
fig, (axes) = plt.subplots(2, int(np.ceil(len(datasets) / 2)), figsize=figsize)
axes = axes.flatten()
legend = False

for i, dataset in enumerate(datasets[:]):
    experiment_path = Path(f"{data_dir}/{dataset}/results/{vi_experiment_name}.json")
    results = json.loads(experiment_path.read_text())
    vi_nlls = results["VI"][metric]

    new_vi_nlls = []

    # try:
    #     experiment_path = Path(f"{data_dir}/{dataset}/results/uci_vi_new-initialization_old-constant_initial-unconstrained-scale-{-5}_{hidden_layers_string}.json")
    #     results = json.loads(experiment_path.read_text())
    #     new_vi_nlls.append(results["VI"][metric])
    # except:
    #     new_vi_nlls.append(np.full_like(results["VI"][metric], np.mean(results["VI"][metric])))

    try:
        experiment_path = Path(
            f"{data_dir}/{dataset}/results/uci_vi_stefan-mean-initialization_old-constant_initial-unconstrained-scale-{-5}_{hidden_layers_string}.json"
        )
        results = json.loads(experiment_path.read_text())
        new_vi_nlls.append(results["VI"][metric])
    except:
        new_vi_nlls.append(
            np.full_like(results["VI"][metric], np.mean(results["VI"][metric]))
        )

    # try:
    #     experiment_path = Path(f"{data_dir}/{dataset}/results/uci_vi_vposterior-like-stefan_old-constant_initial-unconstrained-scale-{-5}_{hidden_layers_string}.json")
    #     results = json.loads(experiment_path.read_text())
    #     new_vi_nlls.append(results["VI"][metric])
    # except:
    #     new_vi_nlls.append(np.full_like(results["VI"][metric], np.mean(results["VI"][metric])))

    # try:
    #     experiment_path = Path(f"{data_dir}/{dataset}/results/uci_vi_stefan-mean-init_trainable-prior_old-constant_initial-unconstrained-scale-{-5}_{hidden_layers_string}.json")
    #     results = json.loads(experiment_path.read_text())
    #     new_vi_nlls.append(results["VI"][metric])
    # except:
    #     new_vi_nlls.append(np.full_like(results["VI"][metric], np.mean(results["VI"][metric])))

    for initial_unconstrained_scale in [-1, -5, -7]:
        experiment_path = Path(
            f"{data_dir}/{dataset}/results/uci_vi_initial-unconstrained-scale-{initial_unconstrained_scale}_{hidden_layers_string}.json"
        )
        results = json.loads(experiment_path.read_text())
        new_vi_nlls.append(results["VI"][metric])

    experiment_path = Path(
        f"{data_dir}/{dataset}/results/ensemble_sizes/{experiment_name}.json"
    )
    results = json.loads(experiment_path.read_text())
    map_nlls = np.array(results["Ensemble"][metric])

    y_label = None
    if i == 0:
        y_label = "Negative Log Likelihood"
    if i == len(datasets) - 1:
        legend = True
    plot_uci_single_benchmark(
        [vi_nlls] + new_vi_nlls + [map_nlls[0]],
        ax_title=dataset,
        legend=legend,
        legend_kwargs=legend_kwargs,
        labels=[None] * 100,
        fig=fig,
        ax=axes[i],
    )
fig.tight_layout()


# %%
figsize = (12, 6)
fig, (axes) = plt.subplots(2, int(np.ceil(len(datasets) / 2)), figsize=figsize)
axes = axes.flatten()
legend = False

for i, dataset in enumerate(datasets[:]):
    # experiment_path = Path(f"{data_dir}/{dataset}/results/{vi_experiment_name}.json")
    # results = json.loads(experiment_path.read_text())
    # vi_nlls = results["VI"][metric]

    new_vi_nlls = []

    # try:
    #     experiment_path = Path(f"{data_dir}/{dataset}/results/uci_vi_new-initialization_old-constant_initial-unconstrained-scale-{-5}_{hidden_layers_string}.json")
    #     results = json.loads(experiment_path.read_text())
    #     new_vi_nlls.append(results["VI"][metric])
    # except:
    #     new_vi_nlls.append(np.full_like(results["VI"][metric], np.mean(results["VI"][metric])))

    # try:
    #     experiment_path = Path(f"{data_dir}/{dataset}/results/uci_vi_stefan-mean-initialization_old-constant_initial-unconstrained-scale-{-5}_{hidden_layers_string}.json")
    #     results = json.loads(experiment_path.read_text())
    #     new_vi_nlls.append(results["VI"][metric])
    # except:
    #     new_vi_nlls.append(np.full_like(results["VI"][metric], np.mean(results["VI"][metric])))

    # try:
    #     experiment_path = Path(f"{data_dir}/{dataset}/results/uci_vi_vposterior-like-stefan_old-constant_initial-unconstrained-scale-{-5}_{hidden_layers_string}.json")
    #     results = json.loads(experiment_path.read_text())
    #     new_vi_nlls.append(results["VI"][metric])
    # except:
    #     new_vi_nlls.append(np.full_like(results["VI"][metric], np.mean(results["VI"][metric])))

    # try:
    #     experiment_path = Path(f"{data_dir}/{dataset}/results/uci_vi_stefan-mean-init_trainable-prior_old-constant_initial-unconstrained-scale-{-5}_{hidden_layers_string}.json")
    #     results = json.loads(experiment_path.read_text())
    #     new_vi_nlls.append(results["VI"][metric])
    # except:
    #     new_vi_nlls.append(np.full_like(results["VI"][metric], np.mean(results["VI"][metric])))

    for initial_unconstrained_scale in [-1, -5, -7]:
        experiment_path = Path(
            f"{data_dir}/{dataset}/results/uci_vi_initial-unconstrained-scale-{initial_unconstrained_scale}_{hidden_layers_string}.json"
        )
        results = json.loads(experiment_path.read_text())
        new_vi_nlls.append(results["VI"][metric])

    experiment_path = Path(
        f"{data_dir}/{dataset}/results/ensemble_sizes/{experiment_name}.json"
    )
    results = json.loads(experiment_path.read_text())
    map_nlls = np.array(results["Ensemble"][metric])

    y_label = None
    if i == 0:
        y_label = "Negative Log Likelihood"
    if i == len(datasets) - 1:
        legend = True
    plot_uci_single_benchmark(
        new_vi_nlls + [map_nlls[-1]],
        ax_title=dataset,
        legend=legend,
        legend_kwargs=legend_kwargs,
        labels=[None] * 100,
        fig=fig,
        ax=axes[i],
    )
fig.tight_layout()


# %% markdown
# # Fixed Epochs RMSEs

# %%
experiment_name = "epochs-40_one-hidden-layer"
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble MM", "LLB Ensemble MM"]


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
save_path = figure_dir.joinpath(f"uci_{experiment_name}_rmses.pdf")
# fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Fixed Epochs NLLs

# %%
experiment_name = "epochs-40_one-hidden-layer"
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble MM", "LLB Ensemble MM"]

fig, (axes) = plt.subplots(1, len(datasets), figsize=figsize)
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_data/{dataset}/results/{experiment_name}.json")
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
save_path = figure_dir.joinpath(f"uci_{experiment_name}_nlls.pdf")
# fig.savefig(save_path, bbox_inches="tight")


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
save_path = figure_dir.joinpath(f"uci_{experiment_name}_rmses.pdf")
# fig.savefig(save_path, bbox_inches="tight")


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
    experiment_path = Path(f"uci_data/{dataset}/results/{experiment_name}.json")
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
save_path = figure_dir.joinpath(f"uci_{experiment_name}_nlls.pdf")
# fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # L2 convergence rmses

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
save_path = figure_dir.joinpath(f"uci_{experiment_name}_rmses.pdf")
# fig.savefig(save_path, bbox_inches="tight")

# %% markdown
# # L2 convergence NLLs

# %%
weight_prior_scale = 1
patience = 20
experiment_name = (
    f"l2-reg-prior-scale-{weight_prior_scale}-patience-{patience}_one-hidden-layer"
)
models = ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
labels = ["VI", "Map", "Last Layer Bayesian", "Ensemble MM", "LLB Ensemble MM"]

figsize = (12, 6)
fig, (axes) = plt.subplots(2, int(np.ceil(len(datasets) / 2)), figsize=figsize)
axes = axes.flatten()
legend = False
for i, dataset in enumerate(datasets):
    experiment_path = Path(f"uci_data/{dataset}/results/{experiment_name}.json")
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
save_path = figure_dir.joinpath(f"uci_{experiment_name}_nlls.pdf")
# fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Moment matched versus Mixture prediction

# Cocnlusion: In terms of log likelihood it does not make a big difference whether to
# use the predictive mixture distribution or the moment matched version
# (ensemble size = 5). Perhaps the moment matched version is slightly superior.

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
save_path = figure_dir.joinpath(f"uci_{experiment_name}_mm_vs_mixture.pdf")
fig.savefig(save_path, bbox_inches="tight")

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
save_path = figure_dir.joinpath(
    f"uci_{experiment_name}_vi_flat_vs_informative_prior.pdf"
)
fig.savefig(save_path, bbox_inches="tight")


# %% markdown
# # Training Times

# Conclusion: Adding the Bayesian linear regression on top adds less than 1% of the MAP
# training time on top. In fact it is more like 0.3%.

# %%
weight_prior_scale = 1
patience = 20
experiment_name = (
    f"l2-reg-prior-scale-{weight_prior_scale}-patience-{patience}_one-hidden-layer"
)
# experiment_name = f"early-stop-patience-{patience}_one-hidden-layer"
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
