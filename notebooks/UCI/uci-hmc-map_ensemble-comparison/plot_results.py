import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pylab
import tikzplotlib as tpl
from brokenaxes import brokenaxes

from core.evaluation_utils import (
    backtransform_normalized_ll,
    backtransform_normalized_rmse,
    get_y_normalization_scales,
)

figure_dir = "figures/hmc_uci"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)


# %%
dataset = "concrete"
data_seed = 0
dataset_name = f"{dataset}_{data_seed + 1:02}"

n_hidden_layers = 1
hidden_layers_string = (
    "two-hidden-layers" if n_hidden_layers == 2 else "one-hidden-layer"
)
plot_hidden_layer_string = "1HL" if n_hidden_layers else "2HL"

save_dir = f".save_uci_models/hmc-map-ensemble-comparison/{hidden_layers_string}/{dataset_name}"
save_dir = Path(save_dir)

# %%
save_path = save_dir.joinpath(f"evaluation_results")
with open(save_path, "rb") as f:
    dic = pickle.load(f)

ensemble_sizes = dic["ensemble_sizes"]
ensemble_rmsess = dic["ensemble_rmsess"]
ensemble_nllss = dic["ensemble_nllss"]
ensemble_wdss = dic["ensemble_wdss"]
llb_ensemble_rmsess = dic["llb_ensemble_rmsess"]
llb_ensemble_nllss = dic["llb_ensemble_nllss"]
llb_ensemble_wdss = dic["llb_ensemble_wdss"]
hmc_ensemble_rmsess = dic["hmc_ensemble_rmsess"]
hmc_ensemble_nllss = dic["hmc_ensemble_nllss"]
hmc_ensemble_wdss = dic["hmc_ensemble_wdss"]
hmc_rmse = dic["hmc_rmse"]
hmc_nll = dic["hmc_nll"]
hmc_wd = dic["hmc_wd"]
large_rmse = dic["large_rmse"]
large_nll = dic["large_nll"]
large_wd = dic["large_wd"]
large_llb_rmse = dic["large_llb_rmse"]
large_llb_nll = dic["large_llb_nll"]
large_llb_wd = dic["large_llb_wd"]

y_normalization_scale = get_y_normalization_scales(dataset, gap_data=False)[0, 0:1]
y_normalization_scale

ensemble_nllss = backtransform_normalized_ll(ensemble_nllss, y_normalization_scale)
llb_ensemble_nllss = backtransform_normalized_ll(
    llb_ensemble_nllss, y_normalization_scale
)
hmc_ensemble_nllss = backtransform_normalized_ll(
    hmc_ensemble_nllss, y_normalization_scale
)
hmc_nll = backtransform_normalized_ll(hmc_nll, y_normalization_scale)
large_nll = backtransform_normalized_ll(large_nll, y_normalization_scale)
large_llb_nll = backtransform_normalized_ll(large_llb_nll, y_normalization_scale)

ensemble_rmsess = backtransform_normalized_rmse(ensemble_rmsess, y_normalization_scale)
llb_ensemble_rmsess = backtransform_normalized_rmse(
    llb_ensemble_rmsess, y_normalization_scale
)
hmc_ensemble_rmsess = backtransform_normalized_rmse(
    hmc_ensemble_rmsess, y_normalization_scale
)
hmc_rmse = backtransform_normalized_rmse(hmc_rmse, y_normalization_scale)
large_rmse = backtransform_normalized_rmse(large_rmse, y_normalization_scale)
large_llb_rmse = backtransform_normalized_rmse(large_llb_rmse, y_normalization_scale)


ensemble_rmsess.shape
hmc_ensemble_rmsess.shape


# %%
def plot_results(ensemble_sizes, results, fig=None, ax=None, label=None):
    if fig is None:
        fig, ax = plt.subplots()
    results = np.array(results)
    size_means = np.mean(results, axis=0)
    size_stds = np.std(results, axis=0)
    lines = ax.errorbar(
        ensemble_sizes,
        size_means,
        size_stds / np.sqrt(results.shape[0]),
        fmt="o",
        markersize=markersize,
        alpha=alpha,
        label=label,
    )
    return lines


# %%
wspace = 0.14
x_offset_around_breaks = 5
x_offset_final = 7
markersize = 4
alpha = 1
figsize = (4, 1.8)

fig = plt.figure(figsize=figsize)
ax = brokenaxes(
    xlims=(
        (-2, 50 + x_offset_around_breaks),
        (
            ensemble_sizes[-3] - x_offset_around_breaks,
            ensemble_sizes[-3] + x_offset_around_breaks,
        ),
        (
            ensemble_sizes[-2] - x_offset_around_breaks,
            ensemble_sizes[-2] + x_offset_around_breaks,
        ),
        (
            ensemble_sizes[-1] - x_offset_around_breaks,
            ensemble_sizes[-1] + x_offset_final,
        ),
    ),
    wspace=wspace,
    despine=False,
)
plot_results(ensemble_sizes[:-2], ensemble_rmsess, fig=fig, ax=ax, label="MAP")
plot_results(ensemble_sizes[:-2], llb_ensemble_rmsess, fig=fig, ax=ax, label="NL")
plot_results(ensemble_sizes, hmc_ensemble_rmsess, fig=fig, ax=ax, label="HMC")
ax.hlines(
    hmc_rmse,
    ensemble_sizes[0],
    ensemble_sizes[-1] + x_offset_around_breaks,
    color="k",
    label="Full HMC",
)
ax.scatter(1000, large_rmse, s=markersize ** 2, alpha=alpha)
ax.scatter(1000, large_llb_rmse, s=markersize ** 2, alpha=alpha)
# ax.set_title(f"{plot_hidden_layer_string}  {dataset}  RMSE")
# ax.set_xlabel("# Ensemble Members")
if True or n_hidden_layers == 1 and dataset == "yacht" or dataset == "boston":
    ax.set_ylabel("RMSE", fontsize=12)
if dataset == "yacht":
    ax.set_ylim([0.25, 1.3])
# ax.set_xticks([1, 25, 50, 100, 1000, 10000])
# ax.set_xticklabels([1, 25, 50, 100, "1k", "10k"])
# import matplotlib.ticker as ticker
# locator = ticker.FixedLocator([10, 20], nbins=10)
# ax.xaxis.set_major_locator(locator)
ax.tick_params(labelbottom=False)
# ax.legend()
save_path = figure_dir.joinpath(
    f"{dataset_name}_{hidden_layers_string}_HMC-Map-Ensemble-RMSE.pdf"
)
fig.savefig(save_path, bbox_inches="tight")
# tex = tpl.get_tikz_code(
#     axis_width="\\figwidth",  # we want LaTeX to take care of the width
#     axis_height="\\figheight",  # we want LaTeX to take care of the height
#     # we want the plot to look *exactly* like here (e.g. axis limits, axis ticks, etc.)
#     # strict=True,
# )

# with save_path.with_suffix(".tex").open("w", encoding="utf-8") as f:
#     f.write(tex)
# %%
fig = plt.figure(figsize=figsize)
ax = brokenaxes(
    xlims=(
        (-2, 50 + x_offset_around_breaks),
        (
            ensemble_sizes[-3] - x_offset_around_breaks,
            ensemble_sizes[-3] + x_offset_around_breaks,
        ),
        (
            ensemble_sizes[-2] - x_offset_around_breaks,
            ensemble_sizes[-2] + x_offset_around_breaks,
        ),
        (
            ensemble_sizes[-1] - x_offset_around_breaks,
            ensemble_sizes[-1] + x_offset_final,
        ),
    ),
    wspace=wspace,
    despine=False,
)
plot_results(ensemble_sizes[:-2], ensemble_nllss, fig=fig, ax=ax, label="MAP")
plot_results(ensemble_sizes[:-2], llb_ensemble_nllss, fig=fig, ax=ax, label="NL")
plot_results(ensemble_sizes, hmc_ensemble_nllss, fig=fig, ax=ax, label="HMC")
ax.hlines(
    hmc_nll,
    ensemble_sizes[0],
    ensemble_sizes[-1] + x_offset_around_breaks,
    color="k",
    label="Full HMC",
)
ax.scatter(1000, large_nll, s=markersize ** 2, alpha=alpha)
ax.scatter(1000, large_llb_nll, s=markersize ** 2, alpha=alpha)
# ax.set_title(f"{dataset_name}; Negative Log Likelihood")
# ax.set_xlabel("# Ensemble Members")
if True or n_hidden_layers == 1 and dataset == "yacht" or dataset == "boston":
    ax.set_ylabel("Negative LL", fontsize=12)
if dataset == "yacht":
    ax.set_ylim([-0.60, 5.5])
ax.tick_params(labelbottom=False)
# ax.set_ylim([None, 4])
# ax.legend()
fig.savefig(
    figure_dir.joinpath(
        f"{dataset_name}_{hidden_layers_string}_HMC-Map-Ensemble-LL.pdf"
    ),
    bbox_inches="tight",
)


# %%
fig = plt.figure(figsize=figsize)
ax = brokenaxes(
    xlims=(
        (-2, 50 + x_offset_around_breaks),
        (
            ensemble_sizes[-3] - x_offset_around_breaks,
            ensemble_sizes[-3] + x_offset_around_breaks,
        ),
        (
            ensemble_sizes[-2] - x_offset_around_breaks,
            ensemble_sizes[-2] + x_offset_around_breaks,
        ),
        (
            ensemble_sizes[-1] - x_offset_around_breaks,
            ensemble_sizes[-1] + x_offset_final,
        ),
    ),
    wspace=wspace,
    despine=False,
)
map_line = plot_results(
    ensemble_sizes[:-2], ensemble_wdss, fig=fig, ax=ax, label="MAP"
)[0]
llb_line = plot_results(
    ensemble_sizes[:-2], llb_ensemble_wdss, fig=fig, ax=ax, label="NL"
)[0]
hmc_line = plot_results(ensemble_sizes, hmc_ensemble_wdss, fig=fig, ax=ax, label="HMC")[
    0
]
full_hmc_line = ax.hlines(
    hmc_wd,
    ensemble_sizes[0],
    ensemble_sizes[-1] + x_offset_around_breaks,
    color="k",
    label="Full HMC",
)[0]
ax.scatter(1000, large_wd, s=markersize ** 2, alpha=alpha)
ax.scatter(1000, large_llb_wd, s=markersize ** 2, alpha=alpha)
# ax.set_title(f"{dataset_name}; Predictive Wasserstein Distance")
ax.set_xlabel("# Ensemble Members", fontsize=12)
if True or n_hidden_layers == 1 and dataset == "yacht" or dataset == "boston":
    ax.set_ylabel("W. Distance", fontsize=12)
if dataset == "yacht":
    ax.set_ylim([-0.006, 0.08])
    if n_hidden_layers == 2:
        ax.set_yticks([0.02, 0.06])
    else:
        ax.set_yticks([0.02, 0.06])

    # ax.set_yticklabels([1, 25, 50, 100, "1k", "10k"])
# ax.legend()
fig.savefig(
    figure_dir.joinpath(
        f"{dataset_name}_{hidden_layers_string}_HMC-Map-Ensemble-predictive-WD.pdf"
    ),
    bbox_inches="tight",
)

fig = pylab.figure()
figlegend = pylab.figure(figsize=(4, 0.3))
figlegend.legend(
    [map_line, llb_line, hmc_line, full_hmc_line],
    ["MAP", "NL", "HMC", "Full HMC"],
    "center",
    ncol=4,
)
figlegend.savefig(figure_dir.joinpath("hmc_uci_legend.pdf"))
