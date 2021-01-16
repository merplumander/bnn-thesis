# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from brokenaxes import brokenaxes

from core.evaluation_utils import rmse, wasserstein_distance_predictive_samples
from core.hmc import hmc_density_network_from_save_path
from core.last_layer import PostHocLastLayerBayesianEnsemble as LLBEnsemble
from core.map import map_density_ensemble_from_save_path
from core.network_utils import (
    build_scratch_density_model,
    make_independent_gaussian_network_prior,
    transform_unconstrained_scale,
)
from core.preprocessing import StandardPreprocessor
from data.uci import load_uci_data

tfd = tfp.distributions

figure_dir = "figures/temp"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)

# %%
dataset = "yacht"
data_seed = 0
dataset_name = f"{dataset}_{data_seed + 1:02}"

n_hidden_layers = 2
hidden_layers_string = (
    "two-hidden-layers" if n_hidden_layers == 2 else "one-hidden-layer"
)

save_dir = f".save_uci_models/hmc-map-ensemble-comparison/{hidden_layers_string}/{dataset_name}"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)


_x, _y, train_indices, _, test_indices = load_uci_data(f"{dataset}")
_x_train = _x[train_indices[data_seed]]
_y_train = _y[train_indices[data_seed]].reshape(-1, 1)
_x_test = _x[test_indices[data_seed]]
_y_test = _y[test_indices[data_seed]].reshape(-1, 1)

x_preprocessor = StandardPreprocessor()
x_train = x_preprocessor.fit_transform(_x_train)
x_test = x_preprocessor.transform(_x_test)
y_preprocessor = StandardPreprocessor()
y_train = y_preprocessor.fit_transform(_y_train)
y_test = y_preprocessor.transform(_y_test)

# unnormalized_ll_constant = np.log(y_preprocessor.scaler.scale_)


# %%
with open("config/uci-hyperparameters-config.yaml") as f:
    experiment_config = yaml.full_load(f)

a = experiment_config["noise_var_prior_a"]
b = experiment_config["noise_var_prior_b"]
_var_d = tfd.InverseGamma([[a]], b)
noise_scale_prior = tfd.TransformedDistribution(
    distribution=_var_d, bijector=tfp.bijectors.Invert(tfp.bijectors.Square())
)

train_seed = experiment_config["train_seed"]
layer_units = experiment_config["layer_units"]
layer_activations = experiment_config["layer_activations"]
if n_hidden_layers == 2:
    layer_units.insert(0, layer_units[0])
    layer_activations.insert(0, layer_activations[0])
initial_unconstrained_scale = experiment_config["initial_unconstrained_scale"]
transform_unconstrained_scale_factor = experiment_config[
    "transform_unconstrained_scale_factor"
]

batch_size = experiment_config["batch_size"]

n_train = x_train.shape[0]
n_test = x_test.shape[0]
n_features = x_train.shape[1]
input_shape = [n_features]
network_prior = make_independent_gaussian_network_prior(
    input_shape=input_shape,
    layer_units=layer_units,
    loc=0.0,
    scale=experiment_config["weight_prior_scale"],
)

# %%
save_path = save_dir.joinpath(f"hmc-{dataset_name}")
hmc_net = hmc_density_network_from_save_path(
    save_path, network_prior=network_prior, noise_scale_prior=noise_scale_prior
)

# %%
ensemble = map_density_ensemble_from_save_path(
    (save_dir.joinpath(f"large_map_ensemble")), noise_scale_prior=noise_scale_prior
)

nets = ensemble.networks


# %%
llb_ensemble = LLBEnsemble(
    n_networks=ensemble.n_networks,
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=initial_unconstrained_scale,
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    last_layer_prior="weakly-informative",
    seed=train_seed,
)

llb_ensemble.fit(
    x_train, y_train, batch_size=batch_size, pretrained_networks=ensemble.networks
)

llb_nets = llb_ensemble.networks


# %%
n_predictive_samples = 1000
hmc_predictive_distribution = hmc_net.predict(x_test, thinning=2)
hmc_predictive_samples = np.squeeze(
    hmc_predictive_distribution.sample(n_predictive_samples).numpy()
)
hmc_rmse = rmse(hmc_predictive_distribution.mean(), y_test)
hmc_nll = -hmc_predictive_distribution.log_prob(y_test) / n_test
_hmc_predictive_samples2 = np.squeeze(
    hmc_predictive_distribution.sample(n_predictive_samples).numpy()
)
hmc_wd = wasserstein_distance_predictive_samples(
    hmc_predictive_samples, _hmc_predictive_samples2
)
hmc_net.n_samples


# %%
max_ensemble_members = 100
n_repetitions = 10

ensemble_sizes = np.arange(50) + 1
ensemble_sizes = np.append(
    ensemble_sizes[np.logical_or(ensemble_sizes <= 20, ensemble_sizes % 5 == 0)], 100
)

# %%
netss = [
    nets[i * max_ensemble_members : (i + 1) * max_ensemble_members]
    for i in range(n_repetitions)
]

ensemble_rmsess = []
ensemble_nllss = []
ensemble_wdss = []
for _nets in netss:
    ensemble_rmses = []
    ensemble_nlls = []
    ensemble_wds = []
    for n_networks in ensemble_sizes:
        ensemble.networks = _nets[0:n_networks]
        pd = ensemble.predict(x_test)
        _rmse = rmse(pd.mean(), y_test)
        nll = -tf.reduce_mean(pd.log_prob(y_test))
        ensemble_predictive_samples = np.squeeze(
            pd.sample(n_predictive_samples).numpy()
        )
        wd = wasserstein_distance_predictive_samples(
            hmc_predictive_samples, ensemble_predictive_samples
        )
        ensemble_rmses.append(_rmse)
        ensemble_nlls.append(nll)
        ensemble_wds.append(wd)
    ensemble_rmsess.append(ensemble_rmses)
    ensemble_nllss.append(ensemble_nlls)
    ensemble_wdss.append(ensemble_wds)
    print(f"Done Ensemble {len(ensemble_rmsess)}")
ensemble.networks = nets


# %%
llb_netss = [
    llb_nets[i * max_ensemble_members : (i + 1) * max_ensemble_members]
    for i in range(n_repetitions)
]

llb_ensemble_rmsess = []
llb_ensemble_nllss = []
llb_ensemble_wdss = []
for _nets in llb_netss:
    ensemble_rmses = []
    ensemble_nlls = []
    ensemble_wds = []
    for n_networks in ensemble_sizes:
        llb_ensemble.networks = _nets[0:n_networks]
        pd = llb_ensemble.predict(x_test)
        _rmse = rmse(pd.mean(), y_test)
        nll = -tf.reduce_mean(pd.log_prob(y_test))
        ensemble_predictive_samples = np.squeeze(
            pd.sample(n_predictive_samples).numpy()
        )
        wd = wasserstein_distance_predictive_samples(
            hmc_predictive_samples, ensemble_predictive_samples
        )
        ensemble_rmses.append(_rmse)
        ensemble_nlls.append(nll)
        ensemble_wds.append(wd)
    llb_ensemble_rmsess.append(ensemble_rmses)
    llb_ensemble_nllss.append(ensemble_nlls)
    llb_ensemble_wdss.append(ensemble_wds)
    print(f"Done Ensemble {len(llb_ensemble_rmsess)}")
llb_ensemble.networks = llb_nets


# %%
ensemble_sizes = np.arange(50) + 1
ensemble_sizes = np.append(
    ensemble_sizes[np.logical_or(ensemble_sizes <= 20, ensemble_sizes % 5 == 0)],
    [100, 1000, 10000],
)

n_chains = hmc_net.n_used_chains
n_samples = hmc_net.n_samples


hmc_ensemble_rmsess = []
hmc_ensemble_nllss = []
hmc_ensemble_wdss = []
for i_ensemble in range(n_repetitions):
    np.random.seed(i_ensemble)
    indices = np.random.randint(
        low=[0, 0], high=[n_samples, n_chains], size=[ensemble_sizes[-1], 2]
    )
    ensemble_rmses = []
    ensemble_nlls = []
    ensemble_wds = []
    for n_networks in ensemble_sizes:
        pd = hmc_net.predict_from_indices(x_test, indices[:n_networks])
        _rmse = rmse(pd.mean(), y_test)
        nll = -pd.log_prob(y_test) / n_test
        ensemble_predictive_samples = np.squeeze(
            pd.sample(n_predictive_samples).numpy()
        )
        wd = wasserstein_distance_predictive_samples(
            hmc_predictive_samples, ensemble_predictive_samples
        )
        ensemble_rmses.append(_rmse)
        ensemble_nlls.append(nll)
        ensemble_wds.append(wd)
    hmc_ensemble_rmsess.append(ensemble_rmses)
    hmc_ensemble_nllss.append(ensemble_nlls)
    hmc_ensemble_wdss.append(ensemble_wds)
    print(f"Done Ensemble {len(hmc_ensemble_rmsess)}")


# %%
pd = ensemble.predict(x_test)
large_rmse = rmse(pd.mean(), y_test)
large_nll = -tf.reduce_mean(pd.log_prob(y_test))
ensemble_predictive_samples = np.squeeze(pd.sample(n_predictive_samples).numpy())
large_wd = wasserstein_distance_predictive_samples(
    hmc_predictive_samples, ensemble_predictive_samples
)

pd = llb_ensemble.predict(x_test)
large_llb_rmse = rmse(pd.mean(), y_test)
large_llb_nll = -tf.reduce_mean(pd.log_prob(y_test))
ensemble_predictive_samples = np.squeeze(pd.sample(n_predictive_samples).numpy())
large_llb_wd = wasserstein_distance_predictive_samples(
    hmc_predictive_samples, ensemble_predictive_samples
)


# %%
def plot_results(ensemble_sizes, results, fig=None, ax=None, label=None):
    if fig is None:
        fig, ax = plt.subplots()
    results = np.array(results)
    size_means = np.mean(results, axis=0)
    size_stds = np.std(results, axis=0)
    ax.errorbar(
        ensemble_sizes,
        size_means,
        size_stds / np.sqrt(results.shape[0]),
        fmt="o",
        label=label,
    )


# %%
wspace = 0.08

fig = plt.figure()
ax = brokenaxes(
    xlims=(
        (-2, 53),
        (ensemble_sizes[-3] - 3, ensemble_sizes[-3] + 3),
        (ensemble_sizes[-2] - 3, ensemble_sizes[-2] + 3),
        (ensemble_sizes[-1] - 3, ensemble_sizes[-1] + 5),
    ),
    wspace=wspace,
    despine=False,
)
plot_results(ensemble_sizes[:-2], ensemble_rmsess, fig=fig, ax=ax, label="MAP Ensemble")
plot_results(
    ensemble_sizes[:-2], llb_ensemble_rmsess, fig=fig, ax=ax, label="LLB Ensemble"
)
plot_results(ensemble_sizes, hmc_ensemble_rmsess, fig=fig, ax=ax, label="HMC Samples")
ax.hlines(
    hmc_rmse, ensemble_sizes[0], ensemble_sizes[-1] + 3, color="k", label="Full HMC"
)
ax.scatter(1000, large_rmse)
ax.scatter(1000, large_llb_rmse)
ax.set_title(f"{dataset_name}; RMSE")
ax.set_xlabel("# Ensemble Members")
ax.set_ylabel("RMSE")
ax.legend()
fig.savefig(
    figure_dir.joinpath(
        f"{dataset_name}_{hidden_layers_string}_HMC-Map-Ensemble-RMSE.pdf"
    ),
    bbox_inches="tight",
)


# %%
fig = plt.figure()
ax = brokenaxes(
    xlims=(
        (-2, 53),
        (ensemble_sizes[-3] - 3, ensemble_sizes[-3] + 3),
        (ensemble_sizes[-2] - 3, ensemble_sizes[-2] + 3),
        (ensemble_sizes[-1] - 3, ensemble_sizes[-1] + 5),
    ),
    wspace=wspace,
    despine=False,
)
plot_results(ensemble_sizes[:-2], ensemble_nllss, fig=fig, ax=ax, label="MAP Ensemble")
plot_results(
    ensemble_sizes[:-2], llb_ensemble_nllss, fig=fig, ax=ax, label="LLB Ensemble"
)
plot_results(ensemble_sizes, hmc_ensemble_nllss, fig=fig, ax=ax, label="HMC Samples")
ax.hlines(
    hmc_nll, ensemble_sizes[0], ensemble_sizes[-1] + 3, color="k", label="Full HMC"
)
ax.scatter(1000, large_nll)
ax.scatter(1000, large_llb_nll)
ax.set_title(f"{dataset_name}; Negative Log Likelihood")
ax.set_xlabel("# Ensemble Members")
ax.set_ylabel("Negative Log Likelihood")
ax.set_ylim([None, 4])
ax.legend()
fig.savefig(
    figure_dir.joinpath(
        f"{dataset_name}_{hidden_layers_string}_HMC-Map-Ensemble-LL.pdf"
    ),
    bbox_inches="tight",
)


# %%
fig = plt.figure()
ax = brokenaxes(
    xlims=(
        (-2, 53),
        (ensemble_sizes[-3] - 3, ensemble_sizes[-3] + 3),
        (ensemble_sizes[-2] - 3, ensemble_sizes[-2] + 3),
        (ensemble_sizes[-1] - 3, ensemble_sizes[-1] + 5),
    ),
    wspace=wspace,
    despine=False,
)
plot_results(ensemble_sizes[:-2], ensemble_wdss, fig=fig, ax=ax, label="MAP Ensemble")
plot_results(
    ensemble_sizes[:-2], llb_ensemble_wdss, fig=fig, ax=ax, label="LLB Ensemble"
)
plot_results(ensemble_sizes, hmc_ensemble_wdss, fig=fig, ax=ax, label="HMC Samples")
ax.hlines(
    hmc_wd, ensemble_sizes[0], ensemble_sizes[-1] + 3, color="k", label="Full HMC"
)
ax.scatter(1000, large_wd)
ax.scatter(1000, large_llb_wd)
ax.set_title(f"{dataset_name}; Predictive Wasserstein Distance")
ax.set_xlabel("# Ensemble Members")
ax.set_ylabel("Wasserstein Distance")
ax.legend()
fig.savefig(
    figure_dir.joinpath(
        f"{dataset_name}_{hidden_layers_string}_HMC-Map-Ensemble-predictive-WD.pdf"
    ),
    bbox_inches="tight",
)


# # %% markdown
# # # Are predictive distributions of members all different to each other?
#
# # %%
# n_predictive_samples = 10000
# individual_nets_wds = []
# pd0 = large_nets[0].predict(x_test)
# predictive_samples0 = np.squeeze(pd0.sample(n_predictive_samples).numpy())
# for net in large_nets[:]:
#     pd1 = net.predict(x_test)
#     predictive_samples1 = np.squeeze(pd1.sample(n_predictive_samples).numpy())
#     individual_nets_wds.append(
#         wasserstein_distance_predictive_samples(
#             predictive_samples0, predictive_samples1
#         )
#     )
#
# # %%
# fig, ax = plt.subplots()
# ax.hist(individual_nets_wds, bins=50)
# fig.savefig(
#     figure_dir.joinpath(f"{dataset_name}_{hidden_layers_string}_map-single-networks-predictive-wd.pdf"),
#     bbox_inches="tight",
# )
#
#
# # %%
# n_predictive_samples = 10000
# llb_individual_nets_wds = []
# pd0 = large_llb_nets[0].predict(x_test)
# predictive_samples0 = np.squeeze(pd0.sample(n_predictive_samples).numpy())
# for net in large_llb_nets[:300]:
#     pd1 = net.predict(x_test)
#     predictive_samples1 = np.squeeze(pd1.sample(n_predictive_samples).numpy())
#     llb_individual_nets_wds.append(
#         wasserstein_distance_predictive_samples(
#             predictive_samples0, predictive_samples1
#         )
#     )
#
# # %%
# fig, ax = plt.subplots()
# ax.hist(llb_individual_nets_wds, bins=50)
# fig.savefig(
#     figure_dir.joinpath(f"{dataset_name}_{hidden_layers_string}_llb-single-networks-predictive-wd.pdf"),
#     bbox_inches="tight",
# )
#
#
# # %%
# n_chains = hmc_net.n_used_chains
# n_samples = hmc_net.n_samples
#
# sample_indices = np.random.choice(n_samples, size=len(large_nets), replace=False)
# chain_indices = np.random.choice(n_chains, size=len(large_nets))
# indices = np.stack([sample_indices, chain_indices], axis=-1)
# n_predictive_samples = 10000
# hmc_individual_nets_wds = []
# pd0 = hmc_net.predict_from_indices(x_test, indices[0:1])
# predictive_samples0 = np.squeeze(pd0.sample(n_predictive_samples).numpy())
# for i in indices[:300]:
#     pd1 = hmc_net.predict_from_indices(x_test, i.reshape(1, 2))
#     predictive_samples1 = np.squeeze(pd1.sample(n_predictive_samples).numpy())
#     hmc_individual_nets_wds.append(
#         wasserstein_distance_predictive_samples(
#             predictive_samples0, predictive_samples1
#         )
#     )
#
# # %%
# fig, ax = plt.subplots()
# ax.hist(hmc_individual_nets_wds, bins=50)
# fig.savefig(
#     figure_dir.joinpath(f"{dataset_name}_{hidden_layers_string}_hmc-single-networks-predictive-wd.pdf"),
#     bbox_inches="tight",
# )
