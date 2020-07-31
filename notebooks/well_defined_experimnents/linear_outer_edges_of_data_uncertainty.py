# %load_ext autoreload
# %autoreload 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from core.hmc import HMCDensityNetwork, hmc_density_network_from_save_path
from core.last_layer import PostHocLastLayerBayesianEnsemble as LLBEnsemble
from core.last_layer import PostHocLastLayerBayesianNetwork as LLBNetwork
from core.map import MapDensityEnsemble, MapDensityNetwork
from core.plotting_utils import (
    plot_distribution_samples,
    plot_ground_truth,
    plot_moment_matched_predictive_normal_distribution,
    plot_moment_matched_predictive_normal_distribution_and_function_samples,
    plot_training_data,
)
from core.preprocessing import StandardPreprocessor
from data.toy_regression import cos_linear_edge_data, ground_truth_cos_function

tfd = tfp.distributions

figure_dir = "figures/linear_outer_edges_uncertainty"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)
experiment_name = "linear-outer-edge"

# %%
seed = 1
n_train = 50


# train and test variables beginning with an underscore are unprocessed.
_x_train, y_train = cos_linear_edge_data(n_data=n_train, sigma=0.1, seed=seed)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(
    _x_train, test_ds=0.8
)
y_ground_truth = ground_truth_cos_function(_x_plot)

layer_units = [50, 20] + [2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]


# %% codecell
y_lim = [-5, 5]
figsize = (10, 6)
fig, ax = plt.subplots(figsize=figsize)
plot_training_data(_x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(_x_plot, y_ground_truth, fig=fig, ax=ax, alpha=0.2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(y_lim)
fig.suptitle("Data with approximately linear edges", fontsize=15)
ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
ax.legend()
fig.savefig(figure_dir.joinpath(f"data_{experiment_name}.pdf"), bbox_inches="tight")

# %%
# General training
epochs = 100
batch_size = n_train

# %% markdown
# # MAP Density Model

# %%
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)


net = MapDensityNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
)

net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)

# %%
prediction = net.predict(x_plot)  # Mixture Of Gaussian prediction
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    save_path=figure_dir.joinpath(f"map_network_{experiment_name}.pdf"),
)


# %% markdown
# # Density Ensemble

# %%
n_networks = 10

# %%
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)

ensemble = MapDensityEnsemble(
    n_networks=n_networks,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=seed,
)

ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


# %%
prediction = ensemble.predict(x_plot)  # Mixture Of Gaussian prediction
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title="Ensemble of ten MAP networks",
    save_path=figure_dir.joinpath(f"ensemble_moment_matched_{experiment_name}.pdf"),
)


# %% codecell
gaussian_predictions = ensemble.predict_list_of_gaussians(x_plot, n_predictions=3)
fig, ax = plt.subplots(figsize=figsize)
plot_distribution_samples(
    x_plot=_x_plot,
    distribution_samples=gaussian_predictions,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    save_path=figure_dir.joinpath(f"ensemble_members_{experiment_name}.pdf"),
)


# %% markdown
# # Neural Linear Model


# %%
layer_units[-1] = 1
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)

# last layer bayesian network
llb_net = LLBNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=4,
)
layer_units[-1] = 2

llb_net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)

# %%
prediction = llb_net.predict(x_plot)
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title="Neural Linear Model",
    save_path=figure_dir.joinpath(f"llb_moment_matched_{experiment_name}.pdf"),
)


# %% markdown
# # HMC

# %%
save_dir = ".save_data/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# %%
layer_units[-1] = 2
prior_std = 1
weight_priors = [tfd.Normal(0, prior_std)] * len(layer_units)
bias_priors = weight_priors

unique_hmc_save_path = f"hmc_n-units-{layer_units}_activations-{layer_activations}_prior-std-{prior_std}_data-cos-linear-edge"
unique_hmc_save_path = save_dir.joinpath(unique_hmc_save_path)

# %%
rerun_training = False


# %%
sampler = "hmc"
num_burnin_steps = 500
num_results = 3000
num_leapfrog_steps = 25
step_size = 0.1

if rerun_training or not unique_hmc_save_path.is_file():
    hmc_net = HMCDensityNetwork(
        [1],
        layer_units,
        layer_activations,
        weight_priors=weight_priors,
        bias_priors=bias_priors,
        seed=0,
    )

    hmc_net.fit(
        x_train,
        y_train,
        sampler=sampler,
        num_burnin_steps=num_burnin_steps,
        num_results=num_results,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=step_size,
        learning_rate=0.01,
        batch_size=n_train,
        epochs=80,
        verbose=0,
    )

    hmc_net.save(unique_hmc_save_path)
else:
    hmc_net = hmc_density_network_from_save_path(unique_hmc_save_path)


# %%
prediction = hmc_net.predict(x_plot, thinning=10)
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title="HMC",
    save_path=figure_dir.joinpath(f"hmc_{experiment_name}.pdf"),
)


# %% markdown
# # Neural Linear Ensemble

# %%
layer_units[-1] = 1
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)
llb_ensemble = LLBEnsemble(
    n_networks=n_networks,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)
layer_units[-1] = 2

llb_ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


# %%
prediction = llb_ensemble.predict(x_plot)
distribution_samples = llb_ensemble.predict_list(x_plot)
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    # distribution_samples=distribution_samples,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title="Neural Linear Ensemble of ten networks",
    save_path=figure_dir.joinpath(f"llb-ensemble_{experiment_name}.pdf"),
)
