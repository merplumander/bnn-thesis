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

figure_dir = "figures/outer_edges_ensemble_uncertainty"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)


# %%
data_seed = 1
n_train = 50


# train and test variables beginning with an underscore are unprocessed.
period = 3.3 * np.pi
_x_train, y_train = cos_linear_edge_data(
    n_data=n_train, period=period, sigma=0.1, seed=data_seed
)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(
    _x_train, test_ds=0.8
)
y_ground_truth = ground_truth_cos_function(_x_plot, period)

layer_units = [50, 20] + [2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]

# %%
experiment_name = "bending-edge"
y_lim = [-5, 5]

fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(_x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(_x_plot, y_ground_truth, fig=fig, ax=ax, alpha=0.2)
ax.legend()
fig.savefig(figure_dir.joinpath(f"{experiment_name}_data.pdf"))

# %%
# General training
train_seed = 0
epochs = 150
batch_size = n_train

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
    seed=train_seed,
)

ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


# %%
prediction = ensemble.predict(x_plot)  # Mixture Of Gaussian prediction
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    save_path=figure_dir.joinpath(f"{experiment_name}_ensemble.pdf"),
)


# %% codecell
gaussian_predictions = ensemble.predict_list_of_gaussians(x_plot, n_predictions=3)
plot_distribution_samples(
    x_plot=_x_plot,
    distribution_samples=gaussian_predictions,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    save_path=figure_dir.joinpath(f"{experiment_name}_ensemble_members.pdf"),
)


# %%
experiment_name = "linear-edge"
period = 4.7 * np.pi
_x_train, y_train = cos_linear_edge_data(
    n_data=n_train, period=period, sigma=0.1, seed=data_seed
)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(
    _x_train, test_ds=0.8
)
y_ground_truth = ground_truth_cos_function(_x_plot, period)

# %% codecell
fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(_x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(_x_plot, y_ground_truth, fig=fig, ax=ax, alpha=0.2)
ax.legend()
fig.savefig(figure_dir.joinpath(f"{experiment_name}_data.pdf"))


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
    seed=train_seed,
)

ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


# %%
prediction = ensemble.predict(x_plot)  # Mixture Of Gaussian prediction
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    save_path=figure_dir.joinpath(f"{experiment_name}_ensemble.pdf"),
)
