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
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions

figure_dir = "figures/non-linear_gap_uncertainty"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)
experiment_name = "non-linear-gap"

# %%
seed = 0
n_train = 50

# train and test variables beginning with an underscore are unprocessed.
amplitude = 1
p = 2
_x_train, y_train = create_split_periodic_data_heteroscedastic(
    n_data=n_train,
    lower1=-4.0,
    upper1=-1.3,
    lower2=4.0,
    upper2=7.0,
    sigma1=0.1,
    sigma2=0.1,
    p=p,
    amplitude=amplitude,
    seed=seed,
)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(
    _x_train, test_ds=0.03
)
y_ground_truth = ground_truth_periodic_function(_x_plot, p=p, amplitude=amplitude)

layer_units = [50, 20] + [2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]

# %% codecell
experiment_name = "nonlinear-gap"
y_lim = [-5, 5]

fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(_x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(_x_plot, y_ground_truth, fig=fig, ax=ax, alpha=0.2)
ax.legend()
fig.savefig(figure_dir.joinpath(f"{experiment_name}_data.pdf"))

# %%
# General training
epochs = 250
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
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    # save_path=figure_dir.joinpath(f"map_network_{experiment_name}.pdf")
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
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    # save_path=figure_dir.joinpath(f"ensemble_moment_matched_{experiment_name}.pdf")
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
    # save_path=figure_dir.joinpath(f"ensemble_members_{experiment_name}.pdf")
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
    seed=seed,
)
layer_units[-1] = 2

llb_net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)

# %%
prediction = llb_net.predict(x_plot)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    # save_path=figure_dir.joinpath(f"llb_moment_matched_{experiment_name}.pdf")
)
