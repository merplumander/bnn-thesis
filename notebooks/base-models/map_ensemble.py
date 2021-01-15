# %load_ext autoreload
# %autoreload 2
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from core.map import MapEnsemble
from core.plotting_utils import (
    plot_ground_truth,
    plot_moment_matched_predictive_normal_distribution,
    plot_moment_matched_predictive_normal_distribution_and_function_samples,
    plot_training_data,
)
from core.preprocessing import StandardPreprocessor
from data.toy_regression import (
    create_split_periodic_data,
    ground_truth_periodic_function,
)

# %% codecell
assert tf.executing_eagerly()

figure_dir = "./figures"


# %% codecell
np.random.seed(0)
n_networks = 5
n_train = 20
batchsize_train = 20


# train and test variables beginning with an underscore are unprocessed.
_x_train, y_train = create_split_periodic_data(n_data=n_train)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(
    _x_train, test_ds=0.5
)
y_ground_truth = ground_truth_periodic_function(_x_plot)

layer_units = [50, 20, 10] + [1]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]


# %% codecell
y_lim = [-5, 5]
fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(_x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(_x_plot, y_ground_truth, fig=fig, ax=ax)
ax.legend()


# %% codecell
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
)


# %% codecell
ensemble = MapEnsemble(
    n_networks=n_networks,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
)


# %% codecell
ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batchsize_train, epochs=250, verbose=0
)


# %% codecell
predictive_mixture_of_gaussians = ensemble.predict(x_plot)
predictive_mixture_of_deltas = ensemble.predict_mixture_of_deltas(x_plot)
predictive_list_of_deltas = ensemble.predict_list_of_deltas(x_plot, n_predictions=3)


plot_moment_matched_predictive_normal_distribution_and_function_samples(
    x_plot=_x_plot,
    predictive_distribution=predictive_mixture_of_gaussians,
    distribution_samples=predictive_list_of_deltas,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    title=f"Predictive mixture of gaussians",
)

# %%
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=predictive_mixture_of_deltas,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    title=f"Predictive mixture of gaussians",
)
