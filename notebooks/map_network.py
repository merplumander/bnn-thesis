# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from core.map import MapNetwork
from core.plotting_utils import (
    plot_ground_truth,
    plot_moment_matched_predictive_normal_distribution,
    plot_training_data,
)
from core.preprocessing import StandardPreprocessor
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions


# %% codecell
np.random.seed(0)
n_train = 150
batchsize_train = 20

# train and test variables beginning with an underscore are unprocessed.
_x_train, y_train = create_split_periodic_data_heteroscedastic(n_data=n_train, seed=42)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(
    _x_train, test_ds=0.5
)
y_ground_truth = ground_truth_periodic_function(_x_plot)


layer_units = [20, 20] + [1]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]


# %% codecell
y_lim = [-5, 5]
fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(_x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(_x_plot, y_ground_truth, fig=fig, ax=ax)
ax.legend()


# %%
initial_learning_rate = 0.05
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
)


# %% codecell
net = MapNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
)


# %% codecell
net.fit(
    x_train=x_train, y_train=y_train, batch_size=batchsize_train, epochs=500, verbose=0
)

# %%
# prediction = net.predict_mean_function(x_plot) # getting predictive mean as array
predictive_normal_distribution = net.predict(
    x_plot
)  # getting predictive normal distribution with estimated sample standard deviation as tf distribution
predictive_delta_distribution = net.predict_delta_distribution(x_plot)

plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=predictive_delta_distribution,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    show_hdr=False,
    y_lim=y_lim,
    title=f"Predictive delta distribution",
)

plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=predictive_normal_distribution,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    title=f"Predictive normal distribution (Std defined by residuals)",
)
