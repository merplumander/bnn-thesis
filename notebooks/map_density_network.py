# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from core.map import MapDensityNetwork
from core.plotting_utils import (
    plot_distribution_samples,
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
n_train = 20
batch_size = n_train
epochs = 150

# train and test variables beginning with an underscore are unprocessed.
_x_train, y_train = create_split_periodic_data_heteroscedastic(
    n_data=n_train, sigma1=0.3, sigma2=0.5, seed=42
)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(_x_train)
y_ground_truth = ground_truth_periodic_function(_x_plot)

input_shape = [1]
layer_units = [100, 50, 20, 10] + [2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]
# values behind the "#" seem to be reasonable for a prior distribution,
# but in practice we need much lower values for training to succeed
l2_weight_lambda = 0.00001  # 2
l2_bias_lambda = 0.0000001  # 5


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


net = MapDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    l2_weight_lambda=l2_weight_lambda,
    l2_bias_lambda=l2_bias_lambda,
    learning_rate=lr_schedule,
)


prior_predictive_distributions = net.predict_with_prior_samples(x_plot, n_samples=4)

plot_distribution_samples(
    x_plot=_x_plot,
    distribution_samples=prior_predictive_distributions,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    # y_lim=[-30, 30],
)


# %% codecell
net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


prediction = net.predict(x_plot)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
)
