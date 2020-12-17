# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler

from core.map import MapDensityNetwork
from core.network_utils import prior_scale_to_regularization_lambda
from core.plotting_utils import (
    plot_distribution_samples,
    plot_ground_truth,
    plot_moment_matched_predictive_normal_distribution,
    plot_training_data,
)
from core.preprocessing import create_x_plot
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions


# %% codecell
np.random.seed(0)
n_train = 90
batch_size = n_train
epochs = 10000

x_train, y_train = create_split_periodic_data_heteroscedastic(
    n_data=n_train, sigma1=0.1, sigma2=0.8, seed=42
)
x_plot = create_x_plot(x_train)
# preprocessor = StandardPreprocessor()
# x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(_x_train)
y_ground_truth = ground_truth_periodic_function(x_plot)

input_shape = [1]
layer_units = [50, 20] + [2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]

weight_prior = tfd.Normal(0, 1)
bias_prior = weight_prior

# %% codecell
y_lim = [-5, 7]
fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(x_plot, y_ground_truth, fig=fig, ax=ax)
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
    weight_prior=weight_prior,
    bias_prior=bias_prior,
    n_train=n_train,
    learning_rate=0.01,
)

prior_predictive_distributions = net.predict_with_prior_samples(x_plot, n_samples=4)

plot_distribution_samples(
    x_plot=x_plot,
    distribution_samples=prior_predictive_distributions,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    # y_lim=[-30, 30],
)

# %%
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=20, verbose=1, restore_best_weights=False
)

net.fit(
    x_train=x_train,
    y_train=y_train,
    batch_size=batch_size,
    epochs=epochs,
    early_stop_callback=early_stop_callback,
    verbose=0,
)

prediction = net.predict(x_plot)
plot_moment_matched_predictive_normal_distribution(
    x_plot=x_plot,
    predictive_distribution=prediction,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
)
print(f"Total training epochs: {net.total_epochs}")


# %% markdown
# You can also train wiht homoscedastic noise by passing a float for
# initial_unconstrained_sigma .

# %%
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, verbose=1, restore_best_weights=True
)

initial_learning_rate = 0.05
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
)
layer_units[-1] = 1
net = MapDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=-2,  # it is usually beneficial to initialize this in the negative range.
    # This way the initial noise variance will be small and the network will be encouraged to fit the data
    # instead of explaining everything as noise.
    transform_unconstrained_scale_factor=0.5,
    weight_prior=weight_prior,
    bias_prior=bias_prior,
    n_train=n_train,
    preprocess_x=True,
    preprocess_y=True,
    learning_rate=0.01,
)
layer_units[-1] = 2


# %%
net.fit(
    x_train=x_train,
    y_train=y_train,
    batch_size=batch_size,
    epochs=epochs,
    early_stop_callback=early_stop_callback,
    validation_split=0.2,
    verbose=0,
)


prediction = net.predict(x_plot)
plot_moment_matched_predictive_normal_distribution(
    x_plot=x_plot,
    predictive_distribution=prediction,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
)
