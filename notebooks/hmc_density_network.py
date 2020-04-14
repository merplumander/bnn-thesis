# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from core.hmc import HMCDensityNetwork
from core.map import MapDensityNetwork
from core.network_utils import (
    activation_strings_to_activation_functions,
    build_scratch_model,
)
from core.preprocessing import preprocess_create_x_train_test
from data.toy_regression import create_linear_data, ground_truth_linear_function

tfd = tfp.distributions


# %% codecell
n_train = 20
m = 1
b = 3
_x_train, y_train = create_linear_data(n_train, m=m, b=b, sigma=0.5)
x_train, _x_test, x_test = preprocess_create_x_train_test(_x_train)
y_test = ground_truth_linear_function(_x_test, m=m, b=b)


# %% codecell
input_shape = [1]
layer_units = [20, 10, 2]
layer_activations = ["relu", "relu", "linear"]
num_burnin_steps = 1000
num_results = 3000

# %% markdown
#### HMCDensityNetwork follows the fit and predict scheme:


# %%
hmc_net = HMCDensityNetwork(input_shape, layer_units, layer_activations, seed=0)

# %%
hmc_net.fit(
    x_train,
    y_train,
    num_burnin_steps=num_burnin_steps,
    num_results=num_results,
    verbose=0,
)


# %% markdown
# Standard prediction returns an equally weighted mixture of Gaussians. One Gaussian for each parameter setting in the chain.
# The Gaussians represent aleatoric uncertainty, while mixing these Gaussians as per the samples from the markov chain represents epistemic uncertainty.

# %%
mog_prediction = hmc_net.predict(x_test, thinning=3)  # Mixture Of Gaussian prediction


# %%
mean = mog_prediction.mean().numpy()
std = mog_prediction.stddev().numpy()
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.1)
ax.plot(_x_test, mean, label=f"Mean prediction", alpha=0.8)
ax.fill_between(
    _x_test.flatten(),
    mean.flatten() - 2 * std.flatten(),
    mean.flatten() + 2 * std.flatten(),
    alpha=0.2,
    label="95% HDR prediction",
)
ax.scatter(_x_train, y_train, c="k", marker="x", s=100, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim([-3, 12])
ax.legend()


# %% markdown
# Plotting individual network predictions from the chain is also possible:

# %% codecell
n_predictions = 3
gaussian_predictions = hmc_net.predict_list_of_gaussians(
    x_test, n_predictions=n_predictions
)

fig, ax = plt.subplots(figsize=(8, 8))
for i, prediction in enumerate(gaussian_predictions):
    mean = prediction.mean()
    std = prediction.stddev()
    c = sns.color_palette()[i]
    ax.plot(_x_test, mean, label=f"Mean prediction", c=c, alpha=0.8)
    ax.fill_between(
        _x_test.flatten(),
        tf.reshape(mean, [mean.shape[0]]) - 2 * tf.reshape(std, [std.shape[0]]),
        tf.reshape(mean, [mean.shape[0]]) + 2 * tf.reshape(std, [std.shape[0]]),
        color=c,
        alpha=0.15,
        label="95% HDR prediction",
    )
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.1)
ax.scatter(_x_train, y_train, c="k", marker="x", s=100, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim([-3, 12])
ax.legend()

# %% markdown
# Or plotting overall uncertainty and some function samples

# %%
n_predictions = 5
mog_prediction = hmc_net.predict(x_test)  # Mixture Of Gaussian prediction
mean = mog_prediction.mean().numpy()
std = mog_prediction.stddev().numpy()
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(_x_test, y_test, label="Ground truth", c="k", alpha=0.1)
c = sns.color_palette()[0]
ax.plot(_x_test, mean, label=f"Mean prediction", c=c, alpha=1)
ax.fill_between(
    _x_test.flatten(),
    mean.flatten() - 2 * std.flatten(),
    mean.flatten() + 2 * std.flatten(),
    color=c,
    alpha=0.2,
    label="95% HDR prediction",
)
gaussian_predictions = hmc_net.predict_list_of_gaussians(
    x_test, n_predictions=n_predictions
)
c = sns.color_palette()[1]
for i, prediction in enumerate(gaussian_predictions[:-1]):
    mean = prediction.mean()
    ax.plot(_x_test, mean, c=c, alpha=0.5)
ax.plot(
    _x_test, gaussian_predictions[-1].mean(), label=f"Function samples", c=c, alpha=0.5
)
ax.scatter(_x_train, y_train, c="k", marker="x", s=100, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim([-3, 12])
ax.legend()


# %%
# resuming running the chain does not work yet
# net.fit(x_train, y_train, resume=True, num_burnin_steps=0, num_results=30, verbose=0)


# %% markdown
# ## You can also pre-train a density network by hand as initial state
# (if you don't provide an initial state via current_state a density network will be pre trained with standard parameters)

# %%
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
)

map_net = MapDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
)

map_net.fit(x_train=x_train, y_train=y_train, batch_size=20, epochs=200, verbose=0)

# %%
prediction = map_net.predict(x_test)
mean = prediction.mean().numpy()
std = prediction.stddev().numpy()

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.1)
ax.plot(_x_test, mean, label=f"Mean prediction", alpha=0.8)
ax.fill_between(
    _x_test.flatten(),
    mean.flatten() - 2 * std.flatten(),
    mean.flatten() + 2 * std.flatten(),
    alpha=0.2,
    label="95% HDR prediction",
)
ax.scatter(_x_train, y_train, c="k", marker="x", s=100, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
ax.legend()

# %%
initial_state = map_net.get_weights()
# initial_state


# %%
hmc_net = HMCDensityNetwork(input_shape, layer_units, layer_activations, seed=0)
hmc_net.fit(
    x_train,
    y_train,
    current_state=initial_state,
    num_burnin_steps=1000,
    num_results=2000,
    verbose=0,
)


# %%
mog_prediction = hmc_net.predict(x_test, thinning=2)  # Mixture Of Gaussian prediction


# %%
mean = mog_prediction.mean().numpy()
std = mog_prediction.stddev().numpy()
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.1)
ax.plot(_x_test, mean, label=f"Mean prediction", alpha=0.8)
ax.fill_between(
    _x_test.flatten(),
    mean.flatten() - 2 * std.flatten(),
    mean.flatten() + 2 * std.flatten(),
    alpha=0.2,
    label="95% HDR prediction",
)
ax.scatter(_x_train, y_train, c="k", marker="x", s=100, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim([-3, 12])
ax.legend()
