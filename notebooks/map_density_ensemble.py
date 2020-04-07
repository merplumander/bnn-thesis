# %load_ext autoreload
# %autoreload 2
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn import preprocessing

from core.map import MapDensityEnsemble
from core.preprocessing import preprocess_create_x_train_test
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions


# %% codecell
assert tf.executing_eagerly()

figure_dir = "./figures"


# %% codecell
np.random.seed(0)
n_networks = 5
n_train = 20
batchsize_train = 20

# train and test variables beginning with an underscore are unprocessed.
_x_train, y_train = create_split_periodic_data_heteroscedastic(n_train=n_train, seed=42)
x_train, _x_test, x_test = preprocess_create_x_train_test(_x_train)
y_test = ground_truth_periodic_function(_x_test)

layer_units = [500] * 4 + [2]
layer_activations = ["relu"] * 4 + ["linear"]


# %% codecell
fig, ax = plt.subplots()
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.1)
ax.scatter(_x_train, y_train, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
ax.legend()


# %%
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
)


# %% codecell
ensemble = MapDensityEnsemble(
    n_networks=n_networks,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
)


# %% codecell
ensemble.fit(x_train=x_train, y_train=y_train, batch_size=batchsize_train, epochs=120)


# %% codecell
mean, std = ensemble.predict(x_test)
mean = mean.numpy()
std = std.numpy()

# %%

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
ax.set_ylim([-5, 5])
ax.legend()
# fig.savefig(os.path.join(figure_dir, f"{n_networks}_ml_density_ensemble_gaussian_heteroscedastic.pdf"))


# %% codecell
means, stds = ensemble.predict_mixture_of_gaussian(x_test)

fig, ax = plt.subplots(figsize=(8, 8))
for i, (mean, std) in enumerate(zip(means, stds)):
    c = sns.color_palette()[i]
    ax.plot(_x_test, mean, label=f"Mean prediction", c=c, alpha=0.8)
    ax.fill_between(
        _x_test.flatten(),
        mean.flatten() - 2 * std.flatten(),
        mean.flatten() + 2 * std.flatten(),
        color=c,
        alpha=0.2,
        label="95% HDR prediction",
    )
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.1)
ax.scatter(_x_train, y_train, c="k", marker="x", s=100, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim([-5, 5])
ax.legend()
# fig.savefig(os.path.join(figure_dir, f"{n_networks}_ml_density_ensemble_mixture_of_gaussian_heteroscedastic.pdf"))


# %% codecell
mean, std = ensemble.predict(x_test)
mean = mean.numpy()
std = std.numpy()

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
means, stds = ensemble.predict_mixture_of_gaussian(x_test)
c = sns.color_palette()[1]
for i, (mean, std) in enumerate(zip(means[:-1], stds[:-1])):
    ax.plot(_x_test, mean, c=c, alpha=0.5)
ax.plot(_x_test, means[-1], label=f"Function samples", c=c, alpha=0.5)
ax.scatter(_x_train, y_train, c="k", marker="x", s=100, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim([-5, 5])
ax.legend()
# fig.savefig(os.path.join(figure_dir, f"{n_networks}_ml_density_ensemble_gaussian_samples_heteroscedastic.pdf"))
