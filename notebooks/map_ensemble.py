# %load_ext autoreload
# %autoreload 2
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from core.map import MapEnsemble
from core.preprocessing import preprocess_create_x_train_test
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
_x_train, y_train = create_split_periodic_data(n_train=n_train)
x_train, _x_test, x_test = preprocess_create_x_train_test(_x_train)
y_test = ground_truth_periodic_function(_x_test)

layer_units = [500] * 4 + [1]
layer_activations = ["relu"] * 4 + ["linear"]


# %% codecell
fig, ax = plt.subplots()
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.3)
ax.scatter(_x_train, y_train, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
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
ensemble.fit(x_train=x_train, y_train=y_train, batch_size=batchsize_train)


# %% codecell
predictions = ensemble.predict(x_test)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.1)
for i, prediction in enumerate(predictions):
    ax.plot(_x_test, prediction, label=f"Model {i+1} prediction", alpha=0.8)
ax.scatter(_x_train, y_train, c="k", marker="x", s=100, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim([-5, 5])
ax.legend()
# fig.savefig(os.path.join(figure_dir, f"{n_networks}_ml_ensemble.pdf"))
