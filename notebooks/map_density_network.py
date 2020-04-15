# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from core.map import MapDensityNetwork
from core.plotting_utils import plot_predictive_distribution
from core.preprocessing import preprocess_create_x_train_test
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions


# %% codecell
np.random.seed(0)
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
net = MapDensityNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
)


# %% codecell
net.fit(
    x_train=x_train, y_train=y_train, batch_size=batchsize_train, epochs=120, verbose=0
)

# %%
prediction = net.predict(x_test)
plot_predictive_distribution(
    x_test=_x_test,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-6, 6],
)
