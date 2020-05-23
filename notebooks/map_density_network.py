# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from core.map import MapDensityNetwork
from core.plotting_utils import plot_distribution_samples, plot_predictive_distribution
from core.preprocessing import preprocess_create_x_train_test
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions


# %% codecell
np.random.seed(0)
n_train = 20
batch_size = 20
epochs = 150

# train and test variables beginning with an underscore are unprocessed.
_x_train, y_train = create_split_periodic_data_heteroscedastic(
    n_train=n_train, sigma1=0.3, sigma2=0.5, seed=42
)
x_train, _x_test, x_test = preprocess_create_x_train_test(_x_train)
y_test = ground_truth_periodic_function(_x_test)

input_shape = [1]
layer_units = [100, 50, 20, 10] + [2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]
# values behind the "#" seem to be reasonable for a prior distribution,
# but in practice we need much lower values for training to succeed
l2_weight_lambda = 0.0000001  # 2
l2_bias_lambda = 0.000000001  # 5


# %% codecell
fig, ax = plt.subplots()
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.1)
ax.scatter(_x_train, y_train, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
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


prior_predictive_distributions = net.predict_with_prior_samples(x_test, n_samples=4)

plot_distribution_samples(
    x_test=_x_test,
    distribution_samples=prior_predictive_distributions,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    # y_lim=[-30, 30],
)


# %% codecell
net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


prediction = net.predict(x_test)
plot_predictive_distribution(
    x_test=_x_test,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-6, 6],
)
