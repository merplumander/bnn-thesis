# %load_ext autoreload
# %autoreload 2
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn import preprocessing

from core.map import MapDensityNetwork
from core.network_utils import (
    build_keras_model,
    prior_scale_to_regularization_lambda,
    transform_unconstrained_scale,
)
from core.plotting_utils import (
    plot_distribution_samples,
    plot_predictive_distribution,
    plot_predictive_distribution_and_function_samples,
)
from core.preprocessing import preprocess_create_x_train_test
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions

# %%
n_train = 20
_x_train, y_train = create_split_periodic_data_heteroscedastic(
    n_train=n_train, sigma1=2, sigma2=2, seed=42
)
x_train, _x_test, x_test = preprocess_create_x_train_test(_x_train)
y_test = ground_truth_periodic_function(_x_test)


# %% codecell
input_shape = [1]
layer_units = [20, 10, 2]  # [50, 30, 20, 10, 2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]

# %%
epochs = 10
batch_size = n_train  # 10
seed = 0


# %%
net = MapDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    l2_weight_lambda=0.005,
    l2_bias_lambda=0.1,
    seed=seed,
)

net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)

# %% markdown
# ### For one layer's weights the following holds as a relationship between l2 loss and a Gaussian prior:

# $c = N  * log(sigma) + N/2 * log(2 * pi)$

# l2 loss $ + c = $negative log prior

# where N is the number of weights and sigma is the scale of the normal distribution


# %%
def l2_loss(weights_list, weight_lambda, bias_lambda):
    loss = 0
    for w, b in zip(weights_list[::2], weights_list[1::2]):
        loss += weight_lambda * tf.reduce_sum(w ** 2)
        loss += bias_lambda * tf.reduce_sum(b ** 2)
    return loss


# %%
l2_by_hand = l2_loss(net.get_weights(), net.l2_weight_lambda, net.l2_bias_lambda)
l2 = sum(net.network.losses)

assert np.isclose(l2, l2_by_hand)

negative_log_prior = net.negative_log_prior()

N = 0
for u1, u2 in zip(net.input_shape + net.layer_units[:-1], net.layer_units):
    N += u1 * u2


c_weights = N * np.log(np.sqrt(1 / (2 * net.l2_weight_lambda))) + N / 2 * np.log(
    2 * np.math.pi
)

N_biases = np.sum(layer_units)
c_biases = N_biases * np.log(
    np.sqrt(1 / (2 * net.l2_bias_lambda))
) + N_biases / 2 * np.log(2 * np.math.pi)
c = c_weights + c_biases
# print(l2, c_weights + c_biases, negative_log_prior)
assert np.isclose(l2_by_hand + c, negative_log_prior)
