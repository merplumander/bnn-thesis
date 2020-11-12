# %load_ext autoreload
# %autoreload 2
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn import preprocessing

from core.hmc import HMCNetwork
from core.map import MapDensityNetwork, MapNetwork
from core.network_utils import (
    build_keras_model,
    prior_scale_to_regularization_lambda,
    regularization_lambda_to_prior_scale,
    transform_unconstrained_scale,
)
from core.preprocessing import StandardPreprocessor
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions

# %%
n_train = 20
_x_train, y_train = create_split_periodic_data_heteroscedastic(
    n_data=n_train, sigma1=2, sigma2=2, seed=42
)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(_x_train)
y_ground_truth = ground_truth_periodic_function(_x_plot)


# %% codecell
input_shape = [1]
layer_units = [30, 20, 1]  # [50, 30, 20, 10, 2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]


# %%
batch_size = n_train
weight_prior_scale = 2
bias_prior_scale = weight_prior_scale
weight_prior = tfd.Normal(0, weight_prior_scale)
bias_prior = tfd.Normal(0, bias_prior_scale)
weight_priors = [weight_prior] * len(layer_units)
bias_priors = [bias_prior] * len(layer_units)


# %%
seed = 0
model = MapDensityNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=0.0,
    weight_prior=weight_prior,
    bias_prior=bias_prior,
    scale_prior=tfd.InverseGamma(0.1, 0.1),
    n_train=n_train,
    seed=seed,
)

models = []
n_models = 4
seeds = np.arange(n_models)
initial_unconstrained_scales = seeds + 0.1
for seed, initial_unconstrained_scale in zip(seeds, initial_unconstrained_scales):

    m = MapDensityNetwork(
        input_shape=[1],
        layer_units=layer_units,
        layer_activations=layer_activations,
        initial_unconstrained_scale=initial_unconstrained_scale,
        weight_prior=weight_prior,
        bias_prior=bias_prior,
        scale_prior=tfd.InverseGamma(0.1, 0.1),
        n_train=n_train,
        # l2_weight_lambda=l2_weight_lambda,
        # l2_bias_lambda=l2_bias_lambda,
        seed=seed,
    )
    models.append(m)

model.fit(x_train=x_train, y_train=y_train, batch_size=20, epochs=10, verbose=0)
# %%
def map_network_likelihood_loss(net, x_train, y_train):
    loss = tf.reduce_mean(
        MapDensityNetwork.negative_log_likelihood(y_train, net.network(x_train))
    )
    return loss


def map_network_prior_loss(net):
    return sum(net.network.losses)


def map_network_loss(net, x_train, y_train):
    loss_by_hand = map_network_likelihood_loss(
        net, x_train, y_train
    ) + map_network_prior_loss(net)
    loss_by_tf = net.network.evaluate(x_train, y_train, verbose=0)
    assert np.isclose(loss_by_hand, loss_by_tf)
    return loss_by_hand


# %%
def negative_log_likelihood(y, p_y):
    return -p_y.log_prob(y)


# %%
map_network_prior_loss(model)


# %%
model.network.losses

# %%
w = model.get_weights()
loss = 0
for _w, _b in zip(w[0:-1:2], w[1::2]):
    # print(_w)
    print(tf.reduce_sum(_w))
    # print(_b)
    _loss = -tf.reduce_sum(weight_prior.log_prob(_w))
    print(_loss)
    loss += _loss
    _loss = -tf.reduce_sum(bias_prior.log_prob(_b))
    # print(_loss)
    loss += _loss
print(loss)
# %%
hmc_net = HMCNetwork(
    [1],
    layer_units,
    layer_activations,
    weight_priors=weight_priors,
    bias_priors=bias_priors,
    std_prior=tfd.InverseGamma(0.1, 0.1),
    seed=0,
)
hmc_log_posterior_fn = hmc_net._target_log_prob_fn_factory(x_train, y_train)


# %%
for model, initial_unconstrained_scale in zip(models, initial_unconstrained_scales):
    hmc_neg_log_likelihood = -hmc_net.log_likelihood(
        model.get_weights()[:-1],
        np.float32(transform_unconstrained_scale(initial_unconstrained_scale)),
        x_train,
        y_train,
    )
    hmc_neg_log_prior = -hmc_net.log_prior(
        model.get_weights()[:-1],
        np.float32(transform_unconstrained_scale(initial_unconstrained_scale)),
    )
    state = model.get_weights()
    hmc_neg_log_posterior = -hmc_log_posterior_fn(*state)
    map_likelihood_loss = map_network_likelihood_loss(model, x_train, y_train)
    map_prior_loss = map_network_prior_loss(model)
    map_loss = map_network_loss(model, x_train, y_train)
    map_loss_first = map_network_loss(model, x_train[0:10], y_train[0:10])
    map_loss_second = map_network_loss(model, x_train[10:], y_train[10:])
    assert np.isclose(
        (map_loss_first + map_loss_second) * (n_train / 2), map_loss * n_train
    )

    print(
        "The posteriors should be equal (up to the multiplicative constant of n_train)"
    )
    print(hmc_neg_log_posterior - (map_likelihood_loss + map_prior_loss) * n_train)
    print()
