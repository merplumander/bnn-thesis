# %% markdown


# ## Approximating a Bayesian Neural Network by a Maxium A Posteriori Approximation.
# In the fully general case, where you do not know the standard deviation, you don't get
# aroung modelling the standard deviation in the MAP network.
# Let's start with some code showing those results.


# %%
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
layer_units = [3, 2, 1]  # [50, 30, 20, 10, 2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]


# %%
batch_size = n_train
weight_prior_scale = 2
bias_prior_scale = weight_prior_scale
weight_priors = [tfd.Normal(0, weight_prior_scale)] * len(layer_units)
bias_priors = [tfd.Normal(0, bias_prior_scale)] * len(layer_units)
l2_weight_lambda = prior_scale_to_regularization_lambda(weight_prior_scale, n_train)
l2_bias_lambda = prior_scale_to_regularization_lambda(bias_prior_scale, n_train)
assert np.isclose(
    weight_prior_scale, regularization_lambda_to_prior_scale(l2_weight_lambda, n_train)
)

# %%
seed = 0
model = MapDensityNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_sigma=0.0,
    l2_weight_lambda=l2_weight_lambda,
    l2_bias_lambda=l2_bias_lambda,
    seed=seed,
)

models = []
n_models = 4
seeds = np.arange(n_models)
initial_unconstrained_sigmas = seeds + 0.1
for seed, initial_unconstrained_sigma in zip(seeds, initial_unconstrained_sigmas):

    m = MapDensityNetwork(
        input_shape=[1],
        layer_units=layer_units,
        layer_activations=layer_activations,
        initial_unconstrained_sigma=initial_unconstrained_sigma,
        l2_weight_lambda=l2_weight_lambda,
        l2_bias_lambda=l2_bias_lambda,
        seed=seed,
    )
    models.append(m)


# %%
def map_network_likelihood_loss(net, x_train, y_train):
    return net.network.loss_functions[0](y_train, net.network(x_train))


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
_l = 0
for _x, _y in zip(x_train, y_train):
    _e = model.evaluate(_x, _y, verbose=0)
    _l += _e
    _likelihood = map_network_likelihood_loss(model, _x.reshape((1, 1)), _y)
    _prior = map_network_prior_loss(model)
    assert np.isclose(_likelihood + _prior, _e)

model.evaluate(x_train, y_train, batch_size=n_train, verbose=0)
assert np.isclose(
    _l / n_train, model.evaluate(x_train, y_train, batch_size=n_train, verbose=0)
)
prior_losses = sum(model.network.losses)
assert np.isclose(
    tf.math.reduce_mean(negative_log_likelihood(y_train, model(x_train)))
    + prior_losses,
    _l / n_train,
)


# %%
hmc_net = HMCNetwork(
    [1],
    layer_units,
    layer_activations,
    weight_priors=weight_priors,
    bias_priors=bias_priors,
    # std_prior=tfd.Normal(0.3, 0.01),
    seed=0,
)
hmc_log_posterior_fn = hmc_net._target_log_prob_fn_factory(x_train, y_train)


# %%
for model, initial_unconstrained_sigma in zip(models, initial_unconstrained_sigmas):
    hmc_neg_log_likelihood = -hmc_net.log_likelihood(
        model.get_weights()[:-1],
        np.float32(transform_unconstrained_scale(initial_unconstrained_sigma)),
        x_train,
        y_train,
    )
    hmc_neg_log_prior = -hmc_net.log_prior(model.get_weights()[:-1])
    state = model.get_weights()[:-1]
    state.insert(
        0, np.float32(transform_unconstrained_scale(initial_unconstrained_sigma))
    )
    hmc_neg_log_posterior = -hmc_log_posterior_fn(*state)
    map_likelihood_loss = map_network_likelihood_loss(model, x_train, y_train)
    map_prior_loss = map_network_prior_loss(model)
    map_loss = map_network_loss(model, x_train, y_train)
    map_loss_first = map_network_loss(model, x_train[0:10], y_train[0:10])
    map_loss_second = map_network_loss(model, x_train[10:], y_train[10:])
    assert np.isclose(
        (map_loss_first + map_loss_second) * (n_train / 2), map_loss * n_train
    )
    # print("The likelihoods should be equal")
    # print("hmc neg log likelihood", hmc_neg_log_likelihood)
    # print("map likelihood loss", map_likelihood_loss * n_train)
    # print()
    #
    # print("hmc neg log prior", hmc_neg_log_prior)
    # print("map prior loss", map_prior_loss)
    # print(hmc_neg_log_prior - map_prior_loss)
    # print()

    print(
        "The posteriors should be equal up to an additive constant (and a multiplicative constant of n_train)"
    )
    print(hmc_neg_log_posterior - (map_likelihood_loss + map_prior_loss) * n_train)
    print()

# %% markdown
# ## Knowing the noise standard deviation $\sigma$
# If you want to use a Gaussian prior on the weights (and biases) w with standard deviation
# $\rho$ or precision $\beta=\frac{1}{\rho^2}$ , you'll need to
# figure out how to choose the L2 regularization $\lambda$ hyperparameter.

# If we assume we know the Gaussian noise's standard deviation $\sigma$ or precision
# $\alpha = \frac{1}{\sigma^2}$, we can derive
# the regularization $\lambda$.

# The log posterior is: $-\frac{\beta}{2} sos  - \frac{\alpha}{2} w^T w + const$
# where $sos$ is the sum of squared errors.

# The regularized loss for a MAP network ($map loss$) is: $sos + \lambda w^T w$

# $ - log posterior \frac{2}{\beta} = sos  + \lambda w^T w  - \frac{2 const}{\beta}$
# where we have chosen $\lambda = \frac{\alpha}{\beta}$.

# $ - log posterior \frac{2}{\beta} = map loss - \frac{2 const}{\beta}$

# $ log posterior =  - \frac{\beta}{2} map loss + const$

# So we see that maximizing the log posterior is equal to minimizing the map loss since
# neither scaling by a constant nor adding a constant changes the location of the local
# optima.

# Phrasing $\lambda$ in terms of the standard deviations would yield: $\lambda = \frac{\rho^2}{\sigma^2}$.


# %%
epochs = 10
seed = 0
noise_sigma = 2
noise_beta = 1 / (noise_sigma ** 2)


weight_prior_scale = 2
bias_prior_scale = weight_prior_scale
weight_priors = [tfd.Normal(0, weight_prior_scale)] * len(layer_units)
bias_priors = [tfd.Normal(0, bias_prior_scale)] * len(layer_units)
alpha = 1 / (weight_prior_scale ** 2)

weight_lambda = alpha / noise_beta
bias_lambda = alpha / noise_beta

# %%
def map_network_likelihood_loss(net, x_train, y_train):
    return net.network.loss_functions[0](y_train, net.network(x_train))


def map_network_prior_loss(net, x_train, y_train):
    return sum(net.network.losses)


def map_network_loss(net, x_train, y_train):
    loss_by_hand = map_network_likelihood_loss(
        net, x_train, y_train
    ) + map_network_prior_loss(net, x_train, y_train)
    loss_by_tf = net.network.evaluate(x_train, y_train, verbose=0)
    assert np.isclose(loss_by_hand, loss_by_tf)
    return loss_by_hand


# %%
hmc_net = HMCNetwork(
    [1],
    layer_units,
    layer_activations,
    weight_priors=weight_priors,
    bias_priors=bias_priors,
    # std_prior=tfd.Normal(0.3, 0.01),
    seed=0,
)
hmc_log_posterior_fn = hmc_net._target_log_prob_fn_factory(x_train, y_train)
# %%


# %%
nets = []
for seed in range(4):
    net = MapNetwork(
        input_shape=input_shape,
        layer_units=layer_units,
        layer_activations=layer_activations,
        l2_weight_lambda=weight_lambda,
        l2_bias_lambda=bias_lambda,
        seed=seed,
        loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
    )
    nets.append(net)

# %%
net = nets[0]
state = net.get_weights()
state.insert(0, noise_sigma)
hmc_log_posterior = hmc_log_posterior_fn(*state)
first_term = -(noise_beta / 2) * map_network_likelihood_loss(net, x_train, y_train)
second_term = -(alpha / 2) * map_network_prior_loss(net, x_train, y_train)
const = hmc_log_posterior - (first_term + second_term)

# %%
for net in nets[1:]:
    state = net.get_weights()
    state.insert(0, noise_sigma)
    hmc_log_posterior = hmc_log_posterior_fn(*state)
    map_loss = map_network_loss(net, x_train, y_train)
    assert np.isclose(hmc_log_posterior, -map_loss * (noise_beta / 2) + const)


# %%
for net in nets[1:]:
    state = net.get_weights()
    state.insert(0, noise_sigma)
    print("hmc log posterior", hmc_log_posterior_fn(*state))
    first_term = -(noise_beta / 2) * map_network_likelihood_loss(net, x_train, y_train)
    second_term = -(alpha / 2) * map_network_prior_loss(net, x_train, y_train)
    print(first_term + second_term + const)

# %%
net = MapNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    l2_weight_lambda=weight_lambda,
    l2_bias_lambda=bias_lambda,
    seed=seed,
    loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
)
print("Likelihood and prior loss combined", map_network_loss(net, x_train, y_train))

# %% markdown
# ## Ensuring full understanding of tensorflow's way of implementing L2 loss and how it realtes to a gaussian prior on the weights.

# ### For one layer's weights the following holds as a relationship between l2 loss and a Gaussian prior:

# $c = N  * log(sigma) + N/2 * log(2 * pi)$

# l2 loss $ + c = $negative log prior

# where N is the number of weights and sigma is the scale of the normal distribution


# %%
def l2_loss(weights_list, weight_lambda, bias_lambda):
    loss = 0
    for w, b in zip(weights_list[::2], weights_list[1::2]):
        loss += weight_lambda * tf.reduce_sum(w ** 2)
        # print(weight_lambda * tf.reduce_sum(w ** 2))
        loss += bias_lambda * tf.reduce_sum(b ** 2)
        # print(bias_lambda * tf.reduce_sum(b ** 2))
    return loss


# %%
net.network.losses
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
