# %load_ext autoreload
# %autoreload 2
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from core.hmc import (
    HMCDensityNetwork,
    hmc_density_network_from_save_path,
    mask_nonsense_chains,
)
from core.map import MapDensityNetwork
from core.network_utils import (
    batch_repeat_matrix,
    make_independent_gaussian_network_prior,
    transform_unconstrained_scale,
)
from core.plotting_utils import (
    plot_distribution_samples,
    plot_moment_matched_predictive_normal_distribution,
)
from core.preprocessing import StandardPreprocessor
from core.sanity_checks import check_posterior_equivalence
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions


# %%
data_seed = 0
n_train = 20

# train and test variables beginning with an underscore are unprocessed.
_x_train, _y_train = create_split_periodic_data_heteroscedastic(
    n_data=n_train,
    lower1=-1,
    upper1=0,
    lower2=1,
    upper2=2,
    sigma1=0.2,
    sigma2=0.2,
    seed=42,
)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(
    _x_train, test_ds=0.5, n_test=200
)
_y_ground_truth = ground_truth_periodic_function(_x_plot)
y_preprocessor = StandardPreprocessor()
y_train = y_preprocessor.fit_transform(_y_train)
y_ground_truth = y_preprocessor.transform(_y_ground_truth)

# %% markdown
# If we simply used two neurons in the output layer we can model heteroscedastic noise

# %%
input_shape = [1]
layer_units = [20, 10, 2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]
transform_unconstrained_scale_factor = 0.5

weight_prior = tfd.Normal(0, 1)
bias_prior = weight_prior
network_prior = make_independent_gaussian_network_prior(
    input_shape=input_shape, layer_units=layer_units, loc=0.0, scale=1.0
)


# %% markdown
# ### Let's first train a map network

# %%
net = MapDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    weight_prior=weight_prior,
    bias_prior=bias_prior,
    n_train=n_train,
    preprocess_y=False,
    learning_rate=0.01,
)
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=20, verbose=1, restore_best_weights=False
)

net.fit(
    x_train=x_train,
    y_train=y_train,
    batch_size=10,
    epochs=10000,
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
    y_lim=[-5, 5],
)
prediction = net.predict(x_train)
tf.reduce_sum(prediction.log_prob(y_train))
net.noise_sigma


# %% markdown
# ### And now towards the HMC network

# %%
n_chains = 128
num_burnin_steps = 100
num_results = 100

_step_size = 0.01
step_size = [np.ones((n_chains, 1, 1)) * _step_size] * (
    len(layer_units) * 2
)  # Individual step sizes for all chains and all layers
# step_size = np.ones((n_chains, 1, 1)) * _step_size # Individual step sizes for all chains
hmc_net = HMCDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    network_prior=network_prior,
    sampler="hmc",
    step_size_adapter="dual_averaging",
    num_burnin_steps=num_burnin_steps,
    step_size=step_size,
    num_leapfrog_steps=100,
    max_tree_depth=10,
    seed=0,
)


prior_samples = [hmc_net.sample_prior_state(seed=seed) for seed in [0, 1, 2, 3, 4]]
prior_predictions = [
    hmc_net.predict_from_sample_parameters(x_plot, prior_sample)
    for prior_sample in prior_samples
]
plot_distribution_samples(
    x_plot=x_plot,
    distribution_samples=prior_predictions,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
)


# %% markdown
# As a sanity check we can assert the posterior equivalence between a map network and an
# hmc network. This checks that the prior and likelihood are equivalent.

# %%
assert check_posterior_equivalence(net, hmc_net, x_train, y_train)


# %%
# How to prepare MAP weights for HMC:
weights = net.get_weights()
weights = [batch_repeat_matrix(w, repeats=(n_chains)) for w in weights]
# current_state = weights


# Using prior samples as starting points
prior_samples = hmc_net.sample_prior_state(n_samples=n_chains, overdisp=2, seed=0)
current_state = prior_samples
# current_state[-1] = tf.fill((n_chains, 1, 1), -5.0) # Sets the initial standard deviation to small values.
# transform_unconstrained_scale(
#     current_state[-1], factor=transform_unconstrained_scale_factor
# )

# %%
hmc_net.fit(x_train, y_train, current_state=current_state, num_results=num_results)

# %% markdown
# When using only few samples in the MCMC Chains and especially if we start from random overdispersed samples from the prior many chains will not have converged yet. We can try to find chains with nonsense samples and mask them out. The masking criteria so far are:

# 1: Acceptance ratios: We are aiming for an acceptance ratio of 0.75. When the acceptance ratio of a chain is outside the interval [06, 0.9] then something probably went wrong and we are going to exclude it.
# 2: Unnormalized log posterior values: A converged chain should have some samples in regions of high (unnormalized) posterior probability. If there is a chain whose sample with highest posterior probability is below the highest of the minimal posterior probabilities across chains, then we exclude it, since apparently it hasn't found a region of high posterior probability yet (as is expected for some chains when we are using a small number of samples and are starting from random locations).


# %%
mask_nonsense_chains(
    hmc_net,
    median_scale_cutter=None,
    lowest_acceptance_ratio=0.6,
    highest_acceptance_ratio=0.9,
    x_train=x_train,
    y_train=y_train,
)
hmc_net.samples[0].shape[:2]

# %% markdown
# With this masking we can get a good prediction even if we only use a few samples per chain. Had we not done any masking the prediction below would look very nonsensical.

# %%
posterior_prediction = hmc_net.predict(x_plot, thinning=10)
plot_moment_matched_predictive_normal_distribution(
    x_plot=x_plot,
    predictive_distribution=posterior_prediction,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    no_ticks=False,
    y_lim=[-5, 5],
)


# %% markdown
# Resuming running the chains is also possible:

# %%
hmc_net.fit(x_train, y_train, num_results=1000, resume=True)
hmc_net.samples[0].shape

# %% markdown
# Now that we have more samples per chain, fewer chains have to be excluded based on the criteria defined above

# %%
mask_nonsense_chains(
    hmc_net,
    median_scale_cutter=None,
    lowest_acceptance_ratio=0.6,
    highest_acceptance_ratio=0.9,
    x_train=x_train,
    y_train=y_train,
)
hmc_net.samples[0].shape[:2]

# %%
posterior_prediction = hmc_net.predict(x_plot, thinning=20)
plot_moment_matched_predictive_normal_distribution(
    x_plot=x_plot,
    predictive_distribution=posterior_prediction,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    no_ticks=False,
    y_lim=[-5, 5],
)


# %% markdown
# And we can also get predictions from random samples of the chains

# %%
sample_predictions = [
    hmc_net.predict_random_sample(x_plot, seed=seed) for seed in [0, 1, 3, 4]
]
plot_distribution_samples(
    x_plot=x_plot,
    distribution_samples=sample_predictions,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
)
