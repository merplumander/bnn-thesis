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


# %%
input_shape = [1]
layer_units = [20, 10, 1]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]
transform_unconstrained_scale_factor = 0.5

weight_prior = tfd.Normal(0, 1)
bias_prior = weight_prior
network_prior = make_independent_gaussian_network_prior(
    input_shape=input_shape, layer_units=layer_units, loc=0.0, scale=1.0
)
noise_scale_prior = tfd.TransformedDistribution(
    distribution=tfd.InverseGamma([[0.5]], 0.01),
    bijector=tfp.bijectors.Invert(tfp.bijectors.Square()),
)  # Transforming an InverseGamma prior on the variance to a prior on the standard deviation.
# noise_scale_prior = tfd.Uniform(low=[[0.0]], high=2)

# %% markdown
# ### Let's first train a map network

# %%
net = MapDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=-1,
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    weight_prior=weight_prior,
    bias_prior=bias_prior,
    noise_scale_prior=noise_scale_prior,
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
    len(layer_units) * 2 + 1  # + 1 for the noise sigma "layer"
)  # Individual step sizes for all chains and all layers
# step_size = np.ones((n_chains, 1, 1)) * _step_size # Individual step sizes for all chains
hmc_net = HMCDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    network_prior=network_prior,
    noise_scale_prior=noise_scale_prior,
    sampler="hmc",
    step_size_adapter="dual_averaging",
    num_burnin_steps=num_burnin_steps,
    step_size=step_size,
    num_leapfrog_steps=100,
    max_tree_depth=10,
    num_steps_between_results=0,
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
# When using only few samples in the MCMC Chains and especially if we start from random overdispersed samples from the prior many chains will not have converged yet. We can try to find chains with nonsense samples and mask them out. So far there are three masking criteria:
# 1: median of the noise scales: Since we are using normalized y_train values, the maximum reasonable standard deviation we should expect is 1. I intentionally use a uniform prior between 0 and 2 and sort out all chains that have a median noise scale above one.
# 2: Acceptance ratios: We are aiming for an acceptance ratio of 0.75. When the acceptance ratio of a chain is outside the interval [06, 0.9] then something probably went wrong and we are going to exclude it.
# 3: Unnormalized log posterior values: A converged chain should have some samples in regions of high (unnormalized) posterior probability. If there is a chain whose sample with highest posterior probability is below the highest of the minimal posterior probabilities across chains, then we exclude it, since apparently it hasn't found a region of high posterior probability yet (as is expected for some chains when we are using a small number of samples and are starting from random locations).


# %%
mask_nonsense_chains(
    hmc_net,
    median_scale_cutter=1.0,
    lowest_acceptance_ratio=0.6,
    highest_acceptance_ratio=0.9,
    x_train=x_train,
    y_train=y_train,
    thinning=1,
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
# Fitting times (num_burnin: 100, num_results: 200, layer_units: [10, 5, 1], n_train: 6):
# 1 chain:    20s
# 128 chains: 24s
# 256 chains: 34s
# 512 chains: 51s


# %% markdown
# HMC networks can be saved and reloaded.

# %%
save_dir = ".hmc_save_dir/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir.joinpath("deletable_toy_hmc_notebook")
hmc_net.save(save_path)

# %%
hmc_net = hmc_density_network_from_save_path(
    save_path, network_prior=network_prior, noise_scale_prior=noise_scale_prior
)
hmc_net._chain[0].shape


# %% markdown
# Resuming running the chains is also possible:

# %%
hmc_net.fit(x_train, y_train, num_results=100, resume=True)
hmc_net.samples[0].shape

# %% markdown
# Now that we have more samples per chain, fewer chains have to be excluded based on the criteria defined above

# %%
mask_nonsense_chains(
    hmc_net,
    median_scale_cutter=1.0,
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
# Getting the predictive distribution of individual chains is quite easy

# %%
i_chains = [0, 1, 2]
chain_predictions = [
    hmc_net.predict_individual_chain(x_plot, i_chain=i_chain, thinning=10)
    for i_chain in i_chains
]
plot_distribution_samples(
    x_plot=x_plot,
    distribution_samples=chain_predictions,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
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


# %% markdown
# The HMC network also provides functions to compute potential scale reduction, effective sample size and acceptance ratios
# %%
reduction = hmc_net.potential_scale_reduction()
tf.nest.map_structure(lambda x: tf.reduce_max(x), reduction)

# %%
ess = hmc_net.effective_sample_size(thinning=10)
tf.nest.map_structure(
    lambda x: (
        int(tf.reduce_mean(x).numpy()),
        int(tf.math.reduce_std(x).numpy()),
        int(tf.reduce_min(x).numpy()),
        int(tf.reduce_max(x).numpy()),
    ),
    ess,
)


# %%
acceptance_ratio = hmc_net.acceptance_ratio()
acceptance_ratio


# %% markdown
# Finally we can look at the posterior distribution of individual parameters

# %%
scales = transform_unconstrained_scale(
    hmc_net.samples[-1], factor=transform_unconstrained_scale_factor
)  # _chain[-1][num_burnin_steps:]
_x = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(figsize=(8, 8))
ax.hist(scales.numpy().flatten(), bins=200, density=True)
ax.plot(_x, noise_scale_prior.prob(_x).numpy().flatten())
# ax.set_xlim([0, 1])


# %%
weight = hmc_net.samples[2][:, :, 2, 4]
_x = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(figsize=(8, 8))
ax.hist(weight.numpy().flatten(), bins=200, density=True)
