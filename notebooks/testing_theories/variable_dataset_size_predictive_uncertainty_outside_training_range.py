# %% markdown
# The question that drives this notebook is what happens to the posterior predictive
# uncertainty outside the range of the training data when the number of training data
# points is very large. My prediction would be that map networks and map ensembles keep
# being extremely overconfident. I would predict that HMC gets somewhat more confident in the
# extrapolation near the training data but has still large uncertainty further away from
# the training data.
# Since the bayesian linear regression in the last layer bayesian network should keep
# reducing its epistemic uncertainty about the weights with more data, the posterior
# predictive uncertainty should probably become smaller.
#
# Results:
# For map networks and map ensembles the prediction that they keep being overconfident
# seems to be true (at least when trained until convergence. Sometimes I had to increase
# the number of epochs (to 200) to get this result).
# For the last layer Bayesian model it seems to be the case that indeed the posterior
# predictive uncertainty outside the training range decreases with training data (this
# is a trend, not a perfect correlation).
# Interestingly, the last layer bayesian ensemble still seems to have quite a bit of
# functional diversity. This is not what I would have expected. But it is also reflected
# in the functional diversity outside the training range for the individual last layer
# bayesian networks when changing the seed.
# HMC seems to keep a very high uncertainty outside the range of the training data. But
# with few training data points it also has high predictive uncertainty within the
# region of the training data. Last layer bayesian networks and ensembles agree with HMC
# that the predictive uncertainty within the region of the training data should be
# higher when the number of training data points is low.
# Overall, last layer bayesian ensembles show a really good behaviour compared to the
# other approximate methods in these experiments.


# %%
# %load_ext autoreload
# %autoreload 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# %%
from core.hmc import HMCDensityNetwork, hmc_density_network_from_save_path
from core.last_layer import PostHocLastLayerBayesianEnsemble as LLBEnsemble
from core.last_layer import PostHocLastLayerBayesianNetwork as LLBNetwork
from core.map import MapDensityEnsemble, MapDensityNetwork
from core.plotting_utils import (
    plot_distribution_samples,
    plot_ground_truth,
    plot_moment_matched_predictive_normal_distribution,
    plot_moment_matched_predictive_normal_distribution_and_function_samples,
    plot_training_data,
)
from core.preprocessing import StandardPreprocessor
from data.toy_regression import cos_linear_edge_data, ground_truth_cos_function

tfd = tfp.distributions

figure_dir = "figures"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)
experiment_name = "epistemic-uncertainty-much-data"


# %% markdown
# ## N Training Data: 15

# %%
seed = 1
n_train = 15


# train and test variables beginning with an underscore are unprocessed.
_x_train, y_train = cos_linear_edge_data(n_data=n_train, sigma=0.1, seed=seed)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(
    _x_train, test_ds=1.8
)
y_ground_truth = ground_truth_cos_function(_x_plot)

layer_units = [20, 5] + [2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]


# %% codecell
y_lim = [-5, 5]
figsize = (10, 6)
fig, ax = plt.subplots(figsize=figsize)
plot_training_data(_x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(_x_plot, y_ground_truth, fig=fig, ax=ax, alpha=0.2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(y_lim)
fig.suptitle("Data with approximately linear edges", fontsize=15)
ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
ax.legend()
# fig.savefig(figure_dir.joinpath(f"data_{experiment_name}.pdf"), bbox_inches="tight")

# %%
# General training
epochs = 100
batch_size = n_train

# %% markdown
# # Density Ensemble

# %%
n_networks = 5

# %%
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)

ensemble = MapDensityEnsemble(
    n_networks=n_networks,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=seed,
)

ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=200, verbose=0
)


# %%
prediction = ensemble.predict(x_plot)  # Mixture Of Gaussian prediction
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title=f"Ensemble of {n_networks} MAP networks",
    # save_path=figure_dir.joinpath(f"ensemble_moment_matched_{experiment_name}.pdf")
)


# %% codecell
gaussian_predictions = ensemble.predict_list_of_gaussians(x_plot, n_predictions=3)
fig, ax = plt.subplots(figsize=figsize)
plot_distribution_samples(
    x_plot=_x_plot,
    distribution_samples=gaussian_predictions,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    save_path=figure_dir.joinpath(f"ensemble_members_{experiment_name}.pdf"),
)


# %% markdown
# # Neural Linear Model


# %%
layer_units[-1] = 1
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)

# last layer bayesian network
llb_net = LLBNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=5,
)
layer_units[-1] = 2

llb_net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=200, verbose=0
)

# %%
prediction = llb_net.predict(x_plot)
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title="Neural Linear Model",
    # save_path=figure_dir.joinpath(f"llb_moment_matched_{experiment_name}.pdf")
)


# %% markdown
# # HMC

# %%
save_dir = ".save_data/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# %%
layer_units[-1] = 2
prior_std = 1
weight_priors = [tfd.Normal(0, prior_std)] * len(layer_units)
bias_priors = weight_priors


# %%
rerun_training = False


# %%
sampler = "nuts"
num_burnin_steps = 200
num_results = 800
num_leapfrog_steps = 25
step_size = 0.1
unique_hmc_save_path = f"hmc_n-units-{layer_units}_activations-{layer_activations}_prior-std-{prior_std}_n-train-{n_train}_sampler-{sampler}_data-cos-linear-edge"
unique_hmc_save_path = save_dir.joinpath(unique_hmc_save_path)

if rerun_training or not unique_hmc_save_path.is_file():
    hmc_net = HMCDensityNetwork(
        [1],
        layer_units,
        layer_activations,
        weight_priors=weight_priors,
        bias_priors=bias_priors,
        seed=0,
    )

    hmc_net.fit(
        x_train,
        y_train,
        sampler=sampler,
        num_burnin_steps=num_burnin_steps,
        num_results=num_results,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=step_size,
        learning_rate=0.01,
        batch_size=n_train,
        epochs=150,
        verbose=0,
    )

    hmc_net.save(unique_hmc_save_path)
else:
    hmc_net = hmc_density_network_from_save_path(unique_hmc_save_path)


# %%
prediction = hmc_net.predict(x_plot, thinning=10)
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title=f"{sampler}",
    # save_path=figure_dir.joinpath(f"hmc_{experiment_name}.pdf")
)


# %% markdown
# # Neural Linear Ensemble

# %%
layer_units[-1] = 1
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)
llb_ensemble = LLBEnsemble(
    n_networks=n_networks,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=4,
)
layer_units[-1] = 2

llb_ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


# %%
prediction = llb_ensemble.predict(x_plot)
distribution_samples = llb_ensemble.predict_list(x_plot)
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    # distribution_samples=distribution_samples,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title=f"Neural Linear Ensemble of {n_networks} networks",
    # save_path=figure_dir.joinpath(f"llb-ensemble_{experiment_name}.pdf")
)


# %% markdown
# ## N Training Data: 80

# %%
seed = 1
n_train = 80


# train and test variables beginning with an underscore are unprocessed.
_x_train, y_train = cos_linear_edge_data(n_data=n_train, sigma=0.1, seed=seed)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(
    _x_train, test_ds=1.8
)
y_ground_truth = ground_truth_cos_function(_x_plot)


# %% codecell
y_lim = [-5, 5]
figsize = (10, 6)
fig, ax = plt.subplots(figsize=figsize)
plot_training_data(_x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(_x_plot, y_ground_truth, fig=fig, ax=ax, alpha=0.2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(y_lim)
fig.suptitle("Data with approximately linear edges", fontsize=15)
ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
ax.legend()
# fig.savefig(figure_dir.joinpath(f"data_{experiment_name}.pdf"), bbox_inches="tight")

# %%
# General training
batch_size = n_train

# %% markdown
# # Density Ensemble

# %%
n_networks = 5

# %%
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)

ensemble = MapDensityEnsemble(
    n_networks=n_networks,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=seed,
)

ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=200, verbose=0
)


# %%
prediction = ensemble.predict(x_plot)  # Mixture Of Gaussian prediction
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title=f"Ensemble of {n_networks} MAP networks",
    # save_path=figure_dir.joinpath(f"ensemble_moment_matched_{experiment_name}.pdf")
)


# %% codecell
gaussian_predictions = ensemble.predict_list_of_gaussians(x_plot, n_predictions=3)
fig, ax = plt.subplots(figsize=figsize)
plot_distribution_samples(
    x_plot=_x_plot,
    distribution_samples=gaussian_predictions,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    save_path=figure_dir.joinpath(f"ensemble_members_{experiment_name}.pdf"),
)


# %% markdown
# # Neural Linear Model


# %%
layer_units[-1] = 1
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)

# last layer bayesian network
llb_net = LLBNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=7,
)
layer_units[-1] = 2

llb_net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)

# %%
prediction = llb_net.predict(x_plot)
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title="Neural Linear Model",
    # save_path=figure_dir.joinpath(f"llb_moment_matched_{experiment_name}.pdf")
)


# %% markdown
# # HMC

# %%
save_dir = ".save_data/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# %%
layer_units[-1] = 2
prior_std = 1
weight_priors = [tfd.Normal(0, prior_std)] * len(layer_units)
bias_priors = weight_priors

# %%
rerun_training = False


# %%
sampler = "hmc"
num_burnin_steps = 500
num_results = 2000
num_leapfrog_steps = 40
step_size = 0.1
unique_hmc_save_path = f"hmc_n-units-{layer_units}_activations-{layer_activations}_prior-std-{prior_std}_n-train-{n_train}_sampler-{sampler}_data-cos-linear-edge"
unique_hmc_save_path = save_dir.joinpath(unique_hmc_save_path)

if rerun_training or not unique_hmc_save_path.is_file():
    hmc_net = HMCDensityNetwork(
        [1],
        layer_units,
        layer_activations,
        weight_priors=weight_priors,
        bias_priors=bias_priors,
        seed=0,
    )

    hmc_net.fit(
        x_train,
        y_train,
        sampler=sampler,
        num_burnin_steps=num_burnin_steps,
        num_results=num_results,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=step_size,
        learning_rate=0.01,
        batch_size=n_train,
        epochs=80,
        verbose=0,
    )

    hmc_net.save(unique_hmc_save_path)
else:
    hmc_net = hmc_density_network_from_save_path(unique_hmc_save_path)


# %%
prediction = hmc_net.predict(x_plot, thinning=10)
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title="HMC",
    # save_path=figure_dir.joinpath(f"hmc_{experiment_name}.pdf")
)


# %% markdown
# # Neural Linear Ensemble

# %%
layer_units[-1] = 1
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)
llb_ensemble = LLBEnsemble(
    n_networks=n_networks,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=4,
)
layer_units[-1] = 2

llb_ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


# %%
prediction = llb_ensemble.predict(x_plot)
distribution_samples = llb_ensemble.predict_list(x_plot)
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    # distribution_samples=distribution_samples,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title=f"Neural Linear Ensemble of {n_networks} networks",
    # save_path=figure_dir.joinpath(f"llb-ensemble_{experiment_name}.pdf")
)


# %% markdown
# ## N Training Data: 1000

# %%
seed = 1
n_train = 1000


# train and test variables beginning with an underscore are unprocessed.
_x_train, y_train = cos_linear_edge_data(n_data=n_train, sigma=0.1, seed=seed)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(
    _x_train, test_ds=1.8
)
y_ground_truth = ground_truth_cos_function(_x_plot)


# %% codecell
y_lim = [-5, 5]
figsize = (10, 6)
fig, ax = plt.subplots(figsize=figsize)
plot_training_data(_x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(_x_plot, y_ground_truth, fig=fig, ax=ax, alpha=0.2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(y_lim)
fig.suptitle("Data with approximately linear edges", fontsize=15)
ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
ax.legend()
# fig.savefig(figure_dir.joinpath(f"data_{experiment_name}.pdf"), bbox_inches="tight")

# %%
# General training
batch_size = n_train

# %% markdown
# # Density Ensemble

# %%
n_networks = 5

# %%
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)

ensemble = MapDensityEnsemble(
    n_networks=n_networks,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=seed,
)

ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=200, verbose=0
)


# %%
prediction = ensemble.predict(x_plot)  # Mixture Of Gaussian prediction
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title="Ensemble of ten MAP networks",
    # save_path=figure_dir.joinpath(f"ensemble_moment_matched_{experiment_name}.pdf")
)


# %% codecell
gaussian_predictions = ensemble.predict_list_of_gaussians(x_plot, n_predictions=3)
fig, ax = plt.subplots(figsize=figsize)
plot_distribution_samples(
    x_plot=_x_plot,
    distribution_samples=gaussian_predictions,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    save_path=figure_dir.joinpath(f"ensemble_members_{experiment_name}.pdf"),
)


# %% markdown
# # Neural Linear Model


# %%
layer_units[-1] = 1
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)

# last layer bayesian network
llb_net = LLBNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=8,
)
layer_units[-1] = 2

llb_net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)

# %%
prediction = llb_net.predict(x_plot)
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title="Neural Linear Model",
    # save_path=figure_dir.joinpath(f"llb_moment_matched_{experiment_name}.pdf")
)


# %% markdown
# # HMC

# %%
save_dir = ".save_data/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# %%
layer_units[-1] = 2
prior_std = 1
weight_priors = [tfd.Normal(0, prior_std)] * len(layer_units)
bias_priors = weight_priors

# %%
rerun_training = False


# %%
sampler = "nuts"
num_burnin_steps = 50
num_results = 200
num_leapfrog_steps = 40
step_size = 0.1
unique_hmc_save_path = f"hmc_n-units-{layer_units}_activations-{layer_activations}_prior-std-{prior_std}_n-train-{n_train}_sampler-{sampler}_data-cos-linear-edge"
unique_hmc_save_path = save_dir.joinpath(unique_hmc_save_path)

if rerun_training or not unique_hmc_save_path.is_file():
    hmc_net = HMCDensityNetwork(
        [1],
        layer_units,
        layer_activations,
        weight_priors=weight_priors,
        bias_priors=bias_priors,
        seed=0,
    )

    hmc_net.fit(
        x_train,
        y_train,
        sampler=sampler,
        num_burnin_steps=num_burnin_steps,
        num_results=num_results,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=step_size,
        learning_rate=0.01,
        batch_size=n_train,
        epochs=80,
        verbose=0,
    )

    hmc_net.save(unique_hmc_save_path)
else:
    hmc_net = hmc_density_network_from_save_path(unique_hmc_save_path)


# %%
prediction = hmc_net.predict(x_plot, thinning=10)
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title=f"{sampler}",
    # save_path=figure_dir.joinpath(f"hmc_{experiment_name}.pdf")
)


# %% markdown
# # Neural Linear Ensemble

# %%
layer_units[-1] = 1
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)
llb_ensemble = LLBEnsemble(
    n_networks=n_networks,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=0,
)
layer_units[-1] = 2

llb_ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


# %%
prediction = llb_ensemble.predict(x_plot)
distribution_samples = llb_ensemble.predict_list(x_plot)
fig, ax = plt.subplots(figsize=figsize)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    # distribution_samples=distribution_samples,
    # x_train=_x_train,
    # y_train=y_train,
    y_ground_truth=y_ground_truth,
    fig=fig,
    ax=ax,
    y_lim=y_lim,
    title=f"Neural Linear Ensemble of {n_networks} networks",
    # save_path=figure_dir.joinpath(f"llb-ensemble_{experiment_name}.pdf")
)
