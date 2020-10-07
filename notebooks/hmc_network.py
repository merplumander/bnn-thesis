# %load_ext autoreload
# %autoreload 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from core.hmc import HMCNetwork, hmc_network_from_save_path
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
from data.toy_regression import ground_truth_x3_function, x3_gap_data

tfd = tfp.distributions

figure_dir = "figures/temp"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)

# %%
data_seed = 0
n_train = 20

# train and test variables beginning with an underscore are unprocessed.
_x_train, y_train = x3_gap_data(
    n_data=n_train, lower1=-4.9, upper2=4.9, sigma=0.3, seed=data_seed
)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(
    _x_train, test_ds=0.05
)
y_ground_truth = ground_truth_x3_function(_x_plot)

layer_units = [3] + [1]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]


# %% codecell
y_lim = [-5, 5]
fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(_x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(_x_plot, y_ground_truth, fig=fig, ax=ax, alpha=0.2)
ax.legend()

# %%
# General training
train_seed = 0
epochs = 150
batch_size = n_train

# %%
save_dir = ".save_data/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# %%
prior_scale = 1
weight_priors = [tfd.Normal(0, prior_scale)] * len(layer_units)
bias_priors = weight_priors


# %%
rerun_training = True

# %%
sampler = "hmc"
num_burnin_steps = 300
num_results = 1500
num_leapfrog_steps = 50
step_size = 0.1
n_chains = 2

unique_hmc_save_path = f"deletable-hmc_n-units-{layer_units}_activations-{layer_activations}_prior-scale-{prior_scale}_sampler-{sampler}_data-x3-gap_unconstrained-scale"
unique_hmc_save_path = save_dir.joinpath(unique_hmc_save_path)


# %%
if rerun_training or not unique_hmc_save_path.is_file():
    hmc_net = HMCNetwork(
        [1],
        layer_units,
        layer_activations,
        transform_unconstrained_scale_factor=0.2,
        weight_priors=weight_priors,
        bias_priors=bias_priors,
        std_prior=tfd.Normal(0.3, 0.01),
        sampler=sampler,
        n_chains=n_chains,
        seed=1,
    )

    hmc_net.fit(
        x_train,
        y_train,
        num_burnin_steps=num_burnin_steps,
        num_results=num_results,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=step_size,
        learning_rate=0.01,
        batch_size=20,
        epochs=500,
        verbose=0,
    )

    hmc_net.save(unique_hmc_save_path)
else:
    hmc_net = hmc_network_from_save_path(unique_hmc_save_path)


# %%
print(f"Acceptance ratios: {hmc_net.acceptance_ratio()}")
print(
    f"Means and stds of leapfrog steps taken: {hmc_net.leapfrog_steps_taken()[0]}, {hmc_net.leapfrog_steps_taken()[1]}"
)


# %%
hmc_net.ess()


# %%
prediction = hmc_net.predict(x_plot, thinning=10)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    # save_path=figure_dir.joinpath(f"hmc_x3_gap.pdf")
)

# %%
n_predictions = 3
gaussian_predictions = hmc_net.predict_list_of_gaussians(
    x_plot, n_predictions=n_predictions
)
plot_distribution_samples(
    x_plot=_x_plot,
    distribution_samples=gaussian_predictions,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
)
