# %load_ext autoreload
# %autoreload 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from core.hmc import HMCDensityNetwork, hmc_density_network_from_save_path
from core.last_layer import PostHocLastLayerBayesianEnsemble as LLBEnsemble
from core.last_layer import PostHocLastLayerBayesianNetwork as LLBNetwork
from core.map import MapDensityEnsemble, MapDensityNetwork, MapEnsemble
from core.network_utils import (
    activation_strings_to_activation_functions,
    build_scratch_model,
)
from core.plotting_utils import (
    plot_distribution_samples,
    plot_predictive_distribution,
    plot_predictive_distribution_and_function_samples,
    plot_weight_space_first_vs_last_layer,
    plot_weight_space_histogram,
)
from core.preprocessing import preprocess_create_x_train_test
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions

save_dir = ".save_data/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

figure_dir = "figures/weight_space_visualization"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)

################################################################################
# %% markdown
# # Do not change this notebook. Otherwise it might be hard to re create the figures.
################################################################################

# %% codecell
n_train = 20
_x_train, y_train = create_split_periodic_data_heteroscedastic(
    n_train=n_train,
    lower1=-2,
    upper1=0,
    lower2=-2,
    upper2=0,
    sigma1=0.8,
    sigma2=0.8,
    seed=0,
)
x_train, _x_test, x_test = preprocess_create_x_train_test(_x_train)
y_test = ground_truth_periodic_function(_x_test)


# %% codecell
input_shape = [1]
layer_units = [3, 2]  # [50, 30, 20, 10, 2]
layer_activations = ["tanh"] * (len(layer_units) - 1) + ["linear"]
prior_scale = 7
weight_priors = [tfd.Normal(0, prior_scale)] * len(layer_units)
bias_priors = weight_priors

unique_hmc_save_path = f"hmc_n-units-{layer_units}_activations-{layer_activations}_prior-scale-{prior_scale}"
unique_hmc_save_path = save_dir.joinpath(unique_hmc_save_path)

# %%
batch_size = 20
epochs = 150


# %%
initial_learning_rate = 0.5
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
)


# %% markdown
# # MAP initialization for HMC

# %%
map_net = MapDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
)

map_net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)

# %%
prediction = map_net.predict(x_test)
plot_predictive_distribution(
    x_test=_x_test,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-10, 10],
)

# %%
initial_state = map_net.get_weights()
initial_state

# %% markdown
# # HMC


# %%
rerun_training = False


# %%
sampler = "nuts"
num_burnin_steps = 200
num_results = 1000
num_leapfrog_steps = 15
step_size = 0.1

# %%
hmc_net = HMCDensityNetwork(
    input_shape,
    layer_units,
    layer_activations,
    weight_priors=weight_priors,
    bias_priors=bias_priors,
    seed=0,
)

prior_predictive_distributions = hmc_net.predict_with_prior_samples(
    x_test, n_samples=4, seed=0
)
plot_distribution_samples(
    x_test=_x_test,
    distribution_samples=prior_predictive_distributions,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-10, 10],
)
# %%
if rerun_training or not unique_hmc_save_path.is_file():
    hmc_net = HMCDensityNetwork(
        input_shape,
        layer_units,
        layer_activations,
        weight_priors=weight_priors,
        bias_priors=bias_priors,
        seed=0,
    )
    hmc_net.fit(
        x_train,
        y_train,
        initial_state=initial_state,
        sampler=sampler,
        num_burnin_steps=num_burnin_steps,
        num_results=num_results,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=step_size,
        verbose=0,
    )

    hmc_net.save(unique_hmc_save_path)
else:
    hmc_net = hmc_density_network_from_save_path(unique_hmc_save_path)


# %%
n_predictions = 5
mog_prediction = hmc_net.predict(x_test, thinning=10)  # Mixture Of Gaussian prediction
gaussian_predictions = hmc_net.predict_list_of_gaussians(
    x_test, n_predictions=n_predictions
)
plot_predictive_distribution_and_function_samples(
    x_test=_x_test,
    predictive_distribution=mog_prediction,
    distribution_samples=gaussian_predictions,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-5, 10],
)

# %%
hmc_net.ess()

# %% markdown
# # Last Layer Bayesian network

# %%
layer_units[-1] = 1
initial_learning_rate = 0.5
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
)

# last layer bayesian network
llb_net = LLBNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
)


llb_net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


prediction = llb_net.predict(x_test)

plot_predictive_distribution(
    x_test=_x_test,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-10, 10],
)

# %%
feature_extractor_weights, last_layer_weight_distribution = llb_net.get_weights()

# %% markdown
# # MAP Ensemble

# %%
layer_units[-1] = 2
n_networks = 3

initial_learning_rate = 0.5
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
)
ensemble = MapDensityEnsemble(
    n_networks=n_networks,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=4,
)

ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=100, verbose=0
)

ensemble_weights = ensemble.get_weights()

prediction = ensemble.predict(x_test)
distribution_samples = ensemble.predict_list_of_gaussians(x_test)
plot_predictive_distribution_and_function_samples(
    x_test=_x_test,
    predictive_distribution=prediction,
    distribution_samples=distribution_samples,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-10, 10],
    # save_path=figure_dir.joinpath(f"pure-relu_n-units-{layer_units}_epochs-{epochs}_last-layer-bayesian.pdf")
)


# %% markdown
# # Last Layer Bayesian Ensemble

# %%
layer_units[-1] = 1
initial_learning_rate = 0.5
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
)
llb_ensemble = LLBEnsemble(
    n_networks=n_networks,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
)


llb_ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)

llb_ensemble_weights = llb_ensemble.get_weights()

prediction = llb_ensemble.predict(x_test)
distribution_samples = llb_ensemble.predict_list(x_test)
plot_predictive_distribution_and_function_samples(
    x_test=_x_test,
    predictive_distribution=prediction,
    distribution_samples=distribution_samples,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-10, 10],
    # save_path=figure_dir.joinpath(f"pure-relu_n-units-{layer_units}_epochs-{epochs}_last-layer-bayesian.pdf")
)

# %% markdown
# # Visualizing specific weight

# %%
bins = 30
kde_bandwidth = 0.5
x_plot_last = np.linspace(-9, 10, 1000)

plot_weight_space_first_vs_last_layer(
    samples_first=hmc_net.samples[0][:, 0, 0],
    samples_last=hmc_net.samples[2][:, 2, 0],
    x_plot_last=x_plot_last,
    kde=True,
    kde_bandwidth=kde_bandwidth,
    point_estimate_first=feature_extractor_weights[0][0, 0],
    point_estimate_last=feature_extractor_weights[2][2, 0],
    fig_title="MAP network",
    # save_path=figure_dir.joinpath(f"map_vs_hmc.pdf")
)

# %%
plot_weight_space_first_vs_last_layer(
    samples_first=hmc_net.samples[0][:, 0, 0],
    samples_last=hmc_net.samples[2][:, 2, 0],
    x_plot_last=x_plot_last,
    kde=True,
    kde_bandwidth=kde_bandwidth,
    point_estimate_first=feature_extractor_weights[0][0, 0],
    distribution_last=last_layer_weight_distribution[2],
    fig_title="Last layer Bayesian network",
    # save_path=figure_dir.joinpath(f"llb_vs_hmc.pdf")
)

# %%
ensemble_point_estimates_first = [weights[0][0, 0] for weights in ensemble_weights]
ensemble_point_estimates_last = [weights[2][2, 0] for weights in ensemble_weights]
plot_weight_space_first_vs_last_layer(
    samples_first=hmc_net.samples[0][:, 0, 0],
    samples_last=hmc_net.samples[2][:, 2, 0],
    x_plot_last=x_plot_last,
    kde_bandwidth=kde_bandwidth,
    ensemble_point_estimates_first=ensemble_point_estimates_first,
    ensemble_point_estimates_last=ensemble_point_estimates_last,
    fig_title="Ensemble of three networks",
    # save_path=figure_dir.joinpath(f"ensemble_vs_hmc.pdf")
)


# %%
llb_ensemble_point_estimates_first = [
    weights[0][0, 0] for weights, dists in llb_ensemble_weights
]
ensemble_distributions_last = [dists[2] for weights, dists in llb_ensemble_weights]
plot_weight_space_first_vs_last_layer(
    samples_first=hmc_net.samples[0][:, 0, 0],
    samples_last=hmc_net.samples[2][:, 2, 0],
    x_plot_last=x_plot_last,
    kde_bandwidth=kde_bandwidth,
    ensemble_point_estimates_first=ensemble_point_estimates_first,
    ensemble_distributions_last=ensemble_distributions_last,
    fig_title="Last layer Bayesian Ensemble of three networks",
    # save_path=figure_dir.joinpath(f"llb_ensemble_vs_hmc.pdf")
)


# %% markdown
# # Visualizing all weights

# %%
for i in range(layer_units[0]):
    plot_weight_space_first_vs_last_layer(
        samples_first=hmc_net.samples[0][:, 0, i],
        samples_last=hmc_net.samples[2][:, i, 0],
        bins=bins,
        kde_bandwidth=kde_bandwidth,
        point_estimate_first=feature_extractor_weights[0][0, i],
        distribution_last=last_layer_weight_distribution[i],
    )

# # %%
# for i in range(layer_units[0]):
#     plot_weight_space_first_vs_last_layer(
#         samples_first=hmc_net.samples[0][:, 0, i],
#         samples_last=hmc_net.samples[2][:, i, 0],
#         bins=bins,
#         kde_bandwidth=kde_bandwidth,
#         point_estimate_first=initial_state[0][0, i],
#         point_estimate_last=initial_state[2][i, 0],
#     )

# %%
for i in range(layer_units[0]):
    plot_weight_space_first_vs_last_layer(
        samples_first=hmc_net.samples[0][:, 0, i],
        samples_last=hmc_net.samples[2][:, i, 0],
        kde_bandwidth=kde_bandwidth,
        point_estimate_first=feature_extractor_weights[0][0, i],
        point_estimate_last=feature_extractor_weights[2][i, 0],
    )

# %%
for i in range(layer_units[0]):
    ensemble_point_estimates_first = [weights[0][0, i] for weights in ensemble_weights]
    ensemble_point_estimates_last = [weights[2][i, 0] for weights in ensemble_weights]
    print(ensemble_point_estimates_first)
    plot_weight_space_first_vs_last_layer(
        samples_first=hmc_net.samples[0][:, 0, i],
        samples_last=hmc_net.samples[2][:, i, 0],
        kde_bandwidth=kde_bandwidth,
        ensemble_point_estimates_first=ensemble_point_estimates_first,
        ensemble_point_estimates_last=ensemble_point_estimates_last,
    )
