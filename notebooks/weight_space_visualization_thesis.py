# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tikzplotlib as tpl
import yaml

from core.hmc import HMCDensityNetwork, hmc_density_network_from_save_path
from core.last_layer import PostHocLastLayerBayesianEnsemble as LLBEnsemble
from core.map import MapDensityEnsemble, MapDensityNetwork
from core.network_utils import make_independent_gaussian_network_prior
from core.plotting_utils import (
    plot_distribution_samples,
    plot_ground_truth,
    plot_moment_matched_predictive_normal_distribution,
    plot_moment_matched_predictive_normal_distribution_and_function_samples,
    plot_training_data,
    plot_weight_space_first_vs_last_layer,
)
from core.preprocessing import StandardPreprocessor
from core.sanity_checks import check_posterior_equivalence
from core.variational import VariationalDensityNetwork
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions

save_dir = f".save_data/toy_weight_space_visualization"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

figure_dir = "figures/weight_space_visualization"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)


# %%
n_train = 20
_x_train, _y_train = create_split_periodic_data_heteroscedastic(
    n_data=n_train,
    lower1=-2,
    upper1=0,
    lower2=-2,
    upper2=0,
    sigma1=0.8,
    sigma2=0.8,
    seed=0,
)

x_preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = x_preprocessor.preprocess_create_x_train_x_plot(_x_train)
y_preprocessor = StandardPreprocessor()
y_train = y_preprocessor.fit_transform(_y_train)
_y_ground_truth = ground_truth_periodic_function(_x_plot)
y_ground_truth = y_preprocessor.transform(_y_ground_truth)

# %%
y_lim = [-5, 5]
fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(x_plot, y_ground_truth, fig=fig, ax=ax)
ax.legend()


# %%
with open("config/uci-hyperparameters-config.yaml") as f:
    experiment_config = yaml.full_load(f)

input_shape = [1]
layer_units = [3, 1]
layer_activations = ["tanh"] * (len(layer_units) - 1) + ["linear"]
prior_scale = 5.0  # not in use  # 7

initial_unconstrained_scale = experiment_config["initial_unconstrained_scale"]
transform_unconstrained_scale_factor = experiment_config[
    "transform_unconstrained_scale_factor"
]

learning_rate = experiment_config["learning_rate"]
epochs = experiment_config["epochs"]

weight_prior = tfd.Normal(0, prior_scale)
bias_prior = tfd.Normal(0, prior_scale)

a = experiment_config["noise_var_prior_a"]
b = experiment_config["noise_var_prior_b"]
_var_d = tfd.InverseGamma([[a]], b)
noise_scale_prior = tfd.TransformedDistribution(
    distribution=_var_d, bijector=tfp.bijectors.Invert(tfp.bijectors.Square())
)
patience = 50
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=patience, verbose=1, restore_best_weights=False
)

n_train = x_train.shape[0]
n_features = x_train.shape[1]
network_prior = make_independent_gaussian_network_prior(
    input_shape=input_shape, layer_units=layer_units, loc=0.0, scale=prior_scale
)


batch_size = n_train
epochs = 10000


# %%
ensemble = MapDensityEnsemble(
    n_networks=5,
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=initial_unconstrained_scale,
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    weight_prior=weight_prior,
    bias_prior=bias_prior,
    noise_scale_prior=noise_scale_prior,
    n_train=n_train,
    learning_rate=learning_rate,
    names=["feature_extractor", "output"],
    seed=[0, 10, 1000, 1000000, 1000001],
)


ensemble.fit(
    x_train=x_train,
    y_train=y_train,
    batch_size=batch_size,
    epochs=epochs,
    early_stop_callback=early_stop_callback,
    verbose=0,
)

individual_predictions = ensemble.predict_list_of_gaussians(x_plot, n_predictions=5)
prediction = ensemble.predict(x_plot)
plot_moment_matched_predictive_normal_distribution_and_function_samples(
    x_plot=x_plot,
    predictive_distribution=prediction,
    distribution_samples=individual_predictions,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
)

# # %%
# fig, ax = plt.subplots()
# for i in range(5):
#     prediction = ensemble.networks[i].predict(x_plot)
#     plot_moment_matched_predictive_normal_distribution(
#         x_plot=x_plot,
#         predictive_distribution=prediction,
#         x_train=x_train,
#         y_train=y_train,
#         y_ground_truth=y_ground_truth,
#         y_lim=y_lim,
#         fig=fig,
#         ax=ax
#     )


# %%
llb_ensemble = LLBEnsemble(
    n_networks=5,
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=initial_unconstrained_scale,
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    last_layer_prior="weakly-informative",
    learning_rate=learning_rate,
    seed=0,
)


llb_ensemble.fit(
    x_train=x_train, y_train=y_train, pretrained_networks=ensemble.networks
)


individual_predictions = llb_ensemble.predict_list(x_plot)
prediction = llb_ensemble.predict(x_plot)
plot_moment_matched_predictive_normal_distribution_and_function_samples(
    x_plot=x_plot,
    predictive_distribution=prediction,
    distribution_samples=individual_predictions,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
)


# %%
variational_network = VariationalDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=initial_unconstrained_scale,  # -10,
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    prior_scale_identity_multiplier=prior_scale,
    kl_weight=1 / n_train,
    noise_scale_prior=noise_scale_prior,
    n_train=n_train,
    learning_rate=learning_rate,
    evaluate_ignore_prior_loss=False,
    mean_init_noise_scale=0.0,
    seed=1,
)

variational_network.fit(x_train, y_train, batch_size=batch_size, epochs=7000, verbose=0)
predictive_distribution = variational_network.predict(x_plot, n_predictions=20)
gaussian_predictions = variational_network.predict_list_of_gaussians(
    x_plot, n_predictions=4
)
plot_moment_matched_predictive_normal_distribution_and_function_samples(
    x_plot=x_plot,
    predictive_distribution=predictive_distribution,
    distribution_samples=gaussian_predictions,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    no_ticks=False,
)


# %%
sampler = "hmc"
num_burnin_steps = 5000
num_results = 3000
num_leapfrog_steps = 50
step_size = 0.01

n_chains = 2

# %%
hmc_net = HMCDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    network_prior=network_prior,
    noise_scale_prior=noise_scale_prior,
    sampler=sampler,
    step_size_adapter="dual_averaging",
    num_burnin_steps=num_burnin_steps,
    step_size=step_size,
    num_leapfrog_steps=num_leapfrog_steps,
    seed=0,
)

assert check_posterior_equivalence(
    ensemble.networks[0], hmc_net, x_train, y_train, n_train
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

overdispersed_prior_samples = hmc_net.sample_prior_state(
    n_samples=n_chains, overdisp=1.2, seed=0
)
current_state = overdispersed_prior_samples


# %% markdown
# # Fitting

# %%
################################################################################
# hmc_net.fit(x_train, y_train, current_state=current_state, num_results=num_results)
################################################################################

# %% markdown
# # Saving

# %%
################################################################################
# save_path = save_dir.joinpath(f"hmc")
# hmc_net.save(save_path)
################################################################################


# %% markdown
# # Loading

# %%
################################################################################
save_path = save_dir.joinpath(f"hmc")
hmc_net = hmc_density_network_from_save_path(
    save_path, network_prior=network_prior, noise_scale_prior=noise_scale_prior
)
################################################################################


# %% markdown
# ## Additional samples

# %%
################################################################################
# hmc_net.fit(x_train, y_train, num_results=20000, resume=True)
################################################################################

# %% markdown


# %%
individual_predictions = [
    hmc_net.predict_random_sample(x_plot, seed=i) for i in range(5)
]
prediction = hmc_net.predict(x_plot, thinning=20)
plot_moment_matched_predictive_normal_distribution_and_function_samples(
    x_plot=x_plot,
    predictive_distribution=prediction,
    distribution_samples=individual_predictions,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    # no_ticks=False,
    y_lim=y_lim,
)


# %%
print("Potential Scale Reduction")
weights_reduction = hmc_net.potential_scale_reduction()
print(
    f"Weight Space (Max per layer): {tf.nest.map_structure(lambda x: tf.reduce_max(x).numpy(), weights_reduction)}"
)

ess = hmc_net.effective_sample_size(thinning=2)
print()
print(
    "Ess (Min per layer)",
    tf.nest.map_structure(lambda x: tf.reduce_max(x).numpy(), ess),
)


acceptance_ratio = hmc_net.acceptance_ratio()
print()
print("Acceptance ratio", acceptance_ratio.numpy())


hmc_net._chain[2].shape
hmc_net.samples[2].shape


# %%
bins = 30
kde_bandwidth_factor = 0.012
x_plot_first = np.linspace(-20, 20, 500)
x_plot_last = np.linspace(-7, 7, 500)

y_lim2 = None
figsize = (8, 2.5)

ih_i = 1
ho_i = 1

thinning = 1

single_network_index = 3

# %%
plot_weight_space_first_vs_last_layer(
    samples_first=hmc_net.samples[0][::thinning, :, 0, ih_i].numpy().flatten(),
    samples_last=hmc_net.samples[2][::thinning, :, ho_i, 0].numpy().flatten(),
    x_plot_first=x_plot_first,
    x_plot_last=x_plot_last,
    kde=True,
    kde_bandwidth_factor=kde_bandwidth_factor,
    point_estimate_first=ensemble.get_weights()[single_network_index][0][0, ih_i],
    point_estimate_last=ensemble.get_weights()[single_network_index][2][ho_i, 0],
    x1_label="$w_1$",
    x2_label="$w_2$",
    y1_label="Density",
    fig_title="MAP network",
    figsize=figsize,
    y_lim2=y_lim2,
    save_path=figure_dir.joinpath(f"map_vs_hmc.pdf"),
)

# %%
llb_ensemble_weights = llb_ensemble.get_weights()
llb_ensemble_point_estimates_first = [
    weights[0][0, ih_i] for weights, dists in llb_ensemble_weights
]
ensemble_distributions_last = [dists[ho_i] for weights, dists in llb_ensemble_weights]
plot_weight_space_first_vs_last_layer(
    samples_first=hmc_net.samples[0][::thinning, :, 0, ih_i].numpy().flatten(),
    samples_last=hmc_net.samples[2][::thinning, :, ho_i, 0].numpy().flatten(),
    x_plot_first=x_plot_first,
    x_plot_last=x_plot_last,
    kde=True,
    kde_bandwidth_factor=kde_bandwidth_factor,
    point_estimate_first=llb_ensemble_point_estimates_first[single_network_index],
    distribution_last=ensemble_distributions_last[single_network_index],
    x1_label="$w_1$",
    x2_label="$w_2$",
    y1_label="Density",
    fig_title="Neural linear model",
    figsize=figsize,
    y_lim2=y_lim2,
    save_path=figure_dir.joinpath(f"llb_vs_hmc.pdf"),
)


# %%
weights = ensemble.get_weights()
ensemble_point_estimates_first = [weight[0][0, ih_i] for weight in weights]
ensemble_point_estimates_last = [weight[2][ho_i, 0] for weight in weights]
plot_weight_space_first_vs_last_layer(
    samples_first=hmc_net.samples[0][::thinning, :, 0, ih_i].numpy().flatten(),
    samples_last=hmc_net.samples[2][::thinning, :, ho_i, 0].numpy().flatten(),
    x_plot_first=x_plot_first,
    x_plot_last=x_plot_last,
    kde_bandwidth_factor=kde_bandwidth_factor,
    ensemble_point_estimates_first=ensemble_point_estimates_first,
    ensemble_point_estimates_last=ensemble_point_estimates_last,
    x1_label="$w_1$",
    x2_label="$w_2$",
    y1_label="Density",
    fig_title="MAP ensemble of five networks",
    figsize=figsize,
    y_lim2=y_lim2,
    save_path=figure_dir.joinpath(f"ensemble_vs_hmc.pdf"),
)


# %%
llb_ensemble_weights = llb_ensemble.get_weights()
llb_ensemble_point_estimates_first = [
    weights[0][0, ih_i] for weights, dists in llb_ensemble_weights
]
ensemble_distributions_last = [dists[ho_i] for weights, dists in llb_ensemble_weights]
plot_weight_space_first_vs_last_layer(
    samples_first=hmc_net.samples[0][::thinning, :, 0, ih_i].numpy().flatten(),
    samples_last=hmc_net.samples[2][::thinning, :, ho_i, 0].numpy().flatten(),
    x_plot_first=x_plot_first,
    x_plot_last=x_plot_last,
    kde_bandwidth_factor=kde_bandwidth_factor,
    ensemble_point_estimates_first=llb_ensemble_point_estimates_first,
    ensemble_distributions_last=ensemble_distributions_last,
    x1_label="$w_1$",
    x2_label="$w_2$",
    y1_label="Density",
    fig_title="Neural linear ensemble of five networks",
    figsize=figsize,
    y_lim2=y_lim2,
    save_path=figure_dir.joinpath(f"llb_ensemble_vs_hmc.pdf"),
)

# tpl.save(
#     "example.tex",  # this is name of the file where your code will lie
#     axis_width="\\figwidth",  # we want LaTeX to take care of the width
#     axis_height="\\figheight",  # we want LaTeX to take care of the height
#     # if the figure contains an image in the background (this one doesn't), this is where LaTeX (!) should search for the png.
#     tex_relative_path_to_data="./",
#     # we want the plot to look *exactly* like here (e.g. axis limits, axis ticks, etc.)
#     strict=True,
# )

# %%
ih_i = 1
ho_i = 1
c = np.log(np.expm1(1e-5))
_weights = variational_network.get_weights()
ih_weights_means = _weights[0][0:3]
ih_weights_scales = 1e-5 + tf.nn.softplus(c + _weights[0][6:9])
# h_biases_means = _weights[0][3:6]
# h_biases_scales = 1e-5 + tf.nn.softplus(c + _weights[0][9:])
ho_weights_means = _weights[1][0:3]
ho_weights_scales = 1e-5 + tf.nn.softplus(c + _weights[1][4:7])
print(
    ih_weights_means[ih_i],
    ih_weights_scales[ih_i],
    ho_weights_means[ho_i],
    ho_weights_scales[ho_i],
)
vi_dist_first = tfd.Normal(ih_weights_means[ih_i], ih_weights_scales[ih_i])
vi_dist_last = tfd.Normal(ho_weights_means[ho_i], ho_weights_scales[ho_i])
plot_weight_space_first_vs_last_layer(
    samples_first=hmc_net.samples[0][::thinning, :, 0, ih_i].numpy().flatten(),
    samples_last=hmc_net.samples[2][::thinning, :, ho_i, 0].numpy().flatten(),
    x_plot_first=x_plot_first,
    x_plot_last=x_plot_last,
    kde=True,
    kde_bandwidth_factor=kde_bandwidth_factor,
    distribution_first=vi_dist_first,
    distribution_last=vi_dist_last,
    vi=True,
    x1_label="$w_1$",
    x2_label="$w_2$",
    y1_label="Density",
    fig_title="Variational network",
    figsize=figsize,
    y_lim2=y_lim2,
    save_path=figure_dir.joinpath(f"vi_vs_hmc.pdf"),
)
