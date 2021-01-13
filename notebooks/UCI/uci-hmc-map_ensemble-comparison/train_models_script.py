################################################################################
# add current path to sys.path so that we can import from other project files
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from scipy.stats import wasserstein_distance

from core.evaluation_utils import predictive_distribution_wasserstein_distance, rmse
from core.hmc import (
    HMCDensityNetwork,
    hmc_density_network_from_save_path,
    mask_nonsense_chains,
    undo_masking,
)
from core.map import (
    MapDensityEnsemble,
    MapDensityNetwork,
    map_density_ensemble_from_save_path,
)
from core.network_utils import (
    batch_repeat_matrix,
    make_independent_gaussian_network_prior,
    transform_unconstrained_scale,
)
from core.plotting_utils import (
    plot_distribution_samples,
    plot_ground_truth,
    plot_moment_matched_predictive_normal_distribution,
)
from core.preprocessing import StandardPreprocessor
from core.sanity_checks import check_posterior_equivalence
from data.uci import load_uci_data

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, ".")


tfd = tfp.distributions


# %% markdown
# # Energy

# %%
dataset = "yacht"
data_seed = 0
dataset_name = f"{dataset}_{data_seed + 1:02}"

n_hidden_layers = 2
hidden_layers_string = (
    "two-hidden-layers" if n_hidden_layers == 2 else "one-hidden-layer"
)

save_dir = f".save_uci_models/hmc-map-ensemble-comparison/{hidden_layers_string}/{dataset_name}"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# train and test variables beginning with an underscore are unprocessed.
_x, _y, train_indices, _, test_indices = load_uci_data(f"{dataset}")
_x_train = _x[train_indices[data_seed]]
_y_train = _y[train_indices[data_seed]].reshape(-1, 1)
_x_test = _x[test_indices[data_seed]]
_y_test = _y[test_indices[data_seed]].reshape(-1, 1)

x_preprocessor = StandardPreprocessor()
x_train = x_preprocessor.fit_transform(_x_train)
x_test = x_preprocessor.transform(_x_test)
y_preprocessor = StandardPreprocessor()
y_train = y_preprocessor.fit_transform(_y_train)
y_test = y_preprocessor.transform(_y_test)

unnormalized_ll_constant = np.log(y_preprocessor.scaler.scale_)

# %%
with open("config/uci-hyperparameters-config.yaml") as f:
    experiment_config = yaml.full_load(f)

train_seed = experiment_config["train_seed"]
layer_units = experiment_config["layer_units"]
layer_activations = experiment_config["layer_activations"]
if n_hidden_layers == 2:
    layer_units.insert(0, layer_units[0])
    layer_activations.insert(0, layer_activations[0])
    assert layer_units == [50, 50, 1]
    assert layer_activations == ["relu", "relu", "linear"]
initial_unconstrained_scale = experiment_config["initial_unconstrained_scale"]
transform_unconstrained_scale_factor = experiment_config[
    "transform_unconstrained_scale_factor"
]

learning_rate = experiment_config["learning_rate"]
epochs = experiment_config["epochs"]
batch_size = experiment_config["batch_size"]

weight_prior = tfd.Normal(0, experiment_config["weight_prior_scale"])
bias_prior = tfd.Normal(0, experiment_config["bias_prior_scale"])

a = experiment_config["noise_var_prior_a"]
b = experiment_config["noise_var_prior_b"]
_var_d = tfd.InverseGamma([[a]], b)
noise_scale_prior = tfd.TransformedDistribution(
    distribution=_var_d, bijector=tfp.bijectors.Invert(tfp.bijectors.Square())
)
patience = experiment_config["convergence_patience"]
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=patience, verbose=0, restore_best_weights=False
)

n_train = x_train.shape[0]
n_test = x_test.shape[0]
n_features = x_train.shape[1]
input_shape = [n_features]
network_prior = make_independent_gaussian_network_prior(
    input_shape=input_shape,
    layer_units=layer_units,
    loc=0.0,
    scale=experiment_config["weight_prior_scale"],
)


# %% markdown
# # HMC

# %%
n_chains = 4

num_burnin_steps = 100  # 10000
num_results = 100  # 100000
num_steps_between_results = 4  # only every fifth sample is saved

sampler = "hmc"
num_leapfrog_steps = 100
_step_size = 0.01
step_size = [np.ones((n_chains, 1, 1)) * _step_size] * (
    len(layer_units) * 2 + 1  # + 1 for the noise sigma "layer"
)  # Individual step sizes for all chains and all layers
# step_size = np.ones((n_chains, 1, 1)) * _step_size # Individual step sizes for all chains


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
    max_tree_depth=10,
    num_steps_between_results=num_steps_between_results,
    seed=train_seed,
)
prior_samples = [hmc_net.sample_prior_state(seed=seed) for seed in [0, 1, 2, 3, 4]]
prior_predictions = [
    hmc_net.predict_from_sample_parameters(x_train, prior_sample)
    for prior_sample in prior_samples
]
[-prediction.log_prob(y_train) / n_train for prediction in prior_predictions]


overdispersed_prior_samples = hmc_net.sample_prior_state(
    n_samples=n_chains,
    overdisp=1.2,
    seed=data_seed
    + hash(
        dataset
    ),  # make sure each dataset( and possibly data split) starts from different prior samples
)
current_state = overdispersed_prior_samples

# %% markdown
# # Fitting

# %%
################################################################################
hmc_net.fit(x_train, y_train, current_state=current_state, num_results=num_results)
################################################################################


# %% markdown
# # Saving

# %%
################################################################################
save_path = save_dir.joinpath(f"ste_hmc-{dataset_name}")
hmc_net.save(save_path)
################################################################################


# %% markdown
# # Loading

# %%
################################################################################
# save_path = save_dir.joinpath(f"_hmc-{dataset_name}")
# hmc_net = hmc_density_network_from_save_path(
#     save_path, network_prior=network_prior, noise_scale_prior=noise_scale_prior
# )
################################################################################


# %% markdown
# ## Additional samples

# %%
################################################################################
# hmc_net.fit(x_train, y_train, num_results=50000, resume=True)
################################################################################

# %%


# %%
acceptance_ratio = hmc_net.acceptance_ratio()
print(f"Acceptance ratio per chain: {acceptance_ratio}")

# %%
hmc_net.leapfrog_steps_taken()


# %%
print("Potential Scale Reduction")

weights_reduction = hmc_net.potential_scale_reduction()
print(
    f"Weight Space (Max per layer): {tf.nest.map_structure(lambda x: tf.reduce_max(x).numpy(), weights_reduction)}"
)


predictive_mean_reduction = tfp.mcmc.potential_scale_reduction(
    hmc_net._base_predict(x_train, thinning=20).mean(), split_chains=False
)
print(
    f"Mean of Predictive means (train set): {tf.reduce_mean(predictive_mean_reduction):.3f}"
)
print(
    f"Max of Predictive means (train set): {tf.reduce_max(predictive_mean_reduction):.3f}"
)

predictive_mean_reduction = tfp.mcmc.potential_scale_reduction(
    hmc_net._base_predict(x_test, thinning=20).mean(), split_chains=False
)
print(
    f"Mean of Predictive means (test set): {tf.reduce_mean(predictive_mean_reduction):.3f}"
)
print(
    f"Max of Predictive means (test set): {tf.reduce_max(predictive_mean_reduction):.3f}"
)
print(
    f"Predictive stddev:{tfp.mcmc.potential_scale_reduction(hmc_net._base_predict(x_train[0:1, :], thinning=2).stddev())[0,0]:.3f}"
)
