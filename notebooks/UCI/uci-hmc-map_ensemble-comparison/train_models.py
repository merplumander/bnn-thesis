# %load_ext autoreload
# %autoreload 2
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
    backtransform_constrained_scale,
    batch_repeat_matrix,
    hmc_to_map_weights,
    make_independent_gaussian_network_prior,
    map_to_hmc_weights,
    transform_unconstrained_scale,
)
from core.preprocessing import StandardPreprocessor
from core.sanity_checks import check_posterior_equivalence
from data.uci import load_uci_data

tfd = tfp.distributions


# %% markdown
#


# %%
dataset = "yacht"
data_seed = 0
dataset_name = f"{dataset}_{data_seed + 1:02}"

n_hidden_layers = 1
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
n_chains = 2

num_burnin_steps = 100000
num_results = 1
num_steps_between_results = 0  # 4 means only every fifth sample is saved
discard_burnin_samples = True

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
    discard_burnin_samples=discard_burnin_samples,
    step_size=step_size,
    num_leapfrog_steps=num_leapfrog_steps,
    max_tree_depth=10,
    num_steps_between_results=num_steps_between_results,
    seed=train_seed,
)

overdispersed_prior_samples = hmc_net.sample_prior_state(
    n_samples=n_chains,
    overdisp=1.2,
    seed=data_seed
    + hash(
        dataset
    ),  # make sure each dataset( and possibly data split) starts from different prior samples
)

prior_predictions = [
    hmc_net.predict_from_sample_parameters(
        x_train, tf.nest.map_structure(lambda x: x[i], overdispersed_prior_samples)
    )
    for i in range(n_chains)
]
[-prediction.log_prob(y_train) / n_train for prediction in prior_predictions]

current_state = overdispersed_prior_samples


# %% markdown
# ## Do a few gradient descent training epochs to help HMC find a region of higher posterior density

# %%
ensemble = MapDensityEnsemble(
    n_networks=n_chains,
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=-1,  # doesn't matter, will be overwritten
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    weight_prior=weight_prior,
    bias_prior=bias_prior,
    noise_scale_prior=noise_scale_prior,
    n_train=n_train,
    learning_rate=0.5,
)

assert check_posterior_equivalence(
    ensemble.networks[0], hmc_net, x_train, y_train, n_train=n_train
)

map_weights = hmc_to_map_weights(overdispersed_prior_samples)
# for weight in map_weights:
#     weight[-1] = backtransform_constrained_scale(0.014, transform_unconstrained_scale_factor).numpy()
ensemble.set_weights(map_weights)
[network.noise_sigma for network in ensemble.networks]
initial_unconstrained_scales = [map_weights[i][-1] for i in range(len(map_weights))]
predictions = [network.predict(x_train) for network in ensemble.networks]
print(
    "Initial samples' mean train negative log likelihood",
    [
        -tf.reduce_mean(prediction.log_prob(y_train)).numpy()
        for prediction in predictions
    ],
)

ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=10, verbose=0
)

predictions = [network.predict(x_train) for network in ensemble.networks]
print(
    "After gradients descent training mean train negative log likelihood",
    [
        -tf.reduce_mean(prediction.log_prob(y_train)).numpy()
        for prediction in predictions
    ],
)
[network.noise_sigma for network in ensemble.networks]
map_weights = ensemble.get_weights()
# for weight, scale in zip(map_weights, initial_unconstrained_scales):
#     weight[-1] = scale # backtransform_constrained_scale(0.014, transform_unconstrained_scale_factor).numpy() #backtransform_constrained_scale(np.array([0.1]), transform_unconstrained_scale_factor).numpy().astype(np.float32)
# ensemble.set_weights(map_weights)
# predictions = [network.predict(x_train) for network in ensemble.networks]
# print(
#     "After gradients descent training mean train negative log likelihood",
#     [-tf.reduce_mean(prediction.log_prob(y_train)).numpy() for prediction in predictions],
# )
print([network.noise_sigma for network in ensemble.networks])
current_state = map_to_hmc_weights(map_weights)


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
save_path = save_dir.joinpath(f"hmc-{dataset_name}")
# hmc_net.save(save_path)
################################################################################


# %% markdown
# # Loading

# %%
################################################################################
save_path = save_dir.joinpath(f"hmc-{dataset_name}")
hmc_net = hmc_density_network_from_save_path(
    save_path, network_prior=network_prior, noise_scale_prior=noise_scale_prior
)
################################################################################


# %% markdown
# ## Additional samples

# %%
################################################################################
# hmc_net.num_steps_between_results = 9
hmc_net.fit(x_train, y_train, num_results=50000, resume=True)
################################################################################

# %%

# %%
undo_masking(hmc_net)
mask_nonsense_chains(
    hmc_net,
    median_scale_cutter=1.0,
    lowest_acceptance_ratio=0.5,
    highest_acceptance_ratio=0.9,
    x_train=x_train,
    y_train=y_train,
    thinning=200,
)
hmc_net.samples[0].shape[:2]
hmc_net._chain[0].shape

# %%
acceptance_ratio = hmc_net.acceptance_ratio()
print(f"Acceptance ratio per chain: {acceptance_ratio}")

# %%
hmc_net.leapfrog_steps_taken()


# %%
print("Potential Scale Reduction\n")

weights_reduction = hmc_net.potential_scale_reduction()
print(
    f"Weight Space (Max per layer): {tf.nest.map_structure(lambda x: tf.reduce_max(x).numpy(), weights_reduction)}\n"
)


predictive_mean_reduction = tfp.mcmc.potential_scale_reduction(
    hmc_net._base_predict(x_train, thinning=100).mean(), split_chains=False
)
# sorted(predictive_mean_reduction)
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
    f"Predictive stddev:{tfp.mcmc.potential_scale_reduction(hmc_net._base_predict(x_train[0:1, :], thinning=100).stddev())[0,0]:.3f}"
)


# %% markdown
# ## Predictive distribution Wasserstein distance per chain

# %%
chain_predictions = [
    hmc_net.predict_individual_chain(x_train, i_chain=i_chain, thinning=100)
    for i_chain in range(hmc_net.n_used_chains)
]

for i in range(len(chain_predictions)):
    for j in range(len(chain_predictions))[i + 1 :]:
        print(
            predictive_distribution_wasserstein_distance(
                chain_predictions[i], chain_predictions[j], n_samples=10000, seed=0
            )
        )


# %%
chain_predictions = [
    hmc_net.predict_individual_chain(x_test, i_chain=i_chain, thinning=10)
    for i_chain in range(hmc_net.n_used_chains)
]
for i in range(len(chain_predictions)):
    for j in range(len(chain_predictions))[i + 1 :]:
        print(
            predictive_distribution_wasserstein_distance(
                chain_predictions[i], chain_predictions[j], n_samples=10000, seed=0
            )
        )


# %%
ess = hmc_net.effective_sample_size(thinning=100)
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
chain_predictions = [
    hmc_net.predict_individual_chain(x_test, i_chain=i_chain, thinning=20)
    for i_chain in range(hmc_net.n_used_chains)
]
[-prediction.log_prob(y_test) / n_test for prediction in chain_predictions]

[
    tf.math.sqrt(tf.reduce_mean((prediction.mean() - y_test) ** 2))
    for prediction in chain_predictions
]


# %%
prediction = hmc_net.predict(x_test, thinning=5)
print(f"negative test LL: {-prediction.log_prob(y_test) / n_test}")
print(f"RMSE: {rmse(prediction.mean(), y_test)}")


# %%
# Effective sample size of predictive distribution parameters
_prediction = hmc_net._base_predict(x_train, thinning=50)
predictive_samples = [_prediction.mean(), _prediction.stddev()[:, :, 0:1]]
cross_chain_dims = None if hmc_net.n_used_chains == 1 else [1, 1]
ess = tfp.mcmc.effective_sample_size(
    predictive_samples, cross_chain_dims=cross_chain_dims
)
print(
    int(tf.reduce_mean(ess[0]).numpy()),
    int(tf.math.reduce_std(ess[0]).numpy()),
    int(tf.reduce_min(ess[0]).numpy()),
    int(tf.reduce_max(ess[0]).numpy()),
)
print(int(ess[1][0, 0].numpy()))

_prediction = hmc_net._base_predict(x_test, thinning=50)
predictive_samples = [_prediction.mean(), _prediction.stddev()[:, :, 0:1]]
cross_chain_dims = None if hmc_net.n_used_chains == 1 else [1, 1]
ess = tfp.mcmc.effective_sample_size(
    predictive_samples, cross_chain_dims=cross_chain_dims
)
print(
    int(tf.reduce_mean(ess[0]).numpy()),
    int(tf.math.reduce_std(ess[0]).numpy()),
    int(tf.reduce_min(ess[0]).numpy()),
    int(tf.reduce_max(ess[0]).numpy()),
)


# %%
fig, ax = plt.subplots(figsize=(8, 8))
for i_chain in range(hmc_net.n_used_chains):
    scales = transform_unconstrained_scale(
        hmc_net._chain[-1][num_burnin_steps:, i_chain],
        factor=transform_unconstrained_scale_factor,
    )
    _x = np.linspace(0, 0.3, 1000)
    ax.hist(scales.numpy().flatten(), bins=500, density=False, alpha=0.2)
# ax.plot(_x, noise_scale_prior.prob(_x).numpy().flatten())

# %%
fig, ax = plt.subplots(figsize=(8, 8))
for i_chain in range(hmc_net.n_used_chains):
    weight = hmc_net.samples[2][:, i_chain, 1, 0]
    ax.hist(weight.numpy().flatten(), bins=40, density=True, alpha=0.2)


# %% markdown
# # Train one really large ensemble

# %%
ensemble_members = 1000

layer_names = [None] * (len(layer_units) - 2) + ["feature_extractor", "output"]

large_ensemble = MapDensityEnsemble(
    n_networks=ensemble_members,
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
    names=layer_names,
    seed=train_seed,
)

print("Done initializing")
assert check_posterior_equivalence(
    large_ensemble.networks[0], hmc_net, x_train, y_train, n_train
)

large_ensemble.fit(
    x_train=x_train,
    y_train=y_train,
    batch_size=batch_size,
    epochs=epochs,
    early_stop_callback=early_stop_callback,
    verbose=0,
)


# %%
large_ensemble.save(save_dir.joinpath(f"large_map_ensemble"))
