# %load_ext autoreload
# %autoreload 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from core.hmc import HMCDensityNetwork, hmc_density_network_from_save_path
from core.map import MapDensityNetwork
from core.network_utils import (
    activation_strings_to_activation_functions,
    build_scratch_model,
)
from core.plotting_utils import (
    plot_distribution_samples,
    plot_moment_matched_predictive_normal_distribution,
    plot_moment_matched_predictive_normal_distribution_and_function_samples,
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

figure_dir = "figures"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)

# %%
n_train = 20
_x_train, y_train = create_split_periodic_data_heteroscedastic(
    n_train=n_train, sigma1=2, sigma2=2, seed=42
)
x_train, _x_test, x_test = preprocess_create_x_train_test(_x_train)
y_test = ground_truth_periodic_function(_x_test)


# %% codecell
input_shape = [1]
layer_units = [20, 10, 2]  # [50, 30, 20, 10, 2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]
prior_scale = 3
weight_priors = [tfd.Normal(0, prior_scale)] * len(layer_units)
bias_priors = weight_priors

unique_hmc_save_path = f"hmc_n-units-{layer_units}_activations-{layer_activations}_prior-scale-{prior_scale}"
unique_hmc_save_path = save_dir.joinpath(unique_hmc_save_path)


# %% markdown
#### HMCDensityNetwork follows the fit and predict scheme:


# %%
rerun_training = False


# %%
sampler = "nuts"
num_burnin_steps = 1000
num_results = 10000
num_leapfrog_steps = 15
step_size = 0.1

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
        sampler=sampler,
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
    hmc_net = hmc_density_network_from_save_path(unique_hmc_save_path)

# %% markdown
# Standard prediction returns an equally weighted mixture of Gaussians. One Gaussian for each parameter setting in the chain.
# The Gaussians represent aleatoric uncertainty, while mixing these Gaussians as per the samples from the markov chain represents epistemic uncertainty.

# %%
mog_prediction = hmc_net.predict(x_test, thinning=10)  # Mixture Of Gaussian prediction
plot_moment_matched_predictive_normal_distribution(
    x_test=_x_test,
    predictive_distribution=mog_prediction,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-3, 10],
)


# %% markdown
# Plotting individual network predictions from the chain is also possible:

# %% codecell
n_predictions = 3
gaussian_predictions = hmc_net.predict_list_of_gaussians(
    x_test, n_predictions=n_predictions
)
plot_distribution_samples(
    x_test=_x_test,
    distribution_samples=gaussian_predictions,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-3, 10],
)


# %% markdown
# Or plotting overall uncertainty and some function samples

# %%
n_predictions = 5
mog_prediction = hmc_net.predict(x_test)  # Mixture Of Gaussian prediction
gaussian_predictions = hmc_net.predict_list_of_gaussians(
    x_test, n_predictions=n_predictions
)
plot_moment_matched_predictive_normal_distribution_and_function_samples(
    x_test=_x_test,
    predictive_distribution=mog_prediction,
    distribution_samples=gaussian_predictions,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-3, 10],
)


# %%
# resuming running the chain does not work yet
# net.fit(x_train, y_train, resume=True, num_burnin_steps=0, num_results=30, verbose=0)


# %% markdown
# ## You can also pre-train a density network by hand as initial state
# (if you don't provide an initial state via current_state a density network will be pre trained with standard parameters)

# %%
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
)

map_net = MapDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
)

map_net.fit(x_train=x_train, y_train=y_train, batch_size=20, epochs=200, verbose=0)

# %%
prediction = map_net.predict(x_test)
plot_moment_matched_predictive_normal_distribution(
    x_test=_x_test,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-3, 10],
)

# %%
initial_state = map_net.get_weights()
# initial_state


# %%
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
    num_burnin_steps=1000,
    num_results=2000,
    verbose=0,
)


# %%
mog_prediction = hmc_net.predict(x_test, thinning=2)  # Mixture Of Gaussian prediction
plot_moment_matched_predictive_normal_distribution(
    x_test=_x_test,
    predictive_distribution=mog_prediction,
    x_train=_x_train,
    y_train=y_train,
    y_test=y_test,
    y_lim=[-3, 10],
)
