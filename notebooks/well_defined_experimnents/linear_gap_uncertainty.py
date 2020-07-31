# %% markdown
# ## Idea:
# einmal im 'gap' wenn beide Enden der Daten direkt aufeinander zeigen und einmal wenn die Daten z.B. einmal nach oben und einmal nach unten oder beide nach oben zeigen. Hintergrund Idee: 2019 Yao argumentiert bereits, dass Ensembles die predictive uncertainty unterschaetzen, wenn die functional diversity nicht hoch genug ist. Ich will zeigen, dass Ensembles in einem konkreten Beispiel ein Problem mit underestimation der predictive uncertainty haben. Die Idee ist, spaeter zu zeigen, dass LLB hier helfen kann indem es mehr predictive uncertainty an dieser Stelle hinzufuegt. Wenn ich zusaetzlich noch zeigen kann, dass Ensembles in bestimmten, wohl definierten Faellen replizierbar ein Problem mit functional diversity und damit mit underestimation der predictive uncertainty haben, dann ist das ein ziemlich nicer Bonus. Hypothese (ReLu Netzwerk) wenn es eine perfekte lineare Interpolation ueber den gap hinweg gibt, dann ist die functional diversity des Ensembles zu klein. In der thesis wuerde man dann vermutlich damit anfangen Nachteile von Ensembles aufzuzaehlen und dann unsere contribution hinzuzufuegen und dann spaeter auf ein konkretes Beispiel gehen um zu zeigen, dass LLB hier eine sinnvolle Ergaenzung waere. Und man sollte zeigen, dass HMC das Problem mit der underestimation der predictive uncertainty nicht hat.
# ## Experimental results:
# Ensembles do indeed show too little predictive uncertainty within a gap in the training data when the ends of the data point at each other (only tested for horizontal actually). However, this is not completely reliable and sometimes the predictive uncertainty is approaching a reasonable level. LLB networks also show this lack of predictive uncertainty in the gap (and a HMC Network does not show this, so it's not a problem of the model class). Therefore this is not a good example case to showcase the benefits of llb Ensembles and in fact it is evidence against the usefulness of llb Ensembles. Maybe it can be used as showcasing that LLB Ensembles still have shortcomings.


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
from data.toy_regression import ground_truth_x3_function, x3_gap_data

tfd = tfp.distributions

figure_dir = "figures/gap_functional_diversity"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)

# %%
data_seed = 0
n_train = 80


# train and test variables beginning with an underscore are unprocessed.
_x_train, y_train = x3_gap_data(
    n_data=n_train, lower1=-4.9, upper2=4.9, sigma=0.3, seed=data_seed
)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(
    _x_train, test_ds=0.05
)
y_ground_truth = ground_truth_x3_function(_x_plot)

layer_units = [50, 20] + [2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]


# %% codecell
y_lim = [-5, 5]
fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(_x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(_x_plot, y_ground_truth, fig=fig, ax=ax, alpha=0.2)
ax.legend()
# fig.savefig(figure_dir.joinpath("data_x3_gap.pdf"))


# %%
# General training
train_seed = 0
epochs = 150
batch_size = n_train


# %% markdown
# # MAP Density Model

# %%
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)


net = MapDensityNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=train_seed,
)

net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)

# %%
prediction = net.predict(x_plot)  # Mixture Of Gaussian prediction
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    # save_path=figure_dir.joinpath(f"map_network_x3_gap.pdf")
)

# %% markdown
# # Density Ensemble

# %%
n_networks = 10

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
    seed=train_seed,
)

ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


# %%
prediction = ensemble.predict(x_plot)  # Mixture Of Gaussian prediction
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    # save_path=figure_dir.joinpath(f"ensemble_moment_matched_x3_gap.pdf")
)


# %% codecell
gaussian_predictions = ensemble.predict_list_of_gaussians(x_plot, n_predictions=3)
plot_distribution_samples(
    x_plot=_x_plot,
    distribution_samples=gaussian_predictions,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    # save_path=figure_dir.joinpath(f"ensemble_members_x3_gap.pdf")
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
    seed=train_seed,
)
layer_units[-1] = 2

llb_net.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)

# %%
prediction = llb_net.predict(x_plot)
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    # save_path=figure_dir.joinpath(f"llb_moment_matched_x3_gap.pdf")
)


# %% markdown
# # HMC

# %%
save_dir = ".save_data/"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# %%
prior_scale = 1
weight_priors = [tfd.Normal(0, prior_scale)] * len(layer_units)
bias_priors = weight_priors

unique_hmc_save_path = f"hmc_n-units-{layer_units}_activations-{layer_activations}_prior-scale-{prior_scale}_data-x3-gap_unconstrained-scale"
unique_hmc_save_path = save_dir.joinpath(unique_hmc_save_path)

# %%
rerun_training = False


# %%
sampler = "hmc"
num_burnin_steps = 500
num_results = 2000
num_leapfrog_steps = 25
step_size = 0.1

if rerun_training or not unique_hmc_save_path.is_file():
    hmc_net = HMCDensityNetwork(
        [1],
        layer_units,
        layer_activations,
        weight_priors=weight_priors,
        bias_priors=bias_priors,
        sampler=sampler,
        seed=0,
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
    hmc_net = hmc_density_network_from_save_path(unique_hmc_save_path)


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
    seed=train_seed,
)
layer_units[-1] = 2

llb_ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


prediction = llb_ensemble.predict(x_plot)
distribution_samples = llb_ensemble.predict_list(x_plot)
plot_moment_matched_predictive_normal_distribution_and_function_samples(
    x_plot=_x_plot,
    predictive_distribution=prediction,
    distribution_samples=distribution_samples,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    # save_path=figure_dir.joinpath(f"llb-ensemble_x3_gap.pdf")
)
