# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler

from core.last_layer import PostHocLastLayerBayesianEnsemble as LLBEnsemble
from core.map import MapDensityEnsemble
from core.network_utils import prior_scale_to_regularization_lambda
from core.plotting_utils import (
    plot_distribution_samples,
    plot_ground_truth,
    plot_moment_matched_predictive_normal_distribution,
    plot_training_data,
)
from core.preprocessing import create_x_plot
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions

# %%
np.random.seed(0)
n_train = 80
batch_size = n_train
epochs = 20

# train and test variables beginning with an underscore are unprocessed.
x_train, y_train = create_split_periodic_data_heteroscedastic(
    n_data=n_train, sigma1=0.3, sigma2=0.5, seed=42
)
x_plot = create_x_plot(x_train)
# preprocessor = StandardPreprocessor()
# x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(_x_train)
y_ground_truth = ground_truth_periodic_function(x_plot)

input_shape = [1]
layer_units = [50, 50] + [1]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]
initial_unconstrained_scale = -2
transform_unconstrained_scale_factor = 0.5
preprocess_x = True
preprocess_y = True

# %%
y_lim = [-5, 5]
fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(x_plot, y_ground_truth, fig=fig, ax=ax)
ax.legend()

# %%
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)

# last layer bayesian network
llb_ensemble = LLBEnsemble(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=initial_unconstrained_scale,
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    preprocess_x=preprocess_x,
    preprocess_y=preprocess_y,
    learning_rate=lr_schedule,
    seed=0,
)


llb_ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)


prediction = llb_ensemble.predict(x_plot)
plot_moment_matched_predictive_normal_distribution(
    x_plot=x_plot,
    predictive_distribution=prediction,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    title="Neural Linear Ensemble",
    # save_path=figure_dir.joinpath(f"llb_moment_matched_{experiment_name}.pdf")
)

# %% markdown
# # Using pretrained networks


# %%
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=n_train, decay_rate=0.9, staircase=True
)
ensemble = MapDensityEnsemble(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=initial_unconstrained_scale,
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    preprocess_x=preprocess_x,
    preprocess_y=preprocess_y,
    learning_rate=lr_schedule,
    names=[None, "feature_extractor", "output"],
    seed=0,
)
ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batch_size, epochs=epochs, verbose=0
)
prediction = ensemble.predict(x_plot)
plot_moment_matched_predictive_normal_distribution(
    x_plot=x_plot,
    predictive_distribution=prediction,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
)

# %%
llb_ensemble = LLBEnsemble(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=initial_unconstrained_scale,
    transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
    preprocess_x=preprocess_x,
    preprocess_y=preprocess_y,
    learning_rate=lr_schedule,
    seed=0,
)


llb_ensemble.fit(
    x_train=x_train, y_train=y_train, pretrained_networks=ensemble.networks
)


prediction = llb_ensemble.predict(x_plot)
plot_moment_matched_predictive_normal_distribution(
    x_plot=x_plot,
    predictive_distribution=prediction,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    title="Neural Linear Ensemble",
    # save_path=figure_dir.joinpath(f"llb_moment_matched_{experiment_name}.pdf")
)


# %%
llb_ensemble.networks = llb_ensemble.networks[:1]
prediction = llb_ensemble.predict(x_plot)
plot_moment_matched_predictive_normal_distribution(
    x_plot=x_plot,
    predictive_distribution=prediction,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    title="Neural Linear Ensemble",
    # save_path=figure_dir.joinpath(f"llb_moment_matched_{experiment_name}.pdf")
)
