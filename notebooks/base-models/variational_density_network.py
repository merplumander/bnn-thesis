# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from core.plotting_utils import (
    plot_distribution_samples,
    plot_ground_truth,
    plot_moment_matched_predictive_normal_distribution,
    plot_moment_matched_predictive_normal_distribution_and_function_samples,
    plot_training_data,
)
from core.preprocessing import create_x_plot
from core.variational import VariationalDensityNetwork
from data.toy_regression import (
    create_linear_data,
    create_split_periodic_data_heteroscedastic,
    ground_truth_linear_function,
    ground_truth_periodic_function,
)

tfd = tfp.distributions


# %% codecell
np.random.seed(0)
n_train = 100
batchsize_train = n_train

x_train, y_train = create_split_periodic_data_heteroscedastic(
    n_data=n_train, sigma1=0.3, sigma2=0.3, seed=0
)
x_plot = create_x_plot(x_train)
# preprocessor = StandardPreprocessor()
# x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(_x_train)
y_ground_truth = ground_truth_periodic_function(x_plot)


layer_units = [50, 50] + [1]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]

_var_d = tfd.InverseGamma(0.5, 0.01)
noise_scale_prior = tfd.TransformedDistribution(
    distribution=_var_d, bijector=tfp.bijectors.Invert(tfp.bijectors.Square())
)

# %% codecell
y_lim = [-12, 12]
fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(x_plot, y_ground_truth, fig=fig, ax=ax)
ax.legend()

# %%
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=20, verbose=1, restore_best_weights=False
)
variational_network = VariationalDensityNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=-10,
    transform_unconstrained_scale_factor=0.5,
    prior_scale_identity_multiplier=1,
    kl_weight=1 / (1e0 * n_train),
    noise_scale_prior=noise_scale_prior,
    n_train=n_train,
    preprocess_x=True,
    preprocess_y=True,
    learning_rate=0.01,
    evaluate_ignore_prior_loss=False,
    seed=0,
)

variational_network.fit(
    x_train, y_train, epochs=2000, verbose=0, early_stop_callback=early_stop_callback
)


# %%
# It's easy to get, change, and set the variational posterior's parameters to test their individual effect
# weights = variational_network.get_weights()
# weights
# weights[3][:] = np.array([-1., 2., 10., 10.])
# weights[3]
# weights[4][:] = -10
# variational_network.set_weights(weights)


# %%
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
plot_distribution_samples(
    x_plot=x_plot,
    distribution_samples=gaussian_predictions,
    x_train=x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
    no_ticks=False,
)
