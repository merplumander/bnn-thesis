# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from core.plotting_utils import (
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
n_train = 50
batchsize_train = n_train

x_train, y_train = create_split_periodic_data_heteroscedastic(
    n_data=n_train, sigma1=0.3, sigma2=0.3, seed=0
)
x_plot = create_x_plot(x_train)
# preprocessor = StandardPreprocessor()
# x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(_x_train)
y_ground_truth = ground_truth_periodic_function(x_plot)


layer_units = [50] + [1]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]


# %% codecell
y_lim = [-20, 20]
fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(x_plot, y_ground_truth, fig=fig, ax=ax)
ax.legend()

# %%
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=30, verbose=1, restore_best_weights=True
)
variational_network = VariationalDensityNetwork(
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=-20,
    transform_unconstrained_scale_factor=0.1,
    prior_scale_identity_multiplier=1,
    kl_weight=1 / (1e1 * n_train),
    preprocess_x=True,
    preprocess_y=True,
    learning_rate=0.01,
    seed=0,
)

variational_network.fit(
    x_train,
    y_train,
    epochs=250,
    verbose=1,
    validation_data=(x_train, y_train),
    early_stop_callback=early_stop_callback,
)

# %% markdown
# Notice that the average validation loss is generally much lower than the average
# training loss. This is due to me having explicitly removed the prior kl_divergence
# term from the validation evaluation, since I think for early stopping it is only
# relevant to monitor the fit to the (validation) data, and not the prior loss
# term. I don't think this is very controversial, if you expect your test data to come
# from the same distribution as your training data. If you expect some sort of dataset
# shift, you might think about including monitoring of this prior term for early
# stopping.


# %%
variational_network.evaluate(x_train, y_train)

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
variational_network.get_weights()
