# %load_ext autoreload
# %autoreload 2
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn import preprocessing

from core.map import MapDensityEnsemble, map_density_ensemble_from_save_path
from core.plotting_utils import (
    plot_distribution_samples,
    plot_ground_truth,
    plot_moment_matched_predictive_normal_distribution,
    plot_moment_matched_predictive_normal_distribution_and_function_samples,
    plot_training_data,
)
from core.preprocessing import StandardPreprocessor
from data.toy_regression import (
    create_split_periodic_data_heteroscedastic,
    ground_truth_periodic_function,
)

tfd = tfp.distributions


# %% codecell
figure_dir = "./figures"


# %% codecell
np.random.seed(0)
n_networks = 10
n_train = 20
batchsize_train = 20

# train and test variables beginning with an underscore are unprocessed.
_x_train, y_train = create_split_periodic_data_heteroscedastic(
    n_data=n_train, sigma1=0.2, sigma2=0.2, seed=42
)
preprocessor = StandardPreprocessor()
x_train, _x_plot, x_plot = preprocessor.preprocess_create_x_train_x_plot(_x_train)
y_ground_truth = ground_truth_periodic_function(_x_plot)

layer_units = [50, 20, 10] + [2]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]


# %% codecell
y_lim = [-8, 8]
fig, ax = plt.subplots(figsize=(8, 8))
plot_training_data(_x_train, y_train, fig=fig, ax=ax, y_lim=y_lim)
plot_ground_truth(_x_plot, y_ground_truth, fig=fig, ax=ax)
ax.legend()


# %%
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
)


ensemble = MapDensityEnsemble(
    n_networks=2,
    input_shape=[1],
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
    seed=0,
)


ensemble.fit(
    x_train=x_train, y_train=y_train, batch_size=batchsize_train, epochs=100, verbose=0
)


# %%
mog_prediction = ensemble.predict(x_plot)  # Mixture Of Gaussian prediction
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=mog_prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
)
# %%
ensemble.fit_additional_memebers(
    x_train=x_train,
    y_train=y_train,
    batch_size=batchsize_train,
    n_new_members=2,
    epochs=100,
    verbose=0,
)
mog_prediction = ensemble.predict(x_plot)  # Mixture Of Gaussian prediction
plot_moment_matched_predictive_normal_distribution(
    x_plot=_x_plot,
    predictive_distribution=mog_prediction,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
)


# %%
save_path = "._toy_ensemble_saving"
ensemble.save(save_path)

# %%
loaded_ensemble = map_density_ensemble_from_save_path(save_path)
ensemble = loaded_ensemble

# %% codecell
gaussian_predictions = ensemble.predict_list_of_gaussians(x_plot, n_predictions=3)
plot_distribution_samples(
    x_plot=_x_plot,
    distribution_samples=gaussian_predictions,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
)
# fig.savefig(os.path.join(figure_dir, f"{n_networks}_ml_density_ensemble_mixture_of_gaussian_heteroscedastic.pdf"))


# %% codecell
gaussian_predictions = ensemble.predict_list_of_gaussians(x_plot, n_predictions=5)
plot_moment_matched_predictive_normal_distribution_and_function_samples(
    x_plot=_x_plot,
    predictive_distribution=mog_prediction,
    distribution_samples=gaussian_predictions,
    x_train=_x_train,
    y_train=y_train,
    y_ground_truth=y_ground_truth,
    y_lim=y_lim,
)
