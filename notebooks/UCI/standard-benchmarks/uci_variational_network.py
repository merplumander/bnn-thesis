# %load_ext autoreload
# %autoreload 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import yaml

from core.network_utils import prior_scale_to_regularization_lambda
from core.plotting_utils import plot_uci_single_benchmark
from core.uci_evaluation import kfold_evaluate_uci, save_results
from core.variational import VariationalDensityNetwork

tfd = tfp.distributions


# %%
def save_plot_results(experiment_name, dataset, results, use_gap_data):
    plot_uci_single_benchmark(model_results=[results[0]], labels=["VI"], y_label="RMSE")
    plot_uci_single_benchmark(
        model_results=[results[1]], labels=["VI"], y_label="Negative Log Likelihood"
    )
    _results = {}
    _results["VI"] = {
        "RMSEs": results[0],
        "NLLs": results[1],
        "total_epochs": results[3],
        "fit_times": results[4],
    }
    save_results(experiment_name, dataset, _results, use_gap_data=use_gap_data)


# %%
n_hidden_layers = 1
hidden_layers_string = (
    "two-hidden-layers" if n_hidden_layers == 2 else "one-hidden-layer"
)

with open("config/uci-hyperparameters-config.yaml") as f:
    experiment_config = yaml.full_load(f)


layer_units = experiment_config["layer_units"]
layer_activations = experiment_config["layer_activations"]
if n_hidden_layers == 2:
    layer_units.insert(0, layer_units[0])
    layer_activations.insert(0, layer_activations[0])

weight_prior = tfd.Normal(0, experiment_config["weight_prior_scale"])
bias_prior = tfd.Normal(0, experiment_config["bias_prior_scale"])
a = experiment_config["noise_var_prior_a"]
b = experiment_config["noise_var_prior_b"]
_var_d = tfd.InverseGamma(a, b)
noise_scale_prior = tfd.TransformedDistribution(
    distribution=_var_d, bijector=tfp.bijectors.Invert(tfp.bijectors.Square())
)
initial_unconstrained_scale = -1  # experiment_config["initial_unconstrained_scale"],


patience = experiment_config["convergence_patience"]
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=patience, verbose=1, restore_best_weights=False
)

experiment_name = f"uci_vi_initial-unconstrained-scale-{initial_unconstrained_scale}_{hidden_layers_string}"
model_kwargs = dict(
    layer_units=layer_units,
    layer_activations=layer_activations,
    initial_unconstrained_scale=initial_unconstrained_scale,
    transform_unconstrained_scale_factor=experiment_config[
        "transform_unconstrained_scale_factor"
    ],
    prior_scale_identity_multiplier=experiment_config["weight_prior_scale"],
    noise_scale_prior=noise_scale_prior,
    preprocess_x=True,
    preprocess_y=True,
    learning_rate=experiment_config["learning_rate"],
    evaluate_ignore_prior_loss=False,
)
fit_kwargs = dict(
    epochs=experiment_config["epochs"],
    batch_size=experiment_config["batch_size"],
    early_stop_callback=early_stop_callback,
)


use_gap_data = True


# %%
datasets = [
    "boston",
    "concrete",
    "energy",
    "kin8nm",
    "naval",
    "power",
    "protein",
    "wine",
    "yacht",
]

# %%
for dataset in datasets:
    results = kfold_evaluate_uci(
        VariationalDensityNetwork,
        dataset=dataset,
        use_gap_data=use_gap_data,
        train_seed=experiment_config["train_seed"],
        model_kwargs=model_kwargs,
        fit_kwargs=fit_kwargs,
    )  # (rmses, negative_log_likelihoods, models, total_epochs, fit_times)
    results = list(results)

    save_plot_results(experiment_name, dataset, results, use_gap_data=use_gap_data)
    print(f"Done {dataset}")
