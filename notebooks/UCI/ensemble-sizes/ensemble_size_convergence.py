# %load_ext autoreload
# %autoreload 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import yaml

from core.hmc import HMCDensityNetwork
from core.map import MapDensityEnsemble, MapDensityNetwork
from core.network_utils import make_independent_gaussian_network_prior
from core.plotting_utils import plot_uci_ensemble_size_benchmark
from core.uci_evaluation import uci_benchmark_ensemble_sizes_save_plot

tfd = tfp.distributions


figure_dir = "figures/temp"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)

# %%
with open("config/uci-hyperparameters-config.yaml") as f:
    experiment_config = yaml.full_load(f)


ensemble_n_networks = 50
weight_prior = tfd.Normal(0, experiment_config["weight_prior_scale"])
bias_prior = tfd.Normal(0, experiment_config["bias_prior_scale"])
a = experiment_config["noise_var_prior_a"]
b = experiment_config["noise_var_prior_b"]

_var_d = tfd.InverseGamma(a, b)
noise_scale_prior = tfd.TransformedDistribution(
    distribution=_var_d, bijector=tfp.bijectors.Invert(tfp.bijectors.Square())
)


patience = experiment_config["convergence_patience"]
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=patience, verbose=0, restore_best_weights=False
)

experiment_name = f"uci_ensemble-size-convergence_one-hidden-layer"
kwargs = dict(
    train_seed=experiment_config["train_seed"],
    layer_units=experiment_config["layer_units"],
    layer_activations=experiment_config["layer_activations"],
    initial_unconstrained_scale=experiment_config["initial_unconstrained_scale"],
    transform_unconstrained_scale_factor=experiment_config[
        "transform_unconstrained_scale_factor"
    ],
    learning_rate=experiment_config["learning_rate"],
    epochs=experiment_config["epochs"],
    batch_size=experiment_config["batch_size"],
    use_gap_data=False,
    experiment_name=experiment_name,
    model_save_dir=".save_uci_models",
    n_networks=ensemble_n_networks,
    early_stop_callback=early_stop_callback,
    weight_prior=weight_prior,
    bias_prior=bias_prior,
    noise_scale_prior=noise_scale_prior,
    last_layer_prior="weakly-informative",
    save=True,
    verbose=True,
)
print(kwargs["epochs"])

# %%
dataset = "boston"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "concrete"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "energy"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "kin8nm"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "naval"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "power"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "protein"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "wine"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "yacht"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)


# %% markdown
#  UCI Gap


# %%
kwargs["experiment_name"] = f"uci-gap_ensemble-size-convergence_one-hidden-layer"
kwargs["use_gap_data"] = (True,)

# %%
dataset = "boston"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "concrete"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "energy"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "kin8nm"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "naval"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "power"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "protein"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "wine"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "yacht"
results = uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)
