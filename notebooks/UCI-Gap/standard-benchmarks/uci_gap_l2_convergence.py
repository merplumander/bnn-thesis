# %load_ext autoreload
# %autoreload 2
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from core.network_utils import prior_scale_to_regularization_lambda
from core.plotting_utils import plot_uci_single_benchmark
from core.uci_evaluation import uci_benchmark_save_plot

figure_dir = "figures/temp"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)

# %%
train_seed = 0
ensemble_n_networks = 5
layer_units = [50, 1]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]
initial_unconstrained_scale = -1
transform_unconstrained_scale_factor = 0.5
learning_rate = 0.01

epochs = int(1e5)
batch_size = 100

patience = 20
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=patience, verbose=0, restore_best_weights=True
)

weight_prior_scale = 1
bias_prior_scale = weight_prior_scale
# l2_weight_lambda = prior_scale_to_regularization_lambda(weight_prior_scale, n_train)
# l2_bias_lambda = prior_scale_to_regularization_lambda(bias_prior_scale, n_train)

experiment_name = (
    f"l2-reg-prior-scale-{weight_prior_scale}-patience-{patience}_one-hidden-layer"
)

n_features = layer_units[-1] + 1
kwargs = {
    "use_gap_data": True,
    "experiment_name": experiment_name,
    "figure_dir": figure_dir,
    "train_seed": train_seed,
    "ensemble_n_networks": ensemble_n_networks,
    "layer_activations": layer_activations,
    "initial_unconstrained_scale": initial_unconstrained_scale,
    "transform_unconstrained_scale_factor": transform_unconstrained_scale_factor,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "early_stop_callback": early_stop_callback,
    "weight_prior_scale": weight_prior_scale,
    "bias_prior_scale": bias_prior_scale,
    "vi_flat_prior": True,
    "evaluate_ignore_prior_loss": False,
    "last_layer_prior": "standard-normal-weights-non-informative-scale",
    "save": True
    # "validation_split": validation_split,
}

# %%
dataset = "boston"
uci_benchmark_save_plot(dataset=dataset, layer_units=layer_units, **kwargs)


# %%
dataset = "concrete"
uci_benchmark_save_plot(dataset=dataset, layer_units=layer_units, **kwargs)


# %%
dataset = "energy"
uci_benchmark_save_plot(dataset=dataset, layer_units=layer_units, **kwargs)


# %%
dataset = "kin8nm"
uci_benchmark_save_plot(dataset=dataset, layer_units=layer_units, **kwargs)


# %%
dataset = "naval"
uci_benchmark_save_plot(dataset=dataset, layer_units=layer_units, **kwargs)


# %%
dataset = "power"
uci_benchmark_save_plot(dataset=dataset, layer_units=layer_units, **kwargs)


# %%
dataset = "protein"
uci_benchmark_save_plot(dataset=dataset, layer_units=[100, 1], **kwargs)


# %%
dataset = "wine"
uci_benchmark_save_plot(dataset=dataset, layer_units=layer_units, **kwargs)


# %%
dataset = "yacht"
uci_benchmark_save_plot(dataset=dataset, layer_units=layer_units, **kwargs)
