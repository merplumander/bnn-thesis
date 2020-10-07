# %load_ext autoreload
# %autoreload 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml

from core.map import MapDensityEnsemble
from core.plotting_utils import plot_uci_ensemble_size_benchmark
from core.uci_evaluation import uci_benchmark_ensemble_sizes_save_plot

figure_dir = "figures/temp"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)

# %%
with open("config/uci-hyperparameters-config.yaml") as f:
    kwargs = yaml.full_load(f)


ensemble_n_networks = 26
weight_prior_scale = 1
bias_prior_scale = weight_prior_scale


patience = 20
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="loss", patience=patience, verbose=0, restore_best_weights=True
)
n_comb_max = 25

experiment_name = f"ensemble-sizes_l2-convergence_n-comb-{n_comb_max}_early-stop-patience-{patience}_one-hidden-layer"
kwargs.update(
    {
        "use_gap_data": True,
        "experiment_name": experiment_name,
        "model_save_dir": None,  # ".save_uci_models", #saving the models doesn't work right now
        "n_networks": ensemble_n_networks,
        "early_stop_callback": early_stop_callback,
        "weight_prior_scale": weight_prior_scale,
        "bias_prior_scale": bias_prior_scale,
        "last_layer_prior": "standard-normal-weights-non-informative-scale",
        "n_comb_max": n_comb_max,
    }
)
print(kwargs)


# %% markdown
# # Benchmarking

# %%
dataset = "boston"
uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "concrete"
uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "energy"
uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "kin8nm"
uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "naval"
uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "power"
uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
layer_units = kwargs.pop("layer_units")
dataset = "protein"
uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, layer_units=[100, 1], **kwargs)
kwargs["layer_units"] = layer_units

# %%
dataset = "wine"
uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "yacht"
uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)
