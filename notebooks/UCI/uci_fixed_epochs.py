# %load_ext autoreload
# %autoreload 2
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from core.plotting_utils import plot_uci_single_benchmark
from core.uci_evaluation import uci_benchmark_save_plot

figure_dir = "figures/temp"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)

# %%
train_seed = 0
ensemble_n_networks = 5
layer_units = [50] + [1]
layer_activations = ["relu"] * (len(layer_units) - 1) + ["linear"]
initial_unconstrained_scale = -1
transform_unconstrained_scale_factor = 0.5
learning_rate = 0.01

epochs = 40
batch_size = 100


# %%

experiment_name = "epochs-40_one-hidden-layer"

kwargs = {
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
