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


patience = 20
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=patience, verbose=0, restore_best_weights=True
)
validation_split = 0.2
n_comb_max = 25

experiment_name = f"ensemble-sizes_n-comb-{n_comb_max}_early-stop-patience-{patience}_one-hidden-layer"
kwargs.update(
    {
        "use_gap_data": False,
        "experiment_name": experiment_name,
        "model_save_dir": ".save_uci_models",
        "n_networks": ensemble_n_networks,
        "early_stop_callback": early_stop_callback,
        "validation_split": validation_split,
        "n_comb_max": n_comb_max,
    }
)
print(kwargs)


# %%
dataset = "boston"
uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "concrete"
uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)

# %%
dataset = "energy"
uci_benchmark_ensemble_sizes_save_plot(dataset=dataset, **kwargs)
