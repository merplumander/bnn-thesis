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
from core.uci_evaluation import kfold_evaluate_uci
from data.uci import load_uci_data

figure_dir = "figures/temp"
figure_dir = Path(figure_dir)
figure_dir.mkdir(parents=True, exist_ok=True)

# %%
# _rmse_metric = tf.keras.metrics.RootMeanSquaredError()
def calculate_rmse(mean_prediction, y):
    return tf.keras.metrics.RootMeanSquaredError()(mean_prediction, y)


with open("config/uci-hyperparameters-config.yaml") as f:
    kwargs = yaml.full_load(f)


# ensemble_n_networks = 26
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
        # "use_gap_data": False,
        # "experiment_name": experiment_name,
        # "model_save_dir": None, # ".save_uci_models",
        "n_networks": 10,
        "early_stop_callback": early_stop_callback,
        "preprocess_x": True,
        "preprocess_y": True,
        # "last_layer_prior": "standard-normal-weights-non-informative-scale",
        # "n_comb_max": n_comb_max,
    }
)
kwargs

kwargs.pop("batch_size")
kwargs.pop("train_seed")
kwargs.pop("epochs")
kwargs.pop("early_stop_callback")
fit_kwargs = {
    "batch_size": 100,
    "epochs": 100000,
    "early_stop_callback": early_stop_callback,
}
# %%

(_, _, ensemble_models, _, _) = kfold_evaluate_uci(
    model_class=MapDensityEnsemble,
    dataset="boston",
    use_gap_data=False,
    train_seed=0,
    validation_split=0.0,
    weight_prior_scale=weight_prior_scale,
    bias_prior_scale=weight_prior_scale,
    model_kwargs=kwargs,
    fit_kwargs=fit_kwargs,
    fit_kwargs_list=None,
    ensemble_predict_moment_matched=False  # If True the NLL will also be computed for
    # the moment matched predictive distribution and not just the mixture predictive
    # distribution.
)

# %%
x, y, train_indices, validation_indices, test_indices = load_uci_data(
    dataset="boston", validation_split=0.0, gap_data=False
)

dataset_splits_ensemble_rmses = []
dataset_splits_single_network_rmses = []
dataset_splits_ensemble_negative_log_likelihoods = []
dataset_splits_single_network_negative_log_likelihoods = []
for i, train, test in zip(range(len(train_indices)), train_indices, test_indices,):
    x_train = x[train]
    x_test = x[test]
    y_train = y[train].reshape(-1, 1)
    y_test = y[test].reshape(-1, 1)
    predictive_distribution = ensemble_models[i].predict(x_train)
    ensemble_rmse = calculate_rmse(predictive_distribution.mean(), y_train)
    ensemble_negative_log_likelihood = -tf.reduce_mean(
        predictive_distribution.log_prob(y_train)
    )
    dataset_splits_ensemble_rmses.append(ensemble_rmse)
    dataset_splits_ensemble_negative_log_likelihoods.append(
        ensemble_negative_log_likelihood
    )
    networks = ensemble_models[i].networks
    single_network_rmses = []
    single_network_negative_log_likelihoods = []
    for net in networks:
        predictive_distribution = net.predict(x_train)
        rmse = calculate_rmse(predictive_distribution.mean(), y_train)
        negative_log_likelihood = -tf.reduce_mean(
            predictive_distribution.log_prob(y_train)
        )
        single_network_rmses.append(rmse)
        single_network_negative_log_likelihoods.append(negative_log_likelihood)
    dataset_splits_single_network_rmses.append(single_network_rmses)
    dataset_splits_single_network_negative_log_likelihoods.append(
        single_network_negative_log_likelihoods
    )

# %%
ensemble_negative_log_likelihood
single_network_negative_log_likelihoods

# %%
ensemble_better_than_best_network = []
ensemble_better_than_average = []
for ensemble_rmse, single_network_rmses in zip(
    dataset_splits_ensemble_rmses, dataset_splits_single_network_rmses
):
    best_net = np.min(single_network_rmses)
    average = np.mean(single_network_rmses)
    ensemble_better_than_average.append((ensemble_rmse <= average))  # or np.isclose())
    ensemble_better_than_best_network.append(ensemble_rmse <= best_net)
print(
    f"The ensemble performed better than the best network on {np.sum(ensemble_better_than_best_network)} out of {len(ensemble_better_than_best_network)} dataset splits in terms of RMSE"
)
print(
    f"The ensemble performed better than the average of the member networks on {np.sum(ensemble_better_than_average)} out of {len(ensemble_better_than_average)} dataset splits in terms of RMSE"
)
print()

ensemble_better_than_best_network = []
ensemble_better_than_average = []
for ensemble_negative_log_likelihood, single_network_negative_log_likelihoods in zip(
    dataset_splits_ensemble_negative_log_likelihoods,
    dataset_splits_single_network_negative_log_likelihoods,
):
    best_net_ll = np.min(single_network_negative_log_likelihoods)
    average = np.mean(single_network_negative_log_likelihoods)
    ensemble_better_than_average.append(
        (ensemble_negative_log_likelihood <= average)
    )  # or np.isclose())
    ensemble_better_than_best_network.append(
        ensemble_negative_log_likelihood <= best_net_ll
    )
print(
    f"The ensemble performed better than the best network on {np.sum(ensemble_better_than_best_network)} out of {len(ensemble_better_than_best_network)} dataset splits in terms of NLL"
)
print(
    f"The ensemble performed better than the average of the member networks on {np.sum(ensemble_better_than_average)} out of {len(ensemble_better_than_average)} dataset splits in terms of NLL"
)


# %% markdown
# ## Is the ensemble better (or equal) than the best network?

# In 8 out of 20 splits the ensemble's RMSE is better than the best member network's RMSE on the full test dataset.
# In 6 out of 20 splits the ensemble's RMSE is better than the best member network's RMSE on the full train dataset.

# In 14 out of 20 splits the ensemble's NLL is better than the best member network's NLL on the full test dataset.
# In 0 out of 20 splits the ensemble's NLL is better than the best member network's NLL on the full train dataset.

# ### The ensemble is better (or equal) in terms of both RMSE and NLL than the average of the member network's RMSEs/NLLs on all test and train data splits.


# %%

# %%
dataset_split = 2
train = train_indices[dataset_split]
test = test_indices[dataset_split]
x_train = x[train]
x_test = x[test]
y_train = y[train].reshape(-1, 1)
y_test = y[test].reshape(-1, 1)
ensemble_rmses = []
ensemble_negative_log_likelihoods = []
single_rmses = []
single_negative_log_likelihoods = []
for _x, _y in zip(x_train, y_train):
    _x = _x.reshape(1, -1)
    predictive_distribution = ensemble_models[dataset_split].predict(_x)

    ensemble_rmse = calculate_rmse(predictive_distribution.mean(), _y)
    # print(predictive_distribution.mean(), _y)
    # break
    ensemble_negative_log_likelihood = -tf.reduce_mean(
        predictive_distribution.log_prob(_y)
    )
    ensemble_rmses.append(ensemble_rmse)
    ensemble_negative_log_likelihoods.append(ensemble_negative_log_likelihood)

    rmses = []
    negative_log_likelihoods = []
    networks = ensemble_models[dataset_split].networks
    _means = []
    for net in networks:
        predictive_distribution = net.predict(_x)
        rmse = calculate_rmse(predictive_distribution.mean(), _y)
        _mean = predictive_distribution.mean()
        _means.append(_mean)
        negative_log_likelihood = -tf.reduce_mean(predictive_distribution.log_prob(_y))
        rmses.append(rmse)
        negative_log_likelihoods.append(negative_log_likelihood)
    _mean_prediction = np.mean(_means)
    _average_rmse = calculate_rmse(_mean_prediction, _y)
    assert (_average_rmse <= np.mean(rmses)) or np.isclose(
        _average_rmse, np.mean(rmses)
    )
    # print(_average_rmse,   np.mean(rmses))
    # break
    single_rmses.append(rmses)
    single_negative_log_likelihoods.append(negative_log_likelihoods)

# %%
rmse_ensemble_best = []
rmse_ensemble_better_than_average = []
for ensemble_rmse, single_rmse in zip(ensemble_rmses, single_rmses):
    best_member = np.min(single_rmse)
    average = np.mean(single_rmse)
    # Doing the additional numpy isclose because there seem to be some small rounding issues when intermixing numpy and tensorflow functions (probably because of tensorflow using float32)
    rmse_ensemble_best.append(
        (ensemble_rmse <= best_member) or np.isclose(ensemble_rmse, best_member)
    )
    rmse_ensemble_better_than_average.append(
        (ensemble_rmse <= average) or np.isclose(ensemble_rmse, average)
    )
print(
    f"The ensemble's RMSE on single data points is better (or equal) than the best network's in {np.sum(rmse_ensemble_best)} out of {len(rmse_ensemble_best)} points"
)
print(
    f"The ensemble's RMSE on single data points is better (or equal) than the average network RMSE in {np.sum(rmse_ensemble_better_than_average)} out of {len(rmse_ensemble_better_than_average)} points"
)
print()

nll_ensemble_best = []
nll_ensemble_better_than_average = []
for ensemble_nll, single_nll in zip(
    ensemble_negative_log_likelihoods, single_negative_log_likelihoods
):
    best_member = np.min(single_nll)
    average = np.mean(single_nll)
    nll_ensemble_best.append(
        (ensemble_nll <= best_member) or np.isclose(ensemble_nll, best_member)
    )
    nll_ensemble_better_than_average.append(
        (ensemble_nll <= average) or np.isclose(ensemble_nll, average)
    )
print(
    f"The ensemble's NLL on single data points is better (or equal) than the best network's in {np.sum(nll_ensemble_best)} out of {len(nll_ensemble_best)} points"
)
print(
    f"The ensemble's NLL on single data points is better (or equal) than the average network NLL in {np.sum(nll_ensemble_better_than_average)} out of {len(nll_ensemble_better_than_average)} points"
)

# %% markdown
# ## Is the ensemble better (or equal) on single data points than the best member network?
# On single test data points the ensemble's RMSE is better in 1/2/2/3/3 out of 51 points (on the first 5 data splits)
# On single train data points the ensemble's RMSE is better in 34/32/19 out of 455 points (on the first 3 data splits).

# On single test data points to ensemble's NLL is better in 0/0/0/0/0 out of 51 points (on the first 5 data splits)
# On single train data points the ensemble's NLL is better in 0/0/0 out of 455 points (on the first 3 data splits)

# ### The ensemble is always better (or equal) in terms of both RMSE and NLL than the average of the member network's RMSEs/NLLs on single data points.
