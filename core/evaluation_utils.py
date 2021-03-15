import numpy as np
import ot
import tensorflow as tf
from scipy.stats import wasserstein_distance

from core.preprocessing import StandardPreprocessor
from data.uci import load_uci_data


def normalize_rmse(rmse, y_normalization_scale):
    return rmse / y_normalization_scale


def backtransform_normalized_rmse(normalized_rmse, y_normalization_scale):
    return normalized_rmse * y_normalization_scale


def normalize_ll(ll, y_normalization_scale):
    log_scale = np.log(y_normalization_scale)
    return ll - log_scale


def backtransform_normalized_ll(normalized_ll, y_normalization_scale):
    log_scale = np.log(y_normalization_scale)
    return normalized_ll + log_scale


def get_y_normalization_scales(dataset, gap_data=False):
    _x, _y, train_indices, _, test_indices = load_uci_data(
        f"{dataset}", gap_data=gap_data
    )
    y_normalization_scales = []
    for i_split in range(len(train_indices)):
        _y_train = _y[train_indices[i_split]].reshape(-1, 1)
        y_preprocessor = StandardPreprocessor()
        y_preprocessor.fit(_y_train)
        y_normalization_scales.append(y_preprocessor.scaler.scale_)
    return np.array(y_normalization_scales)


def rmse(prediction, y):
    return tf.math.sqrt(tf.reduce_mean((prediction - y) ** 2))


def wasserstein_distance_predictive_samples(predictive_samples1, predictive_samples2):
    wds = []
    for i_test_point in range(predictive_samples1.shape[1]):
        _samples1 = predictive_samples1[:, i_test_point]
        _samples2 = predictive_samples2[:, i_test_point]
        wd = wasserstein_distance(_samples1, _samples2)
        wds.append(wd)
    return np.mean(wds)


def predictive_distribution_wasserstein_distance(
    predictive_distribution1, predictive_distribution2, n_samples=1000, seed=0
):
    predictive_samples1 = np.squeeze(
        predictive_distribution1.sample(n_samples, seed=seed).numpy()
    )
    predictive_samples2 = np.squeeze(
        predictive_distribution2.sample(n_samples, seed=seed + 1).numpy()
    )
    wds = []
    for i_test_point in range(predictive_samples1.shape[1]):
        samples1 = predictive_samples1[:, i_test_point]
        samples2 = predictive_samples2[:, i_test_point]
        # ot's  Wasserstein distance is about 30% faster than scipy's
        # wd = wasserstein_distance(samples1, samples2)
        wd = ot.wasserstein_1d(samples1, samples2)
        wds.append(wd)
    return np.mean(wds)
