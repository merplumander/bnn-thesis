import numpy as np
import tensorflow as tf
from scipy.stats import wasserstein_distance


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
        wd = wasserstein_distance(samples1, samples2)
        wds.append(wd)
    return np.mean(wds)
