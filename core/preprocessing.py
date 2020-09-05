import numpy as np
import tensorflow as tf
from sklearn import preprocessing


class StandardPreprocessor:
    def __init__(self, remove_mean=True, divide_std=True):
        self.remove_mean = remove_mean
        self.divide_std = divide_std
        self.scaler = preprocessing.StandardScaler(
            with_mean=self.remove_mean, with_std=self.divide_std
        )

    def fit(self, unprocessed_x_train):
        self.scaler.fit(unprocessed_x_train)

    def transform(self, x):
        return self.scaler.transform(x)

    def fit_transform(self, x):
        return self.scaler.fit_transform(x)

    def inverse_transform(self, x):
        return self.scaler.inverse_transform(x)

    def preprocess_create_x_train_x_plot(
        self, unprocessed_x_train, n_test=500, test_ds=0.5
    ):
        """
        Create unprocessed x_plot and then preprocess unprocessed x_train and x_plot.

        Args:
            test_ds (float):    Determines the range of unprocessed_x_plot.
                                Determines how many ds (d = x_max - x_min) lower from x_min
                                (and higher from x_max) unprocessed_x_test ranges.

        Returns:
            x_train:            preprocessed unprocessed_x_train
            unprocessed_x_plot: unprocessed x for plotting. Computes its range from
                                unprocessed_x_train
            x_plot              preprocessed x_plot
        """
        n_features = unprocessed_x_train.shape[1]
        dtype = unprocessed_x_train.dtype
        x_min, x_max = np.min(unprocessed_x_train), np.max(unprocessed_x_train)
        d = x_max - x_min
        lower_bound = x_min - d * test_ds
        upper_bound = x_max + d * test_ds
        unprocessed_x_plot = (
            np.linspace(lower_bound, upper_bound, n_test)
            .reshape(-1, n_features)
            .astype(dtype)
        )
        self.fit(unprocessed_x_train)
        x_train = self.transform(unprocessed_x_train)
        x_plot = self.transform(unprocessed_x_plot)
        return x_train, unprocessed_x_plot, x_plot


class StandardizePreprocessor:
    def __init__(self, remove_mean=True, divide_std=True):
        self.remove_mean = remove_mean
        self.divide_std = divide_std
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = tf.reduce_mean(x, axis=0)
        self.std = tf.math.reduce_std(x, axis=0)

    def transform(self, x):
        return ((x - self.mean) / self.std).numpy()

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x).numpy()

    def inverse_transform(self, x):
        return (x * self.std + self.mean).numpy()


def create_x_plot(x_train, n_plot=500, plot_ds=0.5):
    n_features = x_train.shape[1]
    dtype = x_train.dtype
    x_min, x_max = np.min(x_train), np.max(x_train)
    d = x_max - x_min
    lower_bound = x_min - d * plot_ds
    upper_bound = x_max + d * plot_ds
    x_plot = (
        np.linspace(lower_bound, upper_bound, n_plot)
        .reshape(-1, n_features)
        .astype(dtype)
    )
    return x_plot
