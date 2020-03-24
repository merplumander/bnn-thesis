import numpy as np
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


def preprocess_create_x_train_test(
    unprocessed_x_train, remove_mean=True, divide_std=True, n_test=500
):
    """
    Create unprocessed x_test and then preprocess unprocessed x_train and x_test.

    Returns:
        x_train:            preprocessed unprocessed_x_train
        unprocessed_x_test: unprocessed x for testing. Computes its range from
                            unprocessed_x_train
        x_test:             preprocessed x_test
    """
    n_features = unprocessed_x_train.shape[1]
    dtype = unprocessed_x_train.dtype
    x_min, x_max = np.min(unprocessed_x_train), np.max(unprocessed_x_train)
    d = x_max - x_min
    lower_bound = x_min - d / 2
    upper_bound = x_max + d / 2
    unprocessed_x_test = (
        np.linspace(lower_bound, upper_bound, n_test)
        .reshape(-1, n_features)
        .astype(dtype)
    )
    preprocessor = StandardPreprocessor(remove_mean=remove_mean, divide_std=divide_std)
    preprocessor.fit(unprocessed_x_train)
    x_train = preprocessor.transform(unprocessed_x_train)
    x_test = preprocessor.transform(unprocessed_x_test)
    return x_train, unprocessed_x_test, x_test
