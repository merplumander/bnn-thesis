import numpy as np


def ground_truth_linear_function(x, m=3, b=1):
    """toy linear equation with slope m and intercept b"""
    return m * x + b


def ground_truth_periodic_function(x):
    """toy periodic function from fBNN paper"""
    return 2 * np.sin(4 * x)


def add_noise(y, sigma=1, seed=0):
    np.random.seed(seed)
    return y + np.random.normal(scale=sigma, size=y.shape)


def create_linear_data(
    n_train, lower=-3, upper=3, m=3, b=1, sigma=1, seed=0, dtype=np.float32
):
    np.random.seed(seed)
    train_x = np.random.uniform(lower, upper, (n_train, 1)).astype(dtype)
    train_y = add_noise(
        ground_truth_linear_function(train_x, m=m, b=b), sigma=sigma, seed=seed
    ).astype(dtype)
    return train_x, train_y


def create_split_periodic_data(
    n_train,
    lower1=-2,
    upper1=-0.5,
    lower2=0.5,
    upper2=2,
    sigma=0.04,
    seed=0,
    dtype=np.float32,
):
    """create x values with gap in between and corresponding y values from periodic
    function plus noise"""
    np.random.seed(seed)
    n_train_half = int(n_train / 2)
    train_x = (
        np.concatenate(
            (
                np.random.uniform(lower1, upper1, n_train_half),
                np.random.uniform(lower2, upper2, n_train_half),
            )
        )
        .reshape(-1, 1)
        .astype(dtype)
    )
    train_y = add_noise(
        ground_truth_periodic_function(train_x), sigma=sigma, seed=seed
    ).astype(dtype)
    return train_x, train_y


def create_split_periodic_data_heteroscedastic(
    n_train,
    lower1=-2,
    upper1=-0.5,
    lower2=0.5,
    upper2=2,
    sigma1=0.01,
    sigma2=0.8,
    seed=0,
    dtype=np.float32,
):
    """create x values with gap in between and corresponding y values from periodic
    function plus noise with two different scales"""
    np.random.seed(seed)
    n_train_half = int(n_train / 2)
    train_x_l = np.random.uniform(lower1, upper1, n_train_half)
    train_x_r = np.random.uniform(lower2, upper2, n_train_half)
    train_x = np.concatenate((train_x_l, train_x_r)).reshape(-1, 1).astype(dtype)
    train_y_l = add_noise(
        ground_truth_periodic_function(train_x_l), sigma=sigma1, seed=seed
    )
    train_y_r = add_noise(
        ground_truth_periodic_function(train_x_r), sigma=sigma2, seed=seed
    )
    train_y = np.concatenate((train_y_l, train_y_r)).reshape(-1, 1).astype(dtype)
    return train_x, train_y
