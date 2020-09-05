import numpy as np
import scipy.stats as stats


def ground_truth_linear_function(x, m=3, b=1):
    """toy linear equation with slope m and intercept b"""
    return m * x + b


def ground_truth_periodic_function(x, p=4, amplitude=2):
    """toy periodic function from fBNN paper"""
    return amplitude * np.sin(p * x)


def ground_truth_x3_function(x):
    offset = 2
    cosine_scaling = 1.5
    y = np.zeros_like(x)
    for i, _x in enumerate(x):
        if _x < -2:
            y[i] = 0.1 * (_x + offset) ** 3
        elif _x < 2:
            y[i] = (
                # 0.01 * stats.norm.pdf(_x, loc=-0.8, scale=0.3)
                # - 0.02 * stats.norm.pdf(_x, loc=0.8, scale=0.3)
                +cosine_scaling * np.cos((np.pi / 2) * (_x))
                + cosine_scaling
            )
        else:
            y[i] = 0.1 * (_x - offset) ** 3
    return y


def ground_truth_cos_function(x, period=2 * np.pi):
    return np.cos((period / (2 * np.pi)) * x)


def sample_x_locations(n_data, lower, upper, seed=0):
    np.random.seed(seed)
    x = np.random.uniform(lower, upper, n_data)
    return x


def sample_x_locations_gap(n_data, lower1, upper1, lower2, upper2, seed=0):
    n_half = int(n_data / 2)
    x_l = np.random.uniform(lower1, upper1, n_half)
    x_r = np.random.uniform(lower2, upper2, n_half)
    x = np.concatenate((x_l, x_r))
    return x_l, x_r, x


def add_noise(y, sigma=1, seed=0):
    np.random.seed(seed)
    return y + np.random.normal(scale=sigma, size=y.shape)


def create_linear_data(
    n_data, lower=-3, upper=3, m=3, b=1, sigma=1, seed=0, dtype=np.float32
):
    np.random.seed(seed)
    x = (
        sample_x_locations(n_data, lower, upper, seed=seed)
        .reshape(n_data, 1)
        .astype(dtype)
    )
    y = add_noise(
        ground_truth_linear_function(x, m=m, b=b), sigma=sigma, seed=seed
    ).astype(dtype)
    return x, y


def create_split_periodic_data(
    n_data,
    lower1=-2,
    upper1=-0.5,
    lower2=0.5,
    upper2=2,
    sigma=0.04,
    p=4,
    amplitude=2,
    seed=0,
    dtype=np.float32,
):
    """create x values with gap in between and corresponding y values from periodic
    function plus noise"""
    np.random.seed(seed)
    _, _, x = sample_x_locations_gap(n_data, lower1, upper1, lower2, upper2, seed=seed)
    x = x.reshape(-1, 1).astype(dtype)
    y = add_noise(
        ground_truth_periodic_function(x, p=p, amplitude=amplitude),
        sigma=sigma,
        seed=seed,
    ).astype(dtype)
    # shuffle
    random_indices = np.random.choice(x.shape[0], size=x.shape[0], replace=False)
    x = x[random_indices]
    y = y[random_indices]
    return x, y


def create_split_periodic_data_heteroscedastic(
    n_data,
    lower1=-2,
    upper1=-0.5,
    lower2=0.5,
    upper2=2,
    sigma1=0.01,
    sigma2=0.8,
    p=4,
    amplitude=2,
    seed=0,
    dtype=np.float32,
):
    """create x values with gap in between and corresponding y values from periodic
    function plus noise with two different scales"""
    np.random.seed(seed)
    x_l, x_r, x = sample_x_locations_gap(
        n_data, lower1, upper1, lower2, upper2, seed=seed
    )
    x = x.reshape(-1, 1).astype(dtype)
    y_l = add_noise(
        ground_truth_periodic_function(x_l, p=p, amplitude=amplitude),
        sigma=sigma1,
        seed=seed,
    )
    y_r = add_noise(
        ground_truth_periodic_function(x_r, p=p, amplitude=amplitude),
        sigma=sigma2,
        seed=seed,
    )
    y = np.concatenate((y_l, y_r)).reshape(-1, 1).astype(dtype)
    # shuffle
    random_indices = np.random.choice(x.shape[0], size=x.shape[0], replace=False)
    x = x[random_indices]
    y = y[random_indices]
    return x, y


def x3_gap_data(
    n_data,
    lower1=-4,
    upper1=-2,
    lower2=2,
    upper2=4,
    sigma=0.25,
    seed=0,
    dtype=np.float32,
):
    """
    Creates x values with gap in between and corresponding y values from 0.1 x^3 + e.
    e ~ N(0, sigma). Inspired by "Quality of Uncertainty Quantification for Bayesian
    Neural Network Inference" Yao et al. 2019.
    """
    np.random.seed(seed)
    _, _, x = sample_x_locations_gap(n_data, lower1, upper1, lower2, upper2, seed=seed)
    x = x.reshape(-1, 1).astype(dtype)
    y = ground_truth_x3_function(x)
    y = add_noise(y, sigma=sigma, seed=seed).astype(dtype)
    # shuffle
    random_indices = np.random.choice(x.shape[0], size=x.shape[0], replace=False)
    x = x[random_indices]
    y = y[random_indices]
    return x, y


def cos_linear_edge_data(
    n_data,
    period=2 * np.pi,
    lower=-2.5,
    upper=2.5,
    sigma=0.25,
    seed=0,
    dtype=np.float32,
):
    np.random.seed(seed)
    x = sample_x_locations(n_data, lower, upper, seed=seed)
    x = x.reshape(-1, 1).astype(dtype)
    y = ground_truth_cos_function(x, period=period)
    y = add_noise(y, sigma=sigma, seed=seed).astype(dtype)
    return x, y
