import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def plot_predictive_distribution(
    x_test,
    predictive_distribution,
    x_train=None,
    y_train=None,
    y_test=None,
    fig=None,
    ax=None,
    y_lim=None,
):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if y_test is not None:
        ax.plot(x_test, y_test, label="Ground truth", c="k", alpha=0.1)
    mean = predictive_distribution.mean().numpy()
    std = predictive_distribution.stddev().numpy()
    ax.plot(x_test, mean, label=f"Mean prediction", alpha=0.8)
    ax.fill_between(
        x_test.flatten(),
        mean.flatten() - 2 * std.flatten(),
        mean.flatten() + 2 * std.flatten(),
        alpha=0.2,
        label="95% HDR prediction",
    )
    if x_train is not None:
        ax.scatter(x_train, y_train, c="k", marker="x", s=100, label="Train data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if y_lim:
        ax.set_ylim(y_lim)
    ax.legend()
    return fig, ax


def plot_distribution_samples(
    x_test,
    distribution_samples,
    x_train=None,
    y_train=None,
    y_test=None,
    fig=None,
    ax=None,
    y_lim=None,
):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if y_test is not None:
        ax.plot(x_test, y_test, label="Ground truth", c="k", alpha=0.1)

    for i, prediction in enumerate(distribution_samples):
        mean = prediction.mean()
        std = prediction.stddev()
        c = sns.color_palette()[i]
        ax.plot(x_test, mean, label=f"Mean prediction", c=c, alpha=0.8)
        ax.fill_between(
            x_test.flatten(),
            tf.reshape(mean, [mean.shape[0]]) - 2 * tf.reshape(std, [std.shape[0]]),
            tf.reshape(mean, [mean.shape[0]]) + 2 * tf.reshape(std, [std.shape[0]]),
            color=c,
            alpha=0.15,
            label="95% HDR prediction",
        )
    if x_train is not None:
        ax.scatter(x_train, y_train, c="k", marker="x", s=100, label="Train data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if y_lim:
        ax.set_ylim(y_lim)
    ax.legend()
    return fig, ax


def plot_predictive_distribution_and_function_samples(
    x_test,
    predictive_distribution,
    distribution_samples,
    x_train=None,
    y_train=None,
    y_test=None,
    fig=None,
    ax=None,
    y_lim=None,
):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if y_test is not None:
        ax.plot(x_test, y_test, label="Ground truth", c="k", alpha=0.1)
    c = sns.color_palette()[1]
    for i, prediction in enumerate(distribution_samples[:-1]):
        mean = prediction.mean()
        ax.plot(x_test, mean, c=c, alpha=0.5)
    ax.plot(
        x_test,
        distribution_samples[-1].mean(),
        label=f"Function samples",
        c=c,
        alpha=0.5,
    )

    mean = predictive_distribution.mean().numpy()
    std = predictive_distribution.stddev().numpy()
    c = sns.color_palette()[0]
    ax.plot(x_test, mean, label=f"Mean prediction", c=c, alpha=1)
    ax.fill_between(
        x_test.flatten(),
        mean.flatten() - 2 * std.flatten(),
        mean.flatten() + 2 * std.flatten(),
        color=c,
        alpha=0.2,
        label="95% HDR prediction",
    )
    if x_train is not None:
        ax.scatter(x_train, y_train, c="k", marker="x", s=100, label="Train data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if y_lim:
        ax.set_ylim(y_lim)
    ax.legend()
    return fig, ax
