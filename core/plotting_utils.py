import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.neighbors import KernelDensity


def _normalize_distribution(x_plot, distribution, kde):
    pdf = distribution.prob(x_plot)
    highest_pdf = np.max(pdf)
    mean = distribution.mean().numpy().reshape(1, 1)
    kde_height = np.exp(kde.score_samples(mean))
    normalized_pdf = pdf * kde_height / highest_pdf
    return normalized_pdf


def plot_training_data(x, y, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, c="k", marker="x", s=100, label="Train data")
    return fig, ax


def plot_validation_data(x, y, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        x, y, c="g", marker="x", s=100, label="Validation data",
    )
    return fig, ax


def plot_test_data(x, y, c="k", alpha=0.1, label="Test data", fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y, c=c, alpha=alpha, label=label)
    return fig, ax


def plot_moment_matched_predictive_normal_distribution(
    x_test,
    predictive_distribution,
    x_train=None,
    y_train=None,
    x_validation=None,
    y_validation=None,
    y_test=None,
    fig=None,
    ax=None,
    y_lim=None,
    label="",
    save_path=None,
):
    """
    Plots the moment-matched Normal distribution to the passed predictive distribution.
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if y_test is not None:
        plot_test_data(x_test, y_test, ax=ax)
    mean = predictive_distribution.mean().numpy()
    std = predictive_distribution.stddev().numpy()
    label = label if label else "Mean prediction"
    ax.plot(x_test, mean, label=label, alpha=0.8)
    ax.fill_between(
        x_test.flatten(),
        mean.flatten() - 2 * std.flatten(),
        mean.flatten() + 2 * std.flatten(),
        alpha=0.2,
        label=f"95% HDR prediction",
    )
    if x_train is not None:
        plot_training_data(x_train, y_train, ax=ax)
    if x_validation is not None:
        plot_validation_data(x_validation, y_validation, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if y_lim:
        ax.set_ylim(y_lim)
    ax.legend()
    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_distribution_samples(
    x_test,
    distribution_samples,
    x_train=None,
    y_train=None,
    x_validation=None,
    y_validation=None,
    y_test=None,
    fig=None,
    ax=None,
    y_lim=None,
    seaborn_pallet_starting_index=0,
    labels=[],
    save_path=None,
):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if y_test is not None:
        plot_test_data(x_test, y_test, ax=ax)
    for i, prediction in enumerate(distribution_samples):
        mean = prediction.mean()
        std = prediction.stddev()
        c = sns.color_palette()[i + seaborn_pallet_starting_index]
        label = labels[i] if labels else "Mean prediction"
        ax.plot(x_test, mean, label=label, c=c, alpha=0.8)
        ax.fill_between(
            x_test.flatten(),
            tf.reshape(mean, [mean.shape[0]]) - 2 * tf.reshape(std, [std.shape[0]]),
            tf.reshape(mean, [mean.shape[0]]) + 2 * tf.reshape(std, [std.shape[0]]),
            color=c,
            alpha=0.15,
            label=f"95% HDR prediction",
        )
    if x_train is not None:
        plot_training_data(x_train, y_train, ax=ax)
    if x_validation is not None:
        plot_validation_data(x_validation, y_validation, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if y_lim:
        ax.set_ylim(y_lim)
    ax.legend()
    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_function_samples(
    x_test,
    distribution_samples,
    c=None,
    x_train=None,
    y_train=None,
    x_validation=None,
    y_validation=None,
    y_test=None,
    fig=None,
    ax=None,
    y_lim=None,
    label="",
    save_path=None,
):
    """
    Plots the means of the passed distributions (i.e. distribution_samples)
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if y_test is not None:
        plot_test_data(x_test, y_test, ax=ax)
    if c is None:
        c = sns.color_palette()[1]
    for i, prediction in enumerate(distribution_samples):
        mean = prediction.mean()
        (line,) = ax.plot(x_test, mean, c=c, alpha=0.2)
    line.set_label("Function samples")
    if x_train is not None:
        plot_training_data(x_train, y_train, ax=ax)
    if x_validation is not None:
        plot_validation_data(x_validation, y_validation, ax=ax)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    if save_path:
        fig.savefig(save_path)
    return fig, ax


def plot_moment_matched_predictive_normal_distribution_and_function_samples(
    x_test,
    predictive_distribution,
    distribution_samples,
    x_train=None,
    y_train=None,
    x_validation=None,
    y_validation=None,
    y_test=None,
    fig=None,
    ax=None,
    y_lim=None,
    label="",
    save_path=None,
):
    fig, ax = plot_moment_matched_predictive_normal_distribution(
        x_test,
        predictive_distribution,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        y_test=y_test,
        fig=fig,
        ax=ax,
        y_lim=y_lim,
        label=label,
    )
    plot_function_samples(
        x_test, distribution_samples, fig=fig, ax=ax, save_path=save_path
    )
    return fig, ax


def plot_weight_space_first_vs_last_layer(
    samples_first,
    samples_last,
    x_plot_first=None,
    x_plot_last=None,
    hist=True,
    bins=None,
    kde=False,
    kde_bandwidth=1,
    point_estimate_first=None,
    point_estimate_last=None,
    ensemble_point_estimates_first=None,
    ensemble_point_estimates_last=None,
    distribution_first=None,
    distribution_last=None,
    ensemble_distributions_last=None,
    fig_title="",
    save_path=None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(fig_title)
    ax1.set_title("Input to hidden layer weight")
    plot_weight_space_histogram(
        samples_first,
        x_plot=x_plot_first,
        bins=bins,
        hist=hist,
        kde=kde,
        kde_bandwidth=kde_bandwidth,
        point_estimate=point_estimate_first,
        ensemble_point_estimates=ensemble_point_estimates_first,
        distribution=distribution_first,
        fig=fig,
        ax=ax1,
    )
    ax2.set_title("Hidden to output layer weight")
    plot_weight_space_histogram(
        samples_last,
        x_plot=x_plot_last,
        bins=bins,
        hist=hist,
        kde=kde,
        kde_bandwidth=kde_bandwidth,
        point_estimate=point_estimate_last,
        ensemble_point_estimates=ensemble_point_estimates_last,
        distribution=distribution_last,
        ensemble_distributions=ensemble_distributions_last,
        fig=fig,
        ax=ax2,
    )
    if save_path:
        fig.savefig(save_path)


def plot_weight_space_histogram(
    samples,
    x_plot=None,
    hist=False,
    bins=None,
    kde=True,
    kde_bandwidth=1,
    point_estimate=None,
    ensemble_point_estimates=None,
    distribution=None,
    ensemble_colour_indices=None,
    ensemble_distributions=None,
    fig=None,
    ax=None,
    save_path=None,
):
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth).fit(
        samples[:, None]
    )
    if x_plot is None:
        x_min = np.min(samples)
        x_max = np.max(samples)
        d = x_max - x_min
        x_plot = np.linspace(x_min - 0.1 * d, x_max + 0.1 * d, 1000)
    kde_sample_density = np.exp(kde.score_samples(x_plot[:, None]))
    ax.plot(x_plot, kde_sample_density, alpha=0.5, label="HMC Ground Truth")

    if point_estimate:
        ax.scatter(
            point_estimate,
            np.exp(kde.score_samples(point_estimate.reshape(1, 1))),
            s=100,
            c="k",
            label="MAP Point Estimate",
        )
    if ensemble_point_estimates:
        # if ensemble_colour_indices is None:
        #     ensemble_colour_indices = np.arange(len(ensemble_point_estimates))
        # c = [sns.color_palette()[i] for i in ensemble_colour_indices]
        ensemble_point_estimates = np.array(ensemble_point_estimates)
        ax.scatter(
            ensemble_point_estimates,
            np.exp(kde.score_samples(ensemble_point_estimates.reshape(-1, 1))),
            s=100,
            c="k",
            alpha=0.6,
            label="Ensemble Point Estimates",
        )
    if distribution:
        normalized_pdf = _normalize_distribution(x_plot, distribution, kde)
        ax.plot(
            x_plot, normalized_pdf, c="k", label="Last Layer Weight Distribution",
        )
    if ensemble_distributions:
        normalized_pdfs = []
        for distribution in ensemble_distributions:
            normalized_pdfs.append(_normalize_distribution(x_plot, distribution, kde))
        normalized_pdfs = np.array(normalized_pdfs)
        print(normalized_pdfs.shape)
        ax.plot(
            x_plot,
            np.sum(normalized_pdfs, axis=0),
            c="k",
            alpha=0.0,
            label="Last Layer Weight Distribution",
        )
        for normalized_pdf in normalized_pdfs:
            ax.plot(
                x_plot, normalized_pdf, c="k", alpha=1,
            )
    ax.legend()
    if save_path:
        fig.savefig(save_path)
    return fig, ax
