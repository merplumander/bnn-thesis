import awkward1 as ak
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.neighbors import KernelDensity


def _normalize_distribution(x_plot, distribution, kde):
    """
    used to "normalize" distribution to the height of the kde at that point
    """
    pdf = distribution.prob(x_plot)
    highest_pdf = np.max(pdf)
    mean = distribution.mean().numpy().reshape(1, 1)
    kde_height = np.exp(kde.score_samples(mean))
    normalized_pdf = pdf * kde_height / highest_pdf
    return normalized_pdf


def _distribution_to_highest_point(
    x_plot, distribution, highest_kde, factor_height_point_estimate
):
    """
    used to fit distribution to highest point of kde
    """
    pdf = distribution.prob(x_plot)
    highest_pdf = np.max(pdf)
    normalized_pdf = pdf * highest_kde / highest_pdf * factor_height_point_estimate
    return normalized_pdf


def plot_training_data(
    x, y, fig=None, ax=None, figsize=None, x_lim=None, y_lim=None, alpha=1
):
    if figsize is None:
        figsize = (8, 8)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, c="k", alpha=alpha, marker="x", s=100, label="Train data")
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    return fig, ax


def plot_validation_data(x, y, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, c="g", marker="x", s=100, label="Validation data")
    return fig, ax


def plot_ground_truth(x, y, c="k", alpha=0.1, label="Ground truth", fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y, c=c, alpha=alpha, label=label)
    return fig, ax


def plot_moment_matched_predictive_normal_distribution(
    x_plot,
    predictive_distribution,
    x_train=None,
    y_train=None,
    x_validation=None,
    y_validation=None,
    y_ground_truth=None,
    fig=None,
    ax=None,
    show_hdr=True,
    y_lim=None,
    label="",
    title="",
    no_ticks=True,
    save_path=None,
):
    """
    Plots the moment-matched Normal distribution to the passed predictive distribution.
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(title, fontsize=15)
    if y_ground_truth is not None:
        plot_ground_truth(x_plot, y_ground_truth, ax=ax)
    mean = predictive_distribution.mean().numpy()
    std = predictive_distribution.stddev().numpy()
    label = label if label else "Mean prediction"
    ax.plot(x_plot, mean, label=label, alpha=0.8)
    if show_hdr:
        ax.fill_between(
            x_plot.flatten(),
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
    ax.set_ylim(y_lim)
    ax.legend()
    if no_ticks:
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_distribution_samples(
    x_plot,
    distribution_samples,
    x_train=None,
    y_train=None,
    x_validation=None,
    y_validation=None,
    y_ground_truth=None,
    fig=None,
    ax=None,
    show_hdr=True,
    x_lim=None,
    y_lim=None,
    seaborn_pallet_starting_index=0,
    labels=[],
    no_ticks=True,
    save_path=None,
):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if y_ground_truth is not None:
        plot_ground_truth(x_plot, y_ground_truth, ax=ax)
    for i, prediction in enumerate(distribution_samples):
        mean = prediction.mean()
        std = prediction.stddev()
        c = sns.color_palette()[i + seaborn_pallet_starting_index]
        label = labels[i] if labels else "Mean prediction"
        ax.plot(x_plot, mean, label=label, c=c, alpha=0.8)
        if show_hdr:
            ax.fill_between(
                x_plot.flatten(),
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
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.legend()
    if no_ticks:
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_function_samples(
    x_plot,
    distribution_samples,
    c=None,
    x_train=None,
    y_train=None,
    x_validation=None,
    y_validation=None,
    y_ground_truth=None,
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
    if y_ground_truth is not None:
        plot_ground_truth(x_plot, y_ground_truth, ax=ax)
    if c is None:
        c = sns.color_palette()[1]
    for i, prediction in enumerate(distribution_samples):
        mean = prediction.mean()
        (line,) = ax.plot(x_plot, mean, c=c, alpha=0.2)
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
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_moment_matched_predictive_normal_distribution_and_function_samples(
    x_plot,
    predictive_distribution,
    distribution_samples,
    x_train=None,
    y_train=None,
    x_validation=None,
    y_validation=None,
    y_ground_truth=None,
    fig=None,
    ax=None,
    show_hdr=True,
    y_lim=None,
    label="",
    title="",
    no_ticks=True,
    save_path=None,
):
    fig, ax = plot_moment_matched_predictive_normal_distribution(
        x_plot,
        predictive_distribution,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        y_ground_truth=y_ground_truth,
        fig=fig,
        ax=ax,
        show_hdr=show_hdr,
        y_lim=y_lim,
        label=label,
        title=title,
        no_ticks=no_ticks,
    )
    plot_function_samples(
        x_plot, distribution_samples, fig=fig, ax=ax, save_path=save_path
    )
    return fig, ax


def plot_predictive_distribution_by_samples(
    x_plot,
    predictive_distribution,
    x_train=None,
    y_train=None,
    x_validation=None,
    y_validation=None,
    y_ground_truth=None,
    samples_per_location=500,
    alpha=0.0025,
    markersize=1.5,
    c=None,
    fig=None,
    ax=None,
    y_lim=None,
    label="Predictive distribution",
    save_path=None,
):
    if alpha < 1 / 255:
        print("Matplotlib cuts off alpha values below 1/255")
    predictive_samples = predictive_distribution.sample(samples_per_location)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if y_ground_truth is not None:
        plot_ground_truth(x_plot, y_ground_truth, ax=ax)
    if c is None:
        c = sns.color_palette()[0]
    lines = ax.plot(
        x_plot.flatten(),
        predictive_samples.numpy().T.reshape(-1, samples_per_location),
        "o",
        c=c,
        markersize=markersize,
        alpha=alpha,
    )
    # lines = ax.hexbin(
    #     np.array(x_plot.flatten().tolist() * samples_per_location),
    #     predictive_samples.numpy().reshape(-1),
    #     gridsize=200,
    #     cmap=plt.get_cmap("Blues"),
    # )
    lines[-1].set_label(label)

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
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_predictive_density(
    x_plot,
    predictive_distribution,
    y_lim=[-10, 10],
    n_y=500,
    levels=100,
    x_train=None,
    y_train=None,
    x_validation=None,
    y_validation=None,
    y_ground_truth=None,
    cmap="Blues",
    fig=None,
    ax=None,
    label="Predictive density",
    save_path=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    if y_ground_truth is not None:
        plot_ground_truth(x_plot, y_ground_truth, ax=ax)
    y = np.linspace(y_lim[0], y_lim[1], n_y)
    xx, yy = np.meshgrid(x_plot, y)

    zz = predictive_distribution.prob(y).numpy().T
    ax.contourf(xx, yy, zz, cmap=plt.get_cmap(cmap), levels=levels, label=label)
    if x_train is not None:
        plot_training_data(x_train, y_train, ax=ax)
    if x_validation is not None:
        plot_validation_data(x_validation, y_validation, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_weight_space_first_vs_last_layer(
    samples_first,
    samples_last,
    x_plot_first=None,
    x_plot_last=None,
    hist=True,
    bins=None,
    kde=False,
    kde_bandwidth_factor=10,
    point_estimate_first=None,
    point_estimate_last=None,
    ensemble_point_estimates_first=None,
    ensemble_point_estimates_last=None,
    distribution_first=None,
    distribution_last=None,
    ensemble_distributions_last=None,
    fig_title="",
    y_lim1=None,
    y_lim2=None,
    save_path=None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(fig_title, fontsize=15)
    ax1.set_title("Input to hidden layer weight")
    plot_weight_space_histogram(
        samples_first,
        x_plot=x_plot_first,
        bins=bins,
        hist=hist,
        kde=kde,
        kde_bandwidth_factor=kde_bandwidth_factor,
        point_estimate=point_estimate_first,
        ensemble_point_estimates=ensemble_point_estimates_first,
        distribution=distribution_first,
        fig=fig,
        ax=ax1,
    )
    ax1.set_ylim(y_lim1)
    ax2.set_title("Hidden to output layer weight")
    plot_weight_space_histogram(
        samples_last,
        x_plot=x_plot_last,
        bins=bins,
        hist=hist,
        kde=kde,
        kde_bandwidth_factor=kde_bandwidth_factor,
        point_estimate=point_estimate_last,
        ensemble_point_estimates=ensemble_point_estimates_last,
        distribution=distribution_last,
        ensemble_distributions=ensemble_distributions_last,
        fig=fig,
        ax=ax2,
    )
    ax2.set_ylim(y_lim2)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


def plot_weight_space_histogram(
    samples,
    x_plot=None,
    hist=False,
    bins=None,
    kde=True,
    kde_bandwidth_factor=0.1,
    point_estimate=None,
    ensemble_point_estimates=None,
    distribution=None,
    ensemble_colour_indices=None,
    ensemble_distributions=None,
    fig=None,
    ax=None,
    save_path=None,
):
    # point_estimate_width = 6
    # factor_height_point_estimate = 1.0
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    # The KDE bandwidth should be wider, the wider the range is.
    range = np.max(samples) - np.min(samples)
    kde_bandwidth = kde_bandwidth_factor * range
    kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth).fit(
        samples[:, None]
    )
    if x_plot is None:
        x_min = np.min(samples)
        x_max = np.max(samples)
        d = x_max - x_min
        x_plot = np.linspace(x_min - 0.1 * d, x_max + 0.1 * d, 1000)
    kde_sample_density = np.exp(kde.score_samples(x_plot[:, None]))
    c = sns.color_palette()[0]
    ax.plot(x_plot, kde_sample_density, c=c, alpha=1, label="HMC")
    lines, labels = ax.get_legend_handles_labels()
    c = sns.color_palette()[2]
    if point_estimate:
        # # Plot lines
        # ax.vlines(
        #     point_estimate,
        #     0,
        #     np.max(kde_sample_density) * factor_height_point_estimate,
        #     color="k",
        #     linewidths=point_estimate_width,
        #     label="MAP Point Estimate",
        # )
        # Plot points
        line = ax.scatter(
            point_estimate,
            np.exp(kde.score_samples(point_estimate.reshape(1, 1))),
            s=100,
            c=c,
            label="MAP",
        )
        lines.append(line)
        labels.append("MAP")
    if ensemble_point_estimates:
        # if ensemble_colour_indices is None:
        #     ensemble_colour_indices = np.arange(len(ensemble_point_estimates))
        # c = [sns.color_palette()[i] for i in ensemble_colour_indices]
        ensemble_point_estimates = np.array(ensemble_point_estimates)
        # # Plot Lines
        # ax.vlines(
        #     ensemble_point_estimates,
        #     0,
        #     np.max(kde_sample_density) * factor_height_point_estimate / len(ensemble_point_estimates),
        #     color="k",
        #     linewidths=point_estimate_width,
        #     label="Ensemble Point Estimates",
        # )
        # Plot points
        line = ax.scatter(
            ensemble_point_estimates,
            np.exp(kde.score_samples(ensemble_point_estimates.reshape(-1, 1))),
            s=100,
            c=c,
            alpha=1,
            label="MAP Ensemble",
        )
        lines.append(line)
        labels.append("MAP Ensemble")
    if distribution:
        # # 'normalized' to kde height
        # normalized_pdf = _normalize_distribution(x_plot, distribution, kde)

        # # 'normalized' to highest point
        # normalized_pdf = _distribution_to_highest_point(
        #     x_plot,
        #     distribution,
        #     np.max(kde_sample_density),
        #     factor_height_point_estimate,
        # )
        # # no normalization
        # ax.plot(x_plot, distribution.prob(x_plot), c="k", label="Last Layer Weight Distribution")
        pdf = distribution.prob(x_plot)
        twin_ax = ax.twinx()
        twin_ax.plot(x_plot, pdf, c=c, label="Last-Layer Distribution")
        _lines, _labels = twin_ax.get_legend_handles_labels()
        lines += _lines
        labels += _labels
        twin_ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    if ensemble_distributions:
        normalized_pdfs = []
        means = []
        for distribution in ensemble_distributions:
            # # 'normalized' to kde at that point
            # normalized_pdfs.append(_normalize_distribution(x_plot, distribution, kde))

            # # 'normalized' to highest point
            # normalized_pdfs.append(
            #     _distribution_to_highest_point(
            #         x_plot, distribution, np.max(kde_sample_density), factor_height_point_estimate
            #     ) / len(ensemble_distributions)
            # )

            # no normalization
            normalized_pdfs.append(
                distribution.prob(x_plot) / len(ensemble_distributions)
            )
            means.append(distribution.mean().numpy())
        normalized_pdfs = np.array(normalized_pdfs)
        means = np.array(means)
        # Plot mixture
        mixture = np.sum(normalized_pdfs, axis=0)
        # highest_mixture = np.max(mixture)
        # highest_kde = np.max(kde_sample_density)
        # mixture = mixture * highest_kde / highest_mixture
        twin_ax = ax.twinx()
        twin_ax.plot(x_plot, mixture, c=c, alpha=1, label="Last-Layer Distribution")
        _lines, _labels = twin_ax.get_legend_handles_labels()
        lines += _lines
        labels += _labels
        twin_ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        # # Plot individual distributions
        # for normalized_pdf in normalized_pdfs:
        #     ax.plot(x_plot, normalized_pdf, c="k", alpha=0)

        # # Plot means
        # ax.scatter(
        #     means,
        #     np.zeros_like(means),
        #     s=20,
        #     c=c,
        #     alpha=1,
        #     # label="Ensemble Point Estimates",
        # )
    # ax.tick_params(axis="both", which="major", labelsize=12)
    # start, end = ax.get_xlim()
    # ax.xaxis.set_ticks(np.arange(start, end, 5))
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

    ax.legend(lines, labels)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_uci_single_benchmark(
    model_results,
    labels=None,
    title=None,
    ax_title=None,
    y_label=None,
    legend=True,
    legend_kwargs={},
    colors=None,
    fig=None,
    ax=None,
    save_path=None,
):
    if colors is None:
        colors = [None] * len(model_results)
    means = [np.mean(result) for result in model_results]
    std_errors = np.array([np.std(result) for result in model_results]) / np.sqrt(
        len(model_results[0])
    )
    if fig is None:
        fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=15)
    ax.set_title(ax_title)
    for i, mean, std_error, label, color in zip(
        range(len(means)), means, std_errors, labels, colors
    ):
        ax.errorbar(i, mean, std_error, fmt="o", label=label, c=color)
    ax.set_ylabel(y_label)
    ax.tick_params(bottom=False, labelbottom=False)
    if legend:
        ax.legend(**legend_kwargs)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return ax.get_legend_handles_labels()


def plot_uci_ensemble_size_benchmark(
    model_results,
    x=None,
    x_add_jitter=0.0,
    normalization=False,
    errorbars=True,
    pareto_point=False,
    labels=None,
    title=None,
    x_label=None,
    y_label=None,
    x_lim=None,
    colors=None,
    alpha=1.0,
    fig=None,
    ax=None,
    save_path=None,
):
    if colors is None:
        colors = [None] * len(model_results)

    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    fig.suptitle(title, fontsize=15)

    for i, results, label, color in zip(
        range(len(model_results)), model_results, labels, colors
    ):
        results = np.array(results)
        size_means = np.mean(results, axis=1)
        size_stds = np.std(results, axis=1)
        if normalization:
            min = np.min(size_means)
            max = np.max(size_means)
            size_means = (size_means - min) / (max - min)
            size_stds = size_stds / (max - min)
        if x is None:
            x = np.arange(len(results)) + 1
        x = x + 0.15 * i
        #     results = ak.Array(results)
        #     size_split_means = np.mean(results, axis=2)
        #     size_means = np.array(np.mean(size_split_means, axis=1))
        #     size_split_stds = np.std(results, axis=2)
        #     size_stds = np.array(np.mean(size_split_stds, axis=1))
        #     print(size_means)
        if errorbars:
            ax.errorbar(
                x + x_add_jitter,
                size_means,
                size_stds / np.sqrt(results.shape[1]),
                c=color,
                fmt="o",
                alpha=alpha,
            )
        ax.plot(x + x_add_jitter, size_means, c=color, alpha=alpha, label=label)
    if pareto_point:
        ax.scatter(10, 0.2, c="k", label="80/20 point")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_lim)
    ax.legend()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
