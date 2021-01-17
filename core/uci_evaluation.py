import inspect
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import scipy as scp
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from sklearn.model_selection import RepeatedKFold, train_test_split

from core.last_layer import PostHocLastLayerBayesianEnsemble as LLBEnsemble
from core.last_layer import PostHocLastLayerBayesianNetwork as LLBNetwork
from core.map import (
    MapDensityEnsemble,
    MapDensityNetwork,
    map_density_ensemble_from_save_path,
)
from core.network_utils import prior_scale_to_regularization_lambda
from core.plotting_utils import (
    plot_uci_ensemble_size_benchmark,
    plot_uci_single_benchmark,
)
from core.variational import VariationalDensityNetwork
from data.uci import load_uci_data

tfd = tfp.distributions

################################################################################
# These helper functions should in principle be written again from scratch.
# They historically grew ever larger, more complicated, and not modular.
# Nevertheless it seems not time efficient right now to rewrite them from scratch.
################################################################################


def load_models(save_dir, load_model_function):
    paths = Path(save_dir).glob("*")
    model_files = sorted(paths)
    print(len(model_files))
    models = []
    if len(model_files) == 0:
        raise RuntimeError(f"No models found at {save_dir}.")
    for file in model_files:
        models.append(load_model_function(file))
    return models


def save_models(save_dir, models):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(models):
        model.save(save_dir.joinpath(f"model_{i:03}"))


def calculate_rmse(mean_prediction, y):
    return tf.keras.metrics.RootMeanSquaredError()(mean_prediction, y)


def evaluate_uci(
    model,
    dataset="boston",
    use_gap_data=False,
    data_seed=0,
    train_seed=0,
    validation_size=0.0,
    model_kwargs={},
    fit_kwargs={},
    verbose=0,
    ensemble_predict_moment_matched=False  # If True the NLL will also be computed for
    # the moment matched predictive distribution and not just the mixture predictive
    # distribution.
):
    x, y, train_indices, validation_indices, test_indices = load_uci_data(
        dataset, validation_size, gap_data=use_gap_data
    )

    input_shape = [x.shape[1]]
    model_kwargs["input_shape"] = input_shape
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=test_size, random_state=data_seed
    # )
    x_train = x[train_indices[data_seed]]
    x_validation = x[validation_indices[data_seed]]
    x_test = x[test_indices[data_seed]]
    y_train = y[train_indices[data_seed]]
    y_validation = y[validation_indices[data_seed]]
    y_test = y[test_indices[data_seed]]
    y_train = y_train.reshape(-1, 1)
    y_validation = y_validation.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    if model == VariationalDensityNetwork:
        model_kwargs["kl_weight"] = 1 / x_train.shape[0]
    if inspect.isclass(model):
        model = model(**model_kwargs, seed=train_seed)
    validation_data = (x_validation, y_validation)
    if validation_size == 0:
        validation_data = None
    model.fit(
        x_train,
        y_train,
        **fit_kwargs,
        validation_data=validation_data,
        verbose=verbose,  # validation_split=validation_size,
    )
    # print("validation loss after training", model.evaluate(x_validation, y_validation))
    predictive_distribution = model.predict(x_validation)
    print(
        "validation loss by hand",
        -tf.reduce_mean(predictive_distribution.log_prob(y_validation)),
    )
    total_epochs = model.total_epochs
    predictive_distribution = model.predict(x_test)
    # independent = tfd.Independent(predictive_distribution, reinterpreted_batch_ndims=2)
    rmse = calculate_rmse(predictive_distribution.mean(), y_test)
    negative_log_likelihood = -tf.reduce_mean(predictive_distribution.log_prob(y_test))
    if ensemble_predict_moment_matched:
        predictive_distribution = model.predict_moment_matched_gaussian(x_test)
        rmse_mm = (
            calculate_rmse(predictive_distribution.mean(), y_test)
            .numpy()
            .astype(np.float)
        )
        assert rmse_mm == rmse
        mm_negative_log_likelihood = (
            -tf.reduce_mean(predictive_distribution.log_prob(y_test))
            .numpy()
            .astype(np.float)
        )
        return (
            rmse,
            negative_log_likelihood,
            mm_negative_log_likelihood,
            model,
            total_epochs,
        )
    # print(tf.reduce_sum(predictive_distribution.log_prob(y_test)) / y_test.shape[0])
    # print(independent.log_prob(y_test) / y_test.shape[0])
    return rmse, negative_log_likelihood, model, total_epochs


def kfold_evaluate_uci(
    model_class,
    dataset="boston",
    use_gap_data=False,
    train_seed=0,
    validation_split=0.0,
    weight_prior_scale=None,
    bias_prior_scale=None,
    model_kwargs={},
    fit_kwargs={},
    fit_kwargs_list=None,
    ensemble_predict_moment_matched=False,  # If True the NLL will also be computed for
    # the moment matched predictive distribution and not just the mixture predictive
    # distribution.
    verbose=False,
):
    x, y, train_indices, validation_indices, test_indices = load_uci_data(
        dataset, validation_split=validation_split, gap_data=use_gap_data
    )
    input_shape = [x.shape[1]]
    model_kwargs["input_shape"] = input_shape
    rmses = []
    negative_log_likelihoods = []
    models = []
    total_epochs = []
    fit_times = []
    if ensemble_predict_moment_matched:
        mm_negative_log_likelihoods = []
    if fit_kwargs_list is None:
        fit_kwargs_list = [fit_kwargs for i in range(len(train_indices))]
    for i, train, validation, test, fit_kwargs in zip(
        range(len(train_indices)),
        train_indices,
        validation_indices,
        test_indices,
        fit_kwargs_list,
    ):
        x_train = x[train]
        x_validation = x[validation]
        x_test = x[test]
        y_train = y[train].reshape(-1, 1)
        y_validation = y[validation].reshape(-1, 1)
        y_test = y[test].reshape(-1, 1)
        validation_data = (x_validation, y_validation)
        if validation.size == 0:
            validation_data = None
        if model_class == VariationalDensityNetwork:
            model_kwargs["kl_weight"] = 1 / x_train.shape[0]
        n_train = x_train.shape[0]
        if "weight_prior" in model_kwargs.keys():
            model_kwargs["n_train"] = n_train
        if weight_prior_scale is not None:
            l2_weight_lambda = prior_scale_to_regularization_lambda(
                weight_prior_scale, n_train
            )
            model_kwargs["l2_weight_lambda"] = l2_weight_lambda
        if bias_prior_scale is not None:
            l2_bias_lambda = prior_scale_to_regularization_lambda(
                bias_prior_scale, n_train
            )
            model_kwargs["l2_bias_lambda"] = l2_bias_lambda
        model = model_class(**model_kwargs, seed=train_seed + i)
        if model_class == VariationalDensityNetwork:
            model_kwargs.pop("kl_weight")
        if "n_train" in model_kwargs.keys():
            model_kwargs.pop("n_train")
        start = time.time()
        model.fit(
            x_train, y_train, **fit_kwargs, validation_data=validation_data, verbose=0
        )
        end = time.time()
        fit_times.append(end - start)
        total_epochs.append(model.total_epochs)

        predictive_distribution = model.predict(x_test)
        rmses.append(
            calculate_rmse(predictive_distribution.mean(), y_test)
            .numpy()
            .astype(np.float)
        )
        negative_log_likelihood = (
            -tf.reduce_mean(predictive_distribution.log_prob(y_test))
            .numpy()
            .astype(np.float)
        )
        negative_log_likelihoods.append(negative_log_likelihood)
        if ensemble_predict_moment_matched:
            predictive_distribution = model.predict_moment_matched_gaussian(x_test)
            rmse_mm = (
                calculate_rmse(predictive_distribution.mean(), y_test)
                .numpy()
                .astype(np.float)
            )
            assert rmse_mm == rmses[-1]
            mm_negative_log_likelihood = (
                -tf.reduce_mean(predictive_distribution.log_prob(y_test))
                .numpy()
                .astype(np.float)
            )
            mm_negative_log_likelihoods.append(mm_negative_log_likelihood)
        models.append(model)
        if verbose:
            print(f"Done Split {i}")
    return_tuple = (
        (
            rmses,
            negative_log_likelihoods,
            mm_negative_log_likelihoods,
            models,
            total_epochs,
            fit_times,
        )
        if ensemble_predict_moment_matched
        else (rmses, negative_log_likelihoods, models, total_epochs, fit_times)
    )
    return return_tuple


def benchmark_models_uci(
    dataset="boston",
    use_gap_data=False,
    train_seed=0,
    layer_units=[50, 1],
    layer_activations=["relu", "linear"],
    initial_unconstrained_scale=-1,
    transform_unconstrained_scale_factor=0.5,
    learning_rate=0.01,
    epochs=40,
    batch_size=100,
    ensemble_n_networks=5,
    early_stop_callback=None,
    validation_split=0.0,
    weight_prior_scale=None,
    bias_prior_scale=None,
    last_layer_prior="non-informative",
    last_layer_prior_params=None,
    vi_flat_prior=True,
    evaluate_ignore_prior_loss=True,
):
    # print(
    #     "Even when passing regularization parameteres, no prior is yet passed to llb nets."
    # )
    results = {}
    fit_kwargs = {
        "epochs": epochs,
        "batch_size": batch_size,
        "early_stop_callback": early_stop_callback,
    }
    layer_names = [None] * (len(layer_units) - 2) + ["feature_extractor", "output"]
    model_kwargs = {
        "layer_units": layer_units,
        "layer_activations": layer_activations,
        "initial_unconstrained_scale": initial_unconstrained_scale,
        "transform_unconstrained_scale_factor": transform_unconstrained_scale_factor,
        "prior_scale_identity_multiplier": 1,
        "preprocess_x": True,
        "preprocess_y": True,
        "learning_rate": learning_rate,
        "names": layer_names,
        "evaluate_ignore_prior_loss": evaluate_ignore_prior_loss,
    }

    (
        vi_prior_rmses,
        vi_prior_negative_log_likelihoods,
        vi_prior_models,
        vi_prior_total_epochs,
        vi_prior_fit_times,
    ) = kfold_evaluate_uci(
        dataset=dataset,
        use_gap_data=use_gap_data,
        model_class=VariationalDensityNetwork,
        train_seed=train_seed,
        validation_split=validation_split,
        model_kwargs=model_kwargs,
        fit_kwargs=fit_kwargs,
    )
    results["VI-Prior"] = {
        "RMSEs": vi_prior_rmses,
        "NLLs": vi_prior_negative_log_likelihoods,
        "total_epochs": vi_prior_total_epochs,
        "fit_times": vi_prior_fit_times,
    }
    print("VI-Prior Done")
    if vi_flat_prior:
        model_kwargs["prior_scale_identity_multiplier"] = 1e6
        (
            vi_flat_prior_rmses,
            vi_flat_prior_negative_log_likelihoods,
            vi_flat_prior_models,
            vi_flat_prior_total_epochs,
            vi_flat_prior_fit_times,
        ) = kfold_evaluate_uci(
            dataset=dataset,
            use_gap_data=use_gap_data,
            model_class=VariationalDensityNetwork,
            train_seed=train_seed,
            validation_split=validation_split,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
        )
        results["VI-Flat-Prior"] = {
            "RMSEs": vi_flat_prior_rmses,
            "NLLs": vi_flat_prior_negative_log_likelihoods,
            "total_epochs": vi_flat_prior_total_epochs,
            "fit_times": vi_flat_prior_fit_times,
        }
        print("VI-Flat-Prior Done")
    model_kwargs.pop("prior_scale_identity_multiplier")
    model_kwargs.pop("evaluate_ignore_prior_loss")
    (
        map_rmses,
        map_negative_log_likelihoods,
        map_models,
        map_total_epochs,
        map_fit_times,
    ) = kfold_evaluate_uci(
        dataset=dataset,
        use_gap_data=use_gap_data,
        model_class=MapDensityNetwork,
        train_seed=train_seed,
        validation_split=validation_split,
        weight_prior_scale=weight_prior_scale,
        bias_prior_scale=bias_prior_scale,
        model_kwargs=model_kwargs,
        fit_kwargs=fit_kwargs,
    )
    results["Map"] = {
        "RMSEs": map_rmses,
        "NLLs": map_negative_log_likelihoods,
        "total_epochs": map_total_epochs,
        "fit_times": map_fit_times,
    }
    print("Map Done")
    model_kwargs.pop("names")
    model_kwargs["last_layer_prior"] = last_layer_prior
    model_kwargs["last_layer_prior_params"] = last_layer_prior_params
    llb_fit_kwargs_list = [{"pretrained_network": model} for model in map_models]
    (
        llb_rmses,
        llb_negative_log_likelihoods,
        llb_models,
        llb_total_epochs,
        llb_fit_times,
    ) = kfold_evaluate_uci(
        dataset=dataset,
        use_gap_data=use_gap_data,
        model_class=LLBNetwork,
        train_seed=train_seed,
        validation_split=validation_split,
        model_kwargs=model_kwargs,
        fit_kwargs_list=llb_fit_kwargs_list,
    )
    results["Last Layer Bayesian"] = {
        "RMSEs": llb_rmses,
        "NLLs": llb_negative_log_likelihoods,
        "fit_times": llb_fit_times,
    }
    print("LLB Done")
    model_kwargs.pop("last_layer_prior")
    model_kwargs.pop("last_layer_prior_params")
    model_kwargs["names"] = layer_names
    model_kwargs["n_networks"] = ensemble_n_networks
    (
        ensemble_rmses,
        ensemble_negative_log_likelihoods,
        ensemble_mm_negative_log_likelihoods,
        ensemble_models,
        ensemble_total_epochs,
        ensemble_fit_times,
    ) = kfold_evaluate_uci(
        dataset=dataset,
        use_gap_data=use_gap_data,
        model_class=MapDensityEnsemble,
        train_seed=train_seed,
        validation_split=validation_split,
        weight_prior_scale=weight_prior_scale,
        bias_prior_scale=bias_prior_scale,
        model_kwargs=model_kwargs,
        fit_kwargs=fit_kwargs,
        ensemble_predict_moment_matched=True,
    )
    results["Ensemble"] = {
        "RMSEs": ensemble_rmses,
        "NLLs": ensemble_negative_log_likelihoods,
        "total_epochs": ensemble_total_epochs,
        "MM-NLLs": ensemble_mm_negative_log_likelihoods,
        "fit_times": ensemble_fit_times,
    }
    print("Ensemble Done")
    model_kwargs.pop("names")
    model_kwargs["last_layer_prior"] = last_layer_prior
    model_kwargs["last_layer_prior_params"] = last_layer_prior_params
    llb_ensemble_fit_kwargs_list = [
        {"pretrained_networks": model.networks} for model in ensemble_models
    ]
    (
        llb_ensemble_rmses,
        llb_ensemble_negative_log_likelihoods,
        llb_ensemble_mm_negative_log_likelihoods,
        llb_ensemble_models,
        llb_ensemble_total_epochs,
        llb_ensemble_fit_times,
    ) = kfold_evaluate_uci(
        dataset=dataset,
        use_gap_data=use_gap_data,
        model_class=LLBEnsemble,
        train_seed=train_seed,
        validation_split=validation_split,
        model_kwargs=model_kwargs,
        fit_kwargs_list=llb_ensemble_fit_kwargs_list,
        ensemble_predict_moment_matched=True,
    )
    results["LLB Ensemble"] = {
        "RMSEs": llb_ensemble_rmses,
        "NLLs": llb_ensemble_negative_log_likelihoods,
        "MM-NLLs": llb_ensemble_mm_negative_log_likelihoods,
        "fit_times": llb_ensemble_fit_times,
    }
    return results


def save_results(
    experiment_name, dataset, results, use_gap_data=False, sub_folder=None
):
    """
    This function receives the experimental results. If already a json file exists to
    this experiment_name and dataset, it will read from that json file and compare the
    results for keys that exist already. When the results are not equivalent it will
    throw an AssertionError. Then it will save results of new keys into the same json
    file. If the file doesn't yet exist it will be created.
    """
    data_path = "uci_gap_data" if use_gap_data else "uci_data"
    dir = Path(f"{data_path}/{dataset}/results/")
    if sub_folder is not None:
        dir = dir.joinpath(sub_folder)
    dir.mkdir(parents=True, exist_ok=True)
    experiment_path = dir / f"{experiment_name}.json"
    if experiment_path.is_file():
        old_results = json.loads(experiment_path.read_text())
        for method_key in old_results.keys():
            if method_key in results.keys():
                for result_key in old_results[method_key].keys():
                    if result_key != "fit_times":
                        assert (
                            old_results[method_key][result_key]
                            == results[method_key][result_key]
                        ), f"New result {result_key} of method {method_key} are not equivalent to existing results. \n\nNew Results: {results[method_key]} \n\nOld Results: {old_results[method_key]}"
        for method_key in results.keys():
            old_results[method_key] = results[method_key]
        results = old_results
    with open(experiment_path, "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


def uci_benchmark_save_plot(
    experiment_name,
    dataset,
    figure_dir,
    use_gap_data=False,
    train_seed=0,
    layer_units=[50, 1],
    layer_activations=["relu", "linear"],
    initial_unconstrained_scale=-1,
    transform_unconstrained_scale_factor=0.5,
    learning_rate=0.01,
    epochs=40,
    batch_size=100,
    ensemble_n_networks=5,
    early_stop_callback=None,
    validation_split=0.0,
    weight_prior_scale=None,
    bias_prior_scale=None,
    last_layer_prior="non-informative",
    last_layer_prior_params=None,
    vi_flat_prior=True,
    evaluate_ignore_prior_loss=True,
    save=True,
):
    results = benchmark_models_uci(
        dataset=dataset,
        use_gap_data=use_gap_data,
        train_seed=train_seed,
        ensemble_n_networks=ensemble_n_networks,
        layer_units=layer_units,
        layer_activations=layer_activations,
        initial_unconstrained_scale=initial_unconstrained_scale,
        transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        early_stop_callback=early_stop_callback,
        validation_split=validation_split,
        weight_prior_scale=weight_prior_scale,
        bias_prior_scale=bias_prior_scale,
        last_layer_prior=last_layer_prior,
        last_layer_prior_params=last_layer_prior_params,
        evaluate_ignore_prior_loss=evaluate_ignore_prior_loss,
    )

    if save:
        save_results(experiment_name, dataset, results, use_gap_data=use_gap_data)

    labels = (
        [
            "VI-Prior",
            "VI-Flat-Prior",
            "Map",
            "Last Layer Bayesian",
            "Ensemble",
            "LLB Ensemble",
        ]
        if vi_flat_prior
        else ["VI-Prior", "Map", "Last Layer Bayesian", "Ensemble", "LLB Ensemble"]
    )
    title = f"{dataset.capitalize()}; Average Test RMSE"
    y_label = "RMSE"
    plot_uci_single_benchmark(
        model_results=[results[label]["RMSEs"] for label in labels],
        labels=labels,
        title=title,
        y_label=y_label,
    )

    title = f"{dataset.capitalize()}; Average Negative Test Log Likelihood"
    y_label = "Negative Log Likelihood"
    plot_uci_single_benchmark(
        model_results=[results[label]["NLLs"] for label in labels],
        labels=labels,
        title=title,
        y_label=y_label,
    )

    total_epochs = []
    total_epoch_labels = []
    for label in labels:
        if "total_epochs" in results[label].keys():
            total_epochs.append(results[label]["total_epochs"])
            total_epoch_labels.append(label)
    title = f"{dataset.capitalize()}; Average Epochs"
    y_label = "Epochs"
    plot_uci_single_benchmark(
        model_results=total_epochs,
        labels=total_epoch_labels,
        title=title,
        y_label=y_label,
    )


def generate_member_indices(n_networks, size, n_comb_max=100, seed=0):
    random.seed(seed)
    if scp.special.comb(n_networks, size, exact=True) < n_comb_max * 10:
        combs = list(itertools.combinations(np.arange(n_networks), r=size))
        random.shuffle(combs)
        combs = combs[0:n_comb_max]
    else:
        combs = set()
        while len(combs) < n_comb_max:
            combs.add(tuple(sorted(random.sample(range(n_networks), size))))
        combs = list(combs)
    return combs


def evaluate_ensemble_size(model, x_test, y_test, size=1, seed=0):
    nets = model.networks
    model.networks = nets[:size]
    model.predict(x_test)
    predictive_distribution = model.predict(x_test)
    rmse = (
        calculate_rmse(predictive_distribution.mean(), y_test).numpy().astype(np.float)
    )
    negative_log_likelihood = (
        -tf.reduce_mean(predictive_distribution.log_prob(y_test))
        .numpy()
        .astype(np.float)
    )
    mm_predictive_distribution = model.predict_moment_matched_gaussian(x_test)
    # rmse_mm = calculate_rmse(mm_predictive_distribution.mean(), y_test)
    # assert rmse_mm == rmse
    mm_negative_log_likelihood = (
        -tf.reduce_mean(mm_predictive_distribution.log_prob(y_test))
        .numpy()
        .astype(np.float)
    )
    # undo side effects
    model.networks = nets
    return rmse, negative_log_likelihood, mm_negative_log_likelihood


def uci_benchmark_ensemble_sizes_save_plot(
    experiment_name,
    model_save_dir=None,
    dataset="boston",
    use_gap_data=False,
    train_seed=0,
    layer_units=[50, 1],
    layer_activations=["relu", "linear"],
    initial_unconstrained_scale=-1,
    transform_unconstrained_scale_factor=0.5,
    learning_rate=0.01,
    epochs=40,
    batch_size=100,
    n_networks=5,
    early_stop_callback=None,
    weight_prior_scale=None,
    bias_prior_scale=None,
    weight_prior=None,
    bias_prior=None,
    noise_scale_prior=None,
    last_layer_prior="non-informative",
    last_layer_prior_params=None,
    validation_split=0.0,
    save=True,  # wether to save the results dict
    verbose=False,
):
    tf.keras.backend.clear_session()
    results = {}
    fit_kwargs = {
        "epochs": epochs,
        "batch_size": batch_size,
        "early_stop_callback": early_stop_callback,
    }
    layer_names = [None] * (len(layer_units) - 2) + ["feature_extractor", "output"]
    model_kwargs = {
        "layer_units": layer_units,
        "layer_activations": layer_activations,
        "initial_unconstrained_scale": initial_unconstrained_scale,
        "transform_unconstrained_scale_factor": transform_unconstrained_scale_factor,
        "weight_prior": weight_prior,
        "bias_prior": bias_prior,
        "noise_scale_prior": noise_scale_prior,
        "preprocess_x": True,
        "preprocess_y": True,
        "learning_rate": learning_rate,
        "names": layer_names,
        "n_networks": n_networks,
    }
    if model_save_dir is not None:
        model_save_dir = Path(model_save_dir)
        model_save_dir.mkdir(parents=True, exist_ok=True)
        ensemble_save_path = model_save_dir.joinpath(
            f"uci_benchmark_ensemble_sizes_{experiment_name}_{dataset}_gap-{use_gap_data}"
        )
        if ensemble_save_path.is_dir():
            print("Loading ensemble from disk")
            ensemble_models = load_models(
                ensemble_save_path,
                load_model_function=map_density_ensemble_from_save_path,
            )
        else:
            (_, _, ensemble_models, _, _) = kfold_evaluate_uci(
                dataset=dataset,
                use_gap_data=use_gap_data,
                model_class=MapDensityEnsemble,
                train_seed=train_seed,
                validation_split=validation_split,
                weight_prior_scale=weight_prior_scale,
                bias_prior_scale=bias_prior_scale,
                model_kwargs=model_kwargs,
                fit_kwargs=fit_kwargs,
                verbose=verbose,
            )
            save_models(ensemble_save_path, ensemble_models)
    else:
        (_, _, ensemble_models, total_epochs, _) = kfold_evaluate_uci(
            dataset=dataset,
            use_gap_data=use_gap_data,
            model_class=MapDensityEnsemble,
            train_seed=train_seed,
            validation_split=validation_split,
            weight_prior_scale=weight_prior_scale,
            bias_prior_scale=bias_prior_scale,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
            verbose=verbose,
        )
        print(total_epochs)

    print("Done Ensemble Training.")
    x, y, train_indices, validation_indices, test_indices = load_uci_data(
        dataset, validation_split=validation_split, gap_data=use_gap_data
    )
    results = {}
    rmses, nlls, mm_nlls = [], [], []
    _net_sizes = np.arange(n_networks) + 1
    _net_sizes = _net_sizes[
        np.logical_or(_net_sizes <= 20, _net_sizes % 5 == 0)
    ]  # Only test for network sizes below 20 or divisible by 5
    for size in _net_sizes:
        size_rmses, size_nlls, size_mm_nlls = [], [], []
        for split, model in enumerate(ensemble_models):
            x_test = x[test_indices[split]]
            y_test = y[test_indices[split]].reshape(-1, 1)
            rmse, nll, mm_nll = evaluate_ensemble_size(
                model, x_test, y_test, size=size, seed=0
            )
            size_rmses.append(rmse)
            size_nlls.append(nll)
            size_mm_nlls.append(mm_nll)
        rmses.append(size_rmses)
        nlls.append(size_nlls)
        mm_nlls.append(size_mm_nlls)
    results["Ensemble"] = {"RMSEs": rmses, "NLLs": nlls, "MM-NLLs": mm_nlls}
    print("Done Ensemble Testing.")

    model_kwargs.pop("names")
    model_kwargs.pop("weight_prior")
    model_kwargs.pop("bias_prior")
    model_kwargs.pop("noise_scale_prior")
    model_kwargs["last_layer_prior"] = last_layer_prior
    model_kwargs["last_layer_prior_params"] = last_layer_prior_params
    llb_ensemble_fit_kwargs_list = [
        {"pretrained_networks": model.networks} for model in ensemble_models
    ]
    _, _, llb_ensemble_models, _, _ = kfold_evaluate_uci(
        dataset=dataset,
        use_gap_data=use_gap_data,
        model_class=LLBEnsemble,
        train_seed=train_seed,
        validation_split=validation_split,
        model_kwargs=model_kwargs,
        fit_kwargs_list=llb_ensemble_fit_kwargs_list,
    )
    print("Done LLB Ensemble Training.")

    rmses, nlls, mm_nlls = [], [], []
    for size in _net_sizes:
        size_rmses, size_nlls, size_mm_nlls = [], [], []
        for split, model in enumerate(llb_ensemble_models):
            x_test = x[test_indices[split]]
            y_test = y[test_indices[split]].reshape(-1, 1)
            rmse, nll, mm_nll = evaluate_ensemble_size(
                model, x_test, y_test, size=size, seed=0
            )
            size_rmses.append(rmse)
            size_nlls.append(nll)
            size_mm_nlls.append(mm_nll)
        rmses.append(size_rmses)
        nlls.append(size_nlls)
        mm_nlls.append(size_mm_nlls)
    results["LLB Ensemble"] = {"RMSEs": rmses, "NLLs": nlls, "MM-NLLs": mm_nlls}

    if save:
        save_results(
            experiment_name,
            dataset,
            results,
            use_gap_data=use_gap_data,
            sub_folder="ensemble_sizes",
        )

    # e_rmses = results["Ensemble"]["RMSEs"]
    e_nlls = results["Ensemble"]["NLLs"]
    e_mm_nlls = results["Ensemble"]["MM-NLLs"]
    # llbe_rmses = results["LLB Ensemble"]["RMSEs"]
    llbe_nlls = results["LLB Ensemble"]["NLLs"]
    llbe_mm_nlls = results["LLB Ensemble"]["MM-NLLs"]
    labels = ["Ensemble MM", "Ensemble", "LLB Ensemble MM", "LLB Ensemble"]
    colors = sns.color_palette()
    with open("config/uci-color-config.yaml") as f:
        color_mapping = yaml.full_load(f)
    plot_uci_ensemble_size_benchmark(
        [e_mm_nlls, e_nlls, llbe_mm_nlls, llbe_nlls],
        labels=labels,
        title=dataset,
        x_label="# ensemble_memebers",
        y_label="Negative Log Likelihood",
        colors=[colors[color_mapping[method]] for method in labels],
    )
    return results
