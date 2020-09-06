import inspect
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import RepeatedKFold, train_test_split

from core.last_layer import PostHocLastLayerBayesianEnsemble as LLBEnsemble
from core.last_layer import PostHocLastLayerBayesianNetwork as LLBNetwork
from core.map import MapDensityEnsemble, MapDensityNetwork
from core.plotting_utils import plot_uci_single_benchmark
from core.variational import VariationalDensityNetwork
from data.uci import (  # load_boston, load_concrete, load_energy, load_kin8nm
    load_uci_data,
)

tfd = tfp.distributions


def calculate_rmse(mean_prediction, y):
    return tf.keras.metrics.RootMeanSquaredError()(mean_prediction, y)


def evaluate_uci(
    model,
    dataset="boston",
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
        dataset, validation_size
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
    train_seed=0,
    validation_split=0.0,
    model_kwargs={},
    fit_kwargs={},
    fit_kwargs_list=None,
    ensemble_predict_moment_matched=False  # If True the NLL will also be computed for
    # the moment matched predictive distribution and not just the mixture predictive
    # distribution.
):
    x, y, train_indices, validation_indices, test_indices = load_uci_data(
        dataset, validation_split=validation_split
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
        model = model_class(**model_kwargs, seed=train_seed + i)
        if model_class == VariationalDensityNetwork:
            model_kwargs.pop("kl_weight")
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
):
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
    }

    (
        vi_prior_rmses,
        vi_prior_negative_log_likelihoods,
        vi_prior_models,
        vi_prior_total_epochs,
        vi_prior_fit_times,
    ) = kfold_evaluate_uci(
        dataset=dataset,
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
    model_kwargs["prior_scale_identity_multiplier"] = 1e6
    (
        vi_flat_prior_rmses,
        vi_flat_prior_negative_log_likelihoods,
        vi_flat_prior_models,
        vi_flat_prior_total_epochs,
        vi_flat_prior_fit_times,
    ) = kfold_evaluate_uci(
        dataset=dataset,
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
    (
        map_rmses,
        map_negative_log_likelihoods,
        map_models,
        map_total_epochs,
        map_fit_times,
    ) = kfold_evaluate_uci(
        dataset=dataset,
        model_class=MapDensityNetwork,
        train_seed=train_seed,
        validation_split=validation_split,
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
    llb_fit_kwargs_list = [{"pretrained_network": model} for model in map_models]
    (
        llb_rmses,
        llb_negative_log_likelihoods,
        llb_models,
        llb_total_epochs,
        llb_fit_times,
    ) = kfold_evaluate_uci(
        dataset=dataset,
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
        model_class=MapDensityEnsemble,
        train_seed=train_seed,
        validation_split=validation_split,
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


def save_results(experiment_name, dataset, results):
    """
    This function receives the experimental results. If already a json file exists to
    this experiment_name and dataset, it will read from that json file and compare the
    results for keys that exist already. When the results are not equivalent it will
    throw an AssertionError. Then it will save results of new keys into the same json
    file. If the file doesn't yet exist it will be created.
    """
    dir = Path(f"uci_data/{dataset}/results/")
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
    save=True,
):
    results = benchmark_models_uci(
        dataset=dataset,
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
    )

    if save:
        save_results(experiment_name, dataset, results)

    labels = [
        "VI-Prior",
        "VI-Flat-Prior",
        "Map",
        "Last Layer Bayesian",
        "Ensemble",
        "LLB Ensemble",
    ]
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
