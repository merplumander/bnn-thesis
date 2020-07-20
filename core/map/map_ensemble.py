from collections import Iterable

import tensorflow as tf
import tensorflow_probability as tfp

from ..network_utils import _linspace_network_indices, build_keras_model

tfd = tfp.distributions


class MapEnsemble:
    def __init__(
        self,
        n_networks=5,
        input_shape=[1],
        layer_units=[200, 100, 1],
        layer_activations=["relu", "relu", "linear"],
        l2_weight_lambda=None,
        l2_bias_lambda=None,
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        seed=0,  # can be number or Iterable of numbers
    ):
        """Class that represents an ensemble of Map (currently ML) trained neural networks.
        All networks share the same architecture specified by input_shape, layer_units,
        and layer_activations. Each network learns an underlying function and does not
        represent either aleatoric or epistemic uncertainty. The ensemble taken
        together, however, can represent some kind of (epistemic) uncertainty about the
        underlying function."""
        self.n_networks = n_networks
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.l2_weight_lambda = l2_weight_lambda
        self.l2_bias_lambda = l2_bias_lambda
        self.learning_rate = learning_rate
        if not isinstance(self.l2_weight_lambda, Iterable):
            self.l2_weight_lambda = [self.l2_weight_lambda] * self.n_networks
        if not isinstance(self.l2_bias_lambda, Iterable):
            self.l2_bias_lambda = [self.l2_bias_lambda] * self.n_networks
        if not isinstance(seed, Iterable):
            seed = [seed + (i / self.n_networks) for i in range(self.n_networks)]
        self.seed = seed
        tf.random.set_seed(self.seed[0])

        self.networks = [
            MapNetwork(
                self.input_shape,
                self.layer_units,
                self.layer_activations,
                learning_rate=self.learning_rate,
                l2_weight_lambda=l2_weight_lambda,
                l2_bias_lambda=l2_bias_lambda,
                seed=seed,
            )
            for l2_weight_lambda, l2_bias_lambda, seed in zip(
                self.l2_weight_lambda, self.l2_bias_lambda, self.seed,
            )
        ]

    def fit(self, x_train, y_train, batch_size, epochs=120, verbose=1):
        tf.random.set_seed(self.seed[0])
        for network in self.networks:
            network.fit(
                x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose
            )
        return self

    def predict_list_of_gaussians(self, x_test, n_predictions=None):
        if n_predictions is None:
            loop_indices = tf.range(0, self.n_networks)
        else:
            loop_indices = _linspace_network_indices(self.n_networks, n_predictions)
        predictive_distributions = []
        for i in loop_indices:
            predictive_distributions.append(
                self.networks[i].predict_gaussian_distribution(x_test)
            )
        return predictive_distributions

    def predict_mixture_of_gaussians(self, x_test):
        gaussians = self.predict_list_of_gaussians(x_test)
        cat_probs = tf.ones(x_test.shape + (self.n_networks,)) / self.n_networks
        return tfd.Mixture(cat=tfd.Categorical(probs=cat_probs), components=gaussians)

    def predict_list_of_deltas(self, x_test, n_predictions=None):
        if n_predictions is None:
            loop_indices = tf.range(0, self.n_networks)
        else:
            loop_indices = _linspace_network_indices(self.n_networks, n_predictions)
        predictive_distributions = []
        for i in loop_indices:
            predictive_distributions.append(
                self.networks[i].predict_delta_distribution(x_test)
            )
        return predictive_distributions

    def predict_mixture_of_deltas(self, x_test):
        deltas = self.predict_list_of_deltas(x_test)
        cat_probs = tf.ones(x_test.shape + (self.n_networks,)) / self.n_networks
        return tfd.Mixture(cat=tfd.Categorical(probs=cat_probs), components=deltas)

    def predict_mean_functions(self, x_test):
        predictions = []
        for network in self.networks:
            prediction = network(x_test)
            predictions.append(prediction)
        return predictions

    def predict(self, x_test):
        return self.predict_mixture_of_gaussians(x_test)

    def get_weights(self):
        networks_weights_list = []
        for network in self.networks:
            networks_weights_list.append(network.get_weights())
        return networks_weights_list

    def set_weights(self, networks_weights_list):
        for network, network_weights in zip(self.networks, networks_weights_list):
            network.set_weights(network_weights)


class MapNetwork:
    def __init__(
        self,
        input_shape=[1],
        layer_units=[200, 100, 1],
        layer_activations=["relu", "relu", "linear"],
        l2_weight_lambda=None,  # float or list of floats
        l2_bias_lambda=None,
        names=None,
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        seed=0,
    ):
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.l2_weight_lambda = l2_weight_lambda
        self.l2_bias_lambda = l2_bias_lambda
        self.names = names
        self.learning_rate = learning_rate
        self.seed = seed
        tf.random.set_seed(self.seed)
        self.network = build_keras_model(
            self.input_shape,
            self.layer_units,
            self.layer_activations,
            l2_weight_lambda=self.l2_weight_lambda,
            l2_bias_lambda=self.l2_bias_lambda,
            names=self.names,
        )

        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mean_squared_error",
        )

    # @tf.function
    def fit(self, x_train, y_train, batch_size, epochs=120, verbose=1):
        # steps_per_epoch = x_train.shape[0] // batch_size
        tf.random.set_seed(self.seed)
        self.network.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            # necessary for tf function
            # steps_per_epoch=steps_per_epoch,
            verbose=verbose,
        )
        n_train = x_train.shape[0]
        # using Bessel's correction to get an unbiased estimator:
        self.sample_std = (n_train / (n_train - 1)) * tf.math.reduce_std(
            self.network.predict(x_train) - y_train
        )
        return self

    def predict_mean_function(self, x_test):
        prediction = self.network(x_test)
        return prediction

    def predict_gaussian_distribution(self, x_test):
        prediction = self.predict_mean_function(x_test)
        predictive_distribution = tfd.Normal(loc=prediction, scale=self.sample_std)
        return predictive_distribution

    def predict_delta_distribution(self, x_test):
        prediction = self.predict_mean_function(x_test)
        predictive_delta_distribution = tfp.distributions.Deterministic(loc=prediction)
        return predictive_delta_distribution

    def predict(self, x_test):
        return self.predict_gaussian_distribution(x_test)

    def __call__(self, x_test):
        return self.predict(x_test)

    def get_weights(self):
        return self.network.get_weights()

    def set_weights(self, weights_list):
        self.network.set_weights(weights_list)
