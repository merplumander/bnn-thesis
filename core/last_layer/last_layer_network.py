from collections import Iterable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from core.map import MapDensityNetwork, MapNetwork

from .bayesian_linear_regression import BayesianLinearRegression

tfd = tfp.distributions


class PostHocLastLayerBayesianEnsemble:
    def __init__(
        self,
        n_networks=5,
        input_shape=[1],
        layer_units=[200, 100, 1],
        layer_activations=["relu", "relu", "linear"],
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        seed=0,
    ):

        self.n_networks = n_networks
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.learning_rate = learning_rate
        if not isinstance(seed, Iterable):
            seed = [seed + (i / self.n_networks) for i in range(self.n_networks)]
        self.seed = seed
        tf.random.set_seed(self.seed[0])
        self.networks = [
            PostHocLastLayerBayesianNetwork(
                self.input_shape,
                self.layer_units,
                self.layer_activations,
                learning_rate=self.learning_rate,
                seed=seed,
            )
            for seed in self.seed
        ]

    def fit(self, x_train, y_train, batch_size, epochs=120, verbose=1):
        tf.random.set_seed(self.seed[0])
        for network in self.networks:
            network.fit(
                x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose
            )
        return self

    def predict_list(self, x_test):
        predictions = []
        for network in self.networks:
            prediction = network(x_test)
            predictions.append(prediction)
        return predictions

    def predict_mixture(self, x_test):
        prediction_list = self.predict_list(x_test)
        cat_probs = tf.ones((x_test.shape + (self.n_networks,))) / self.n_networks
        prediction = tfd.Mixture(
            cat=tfd.Categorical(probs=cat_probs), components=prediction_list
        )
        return prediction

    def predict(self, x_test):
        return self.predict_mixture(x_test)

    def get_weights(self):
        networks_weights_list = []
        for network in self.networks:
            networks_weights_list.append(network.get_weights())
        return networks_weights_list


class PostHocLastLayerBayesianNetwork:
    def __init__(
        self,
        input_shape=[1],
        layer_units=[200, 100, 1],
        layer_activations=["relu", "relu", "linear"],
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        seed=0,
    ):
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.learning_rate = learning_rate
        self.seed = seed
        tf.random.set_seed(self.seed)
        names = [None] * (len(self.layer_units) - 2) + ["feature_extractor", "output"]
        self.network = MapNetwork(
            self.input_shape,
            self.layer_units,
            self.layer_activations,
            learning_rate=self.learning_rate,
            names=names,
            seed=self.seed,
        )

    def fit(self, x_train, y_train, batch_size, epochs=120, verbose=1):
        tf.random.set_seed(self.seed)
        self.network.fit(
            x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose
        )

        self.feature_extractor = tf.keras.Model(
            self.network.network.inputs,
            self.network.network.get_layer("feature_extractor").output,
        )
        features_train = self.feature_extractor(x_train).numpy()
        features_train = np.hstack((features_train, np.ones((x_train.shape[0], 1))))

        # "fit" bayesian linear regression
        self.blr_model = BayesianLinearRegression(
            mu=np.zeros(features_train.shape[1]),
            v=1000 * np.eye(features_train.shape[1]),
            a=0.5,
            b=0.5,
        )
        self.blr_model.fit(features_train, y_train.flatten())
        return self

    def predict(self, x_test):
        features_test = self.feature_extractor(x_test).numpy().astype("float32")
        features_test = np.hstack((features_test, np.ones((x_test.shape[0], 1))))
        df, loc, scale = self.blr_model.predict(features_test)
        df = np.float32(df)
        loc = loc.astype("float32")
        loc = np.expand_dims(loc, axis=loc.ndim)
        scale = scale.astype("float32")
        scale = np.expand_dims(scale, axis=scale.ndim)
        return tfd.StudentT(df=df, loc=loc, scale=scale)

    def __call__(self, x_test):
        return self.predict(x_test)

    def get_weights(self):
        """
        Returns the weights of all layers including the one that is discarded and the marginal t distribution
        of the last layer weights.
        """
        df, loc, scale = self.blr_model.get_marginal_beta()
        last_layer_weight_distribution = tfd.StudentT(df=df, loc=loc, scale=scale)
        return self.network.get_weights(), last_layer_weight_distribution
