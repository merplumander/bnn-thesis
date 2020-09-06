from collections import Iterable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from core.map import MapDensityNetwork, MapNetwork

from ..preprocessing import StandardizePreprocessor
from .bayesian_linear_regression import BayesianLinearRegression

tfd = tfp.distributions


class PostHocLastLayerBayesianEnsemble:
    def __init__(
        self,
        n_networks=5,
        input_shape=[1],
        layer_units=[200, 100, 1],
        layer_activations=["relu", "relu", "linear"],
        initial_unconstrained_scale=None,
        transform_unconstrained_scale_factor=0.05,  # factor to be used in the calculation of the actual noise std.
        l2_weight_lambda=None,  # float or list of floats
        l2_bias_lambda=None,
        preprocess_x=False,
        preprocess_y=False,
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        seed=0,
    ):

        self.n_networks = n_networks
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        if not isinstance(initial_unconstrained_scale, Iterable):
            self.initial_unconstrained_scale = [
                initial_unconstrained_scale
            ] * self.n_networks
        self.transform_unconstrained_scale_factor = transform_unconstrained_scale_factor
        self.l2_weight_lambda = l2_weight_lambda
        self.l2_bias_lambda = l2_bias_lambda
        self.preprocess_x = preprocess_x
        self.preprocess_y = preprocess_y
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
                initial_unconstrained_scale=initial_unconstrained_scale,
                transform_unconstrained_scale_factor=self.transform_unconstrained_scale_factor,
                l2_weight_lambda=self.l2_weight_lambda,
                l2_bias_lambda=self.l2_bias_lambda,
                preprocess_x=self.preprocess_x,
                preprocess_y=self.preprocess_y,
                learning_rate=self.learning_rate,
                seed=seed,
            )
            for seed, initial_unconstrained_scale in zip(
                self.seed, self.initial_unconstrained_scale
            )
        ]

    @property
    def total_epochs(self):
        return [network.total_epochs for network in self.networks]

    def fit(
        self,
        x_train,
        y_train,
        batch_size=1,
        epochs=1,
        early_stop_callback=None,
        validation_split=0.0,
        validation_data=None,
        verbose=1,
        pretrained_networks=None,
    ):
        tf.random.set_seed(self.seed[0])
        if pretrained_networks is None:
            pretrained_networks = [None] * self.n_networks

        for network, pretrained_network in zip(self.networks, pretrained_networks):
            network.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                early_stop_callback=early_stop_callback,
                validation_split=validation_split,
                validation_data=validation_data,
                verbose=verbose,
                pretrained_network=pretrained_network,
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
        cat_probs = (
            tf.ones((x_test.shape[0],) + (1, self.n_networks)) / self.n_networks
        )  # tf.ones((x_test.shape + (self.n_networks,))) / self.n_networks
        prediction = tfd.Mixture(
            cat=tfd.Categorical(probs=cat_probs), components=prediction_list
        )
        return prediction

    def predict_moment_matched_gaussian(self, x_test):
        predictive_mixture = self.predict_mixture(x_test)
        return tfd.Normal(
            loc=predictive_mixture.mean(), scale=predictive_mixture.stddev()
        )

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
        initial_unconstrained_scale=None,
        transform_unconstrained_scale_factor=0.05,  # factor to be used in the calculation of the actual noise std.
        l2_weight_lambda=None,  # float or list of floats
        l2_bias_lambda=None,
        preprocess_x=False,
        preprocess_y=False,
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        seed=0,
    ):
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.initial_unconstrained_scale = initial_unconstrained_scale
        self.transform_unconstrained_scale_factor = transform_unconstrained_scale_factor
        self.l2_weight_lambda = l2_weight_lambda
        self.l2_bias_lambda = l2_bias_lambda
        self.preprocess_x = preprocess_x
        self.preprocess_y = preprocess_y
        self.learning_rate = learning_rate
        self.seed = seed

        if self.preprocess_y:
            self.y_preprocessor = StandardizePreprocessor()
        names = [None] * (len(self.layer_units) - 2) + ["feature_extractor", "output"]
        tf.random.set_seed(self.seed)
        if self.initial_unconstrained_scale is None:
            self.network = MapNetwork(
                self.input_shape,
                self.layer_units,
                self.layer_activations,
                self.l2_weight_lambda,
                self.l2_bias_lambda,
                preprocess_x=self.preprocess_x,
                learning_rate=self.learning_rate,
                names=names,
                seed=self.seed,
            )
        else:
            self.network = MapDensityNetwork(
                self.input_shape,
                self.layer_units,
                self.layer_activations,
                self.initial_unconstrained_scale,
                self.transform_unconstrained_scale_factor,
                self.l2_weight_lambda,
                self.l2_bias_lambda,
                preprocess_x=self.preprocess_x,
                learning_rate=self.learning_rate,
                names=names,
                seed=self.seed,
            )

    @property
    def total_epochs(self):
        return self.network.total_epochs

    def fit_preprocessing(self, y_train):
        if self.preprocess_y:
            self.y_preprocessor.fit(y_train)

    def fit(
        self,
        x_train,
        y_train,
        batch_size=1,
        epochs=1,
        early_stop_callback=None,
        validation_split=0.0,
        validation_data=None,
        verbose=1,
        pretrained_network=None,
    ):
        tf.random.set_seed(self.seed)
        self.fit_preprocessing(y_train)
        if self.preprocess_y:
            y_train = self.y_preprocessor.transform(y_train)
        if pretrained_network is None:
            self.network.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                early_stop_callback=early_stop_callback,
                validation_split=validation_split,
                validation_data=validation_data,
                verbose=verbose,
            )
        else:
            self.network = pretrained_network

        self.feature_extractor = tf.keras.Model(
            self.network.network.inputs,
            self.network.network.get_layer("feature_extractor").output,
        )
        features_train = self.feature_extractor(x_train).numpy()
        features_train = np.hstack((features_train, np.ones((x_train.shape[0], 1))))
        # "fit" bayesian linear regression
        n_features = features_train.shape[1]
        self.blr_model = BayesianLinearRegression(
            mu_0=np.zeros((n_features, 1)),  # np.zeros(n_features),
            V_0=1e3 * np.eye(n_features),  #
            a_0=-n_features / 2,
            b_0=0,
        )
        self.blr_model.fit(features_train, y_train)
        return self

    def predict(self, x):
        features_test = self.feature_extractor(x).numpy().astype("float32")
        features_test = np.hstack((features_test, np.ones((x.shape[0], 1))))
        df, loc, scale = self.blr_model.predict(features_test)
        df = np.float32(df)
        loc = loc.astype("float32")
        scale = scale.astype("float32")
        scale = np.expand_dims(scale, axis=scale.ndim)
        if self.preprocess_y:
            loc = self.y_preprocessor.inverse_transform(loc)
            if self.y_preprocessor.std is not None:
                scale *= self.y_preprocessor.std
        return tfd.StudentT(df=df, loc=loc, scale=scale)

    def __call__(self, x):
        return self.predict(x)

    def get_weights(self):
        """
        Returns the weights of all layers including the one that is discarded and the marginal t distribution
        of the last layer weights.
        """
        df, loc, scale = self.blr_model.get_marginal_beta()
        last_layer_weight_distribution = tfd.StudentT(df=df, loc=loc, scale=scale)
        return self.network.get_weights(), last_layer_weight_distribution
