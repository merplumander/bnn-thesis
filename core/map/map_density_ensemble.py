from collections import Iterable

import tensorflow as tf
import tensorflow_probability as tfp

from ..network_utils import (
    AddSigmaLayer,
    _linspace_network_indices,
    build_keras_model,
    regularization_lambda_to_prior_scale,
    transform_unconstrained_scale,
)
from ..preprocessing import StandardizePreprocessor

tfd = tfp.distributions


class MapDensityEnsemble:
    def __init__(
        self,
        n_networks=5,
        input_shape=[1],
        layer_units=[200, 100, 2],
        layer_activations=["relu", "relu", "linear"],
        initial_unconstrained_scale=None,  # None, float or Iterable
        transform_unconstrained_scale_factor=0.05,
        l2_weight_lambda=None,
        l2_bias_lambda=None,
        preprocess_x=False,
        preprocess_y=False,
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        names=None,
        seed=0,  # can be number or Iterable of numbers
    ):
        """
        Class that represents an ensemble of Map trained neural networks.
        Each network is trained to minimize the (gaussian) log likelihood of the data
        and learns both the mean and the variance of this gaussian. Each network thereby
        represents aleatoric uncertainty. Taken together as an ensemble, epistemic
        uncertainty is also represented in the different mean functions that are
        learned.
        """
        self.n_networks = n_networks
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.initial_unconstrained_scale = initial_unconstrained_scale
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
        if not isinstance(self.l2_weight_lambda, Iterable):
            self.l2_weight_lambda = [self.l2_weight_lambda] * self.n_networks
        if not isinstance(self.l2_bias_lambda, Iterable):
            self.l2_bias_lambda = [self.l2_bias_lambda] * self.n_networks
        if not isinstance(seed, Iterable):
            seed = [seed + (i / self.n_networks) * 1e3 for i in range(self.n_networks)]
        self.names = names
        self.seed = seed
        tf.random.set_seed(self.seed[0])

        self.networks = [
            MapDensityNetwork(
                self.input_shape,
                self.layer_units,
                self.layer_activations,
                initial_unconstrained_scale=initial_unconstrained_scale,
                transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
                l2_weight_lambda=l2_weight_lambda,
                l2_bias_lambda=l2_bias_lambda,
                preprocess_x=self.preprocess_x,
                preprocess_y=self.preprocess_y,
                learning_rate=self.learning_rate,
                names=self.names,
                seed=seed,
            )
            for initial_unconstrained_scale, l2_weight_lambda, l2_bias_lambda, seed in zip(
                self.initial_unconstrained_scale,
                self.l2_weight_lambda,
                self.l2_bias_lambda,
                self.seed,
            )
        ]

    def clone(self):
        if isinstance(
            self.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule
        ):
            print(
                "Cloning will likely not work as desired. Check wether the learning rate schedule is reset."
            )
        return MapDensityEnsemble(
            n_networks=self.n_networks,
            input_shape=self.input_shape,
            layer_units=self.layer_units,
            layer_activations=self.layer_activations,
            initial_unconstrained_scale=self.initial_unconstrained_scale,
            transform_unconstrained_scale_factor=self.transform_unconstrained_scale_factor,
            l2_weight_lambda=self.l2_weight_lambda,
            l2_bias_lambda=self.l2_bias_lambda,
            preprocess_x=self.preprocess_x,
            preprocess_y=self.preprocess_y,
            learning_rate=self.learning_rate,
            seed=self.seed,
        )

    def fit(
        self,
        x_train,
        y_train,
        batch_size,
        epochs=200,
        early_stop_callback=None,
        validation_split=0.0,
        validation_data=None,
        verbose=1,
    ):
        tf.random.set_seed(self.seed[0])
        if not isinstance(epochs, Iterable):
            epochs = [epochs] * self.n_networks
        self.total_epochs = []
        for network, epoch in zip(self.networks, epochs):
            network.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epoch,
                early_stop_callback=early_stop_callback,
                validation_split=validation_split,
                validation_data=validation_data,
                verbose=verbose,
            )
            self.total_epochs.append(network.total_epochs)
        return self

    def rmse(self, x, y):
        mean_prediction = self.predict(x).mean()
        return tf.keras.metrics.RootMeanSquaredError()(mean_prediction, y)

    def predict_list_of_gaussians(self, x_test, n_predictions=None):
        if n_predictions is None:
            loop_indices = tf.range(0, self.n_networks)
        else:
            loop_indices = _linspace_network_indices(self.n_networks, n_predictions)
        predictive_distributions = []
        for i in loop_indices:
            predictive_distributions.append(self.networks[i](x_test))
        return predictive_distributions

    def predict_mixture_of_gaussians(self, x_test):
        gaussians = self.predict_list_of_gaussians(x_test)
        cat_probs = tf.ones((x_test.shape[0],) + (1, self.n_networks)) / self.n_networks
        return tfd.Mixture(cat=tfd.Categorical(probs=cat_probs), components=gaussians)

    def predict_moment_matched_gaussian(self, x_test):
        predictive_mixture = self.predict_mixture_of_gaussians(x_test)
        return tfd.Normal(
            loc=predictive_mixture.mean(), scale=predictive_mixture.stddev()
        )

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


class MapDensityNetwork:
    def __init__(
        self,
        input_shape=[1],
        layer_units=[200, 100, 2],
        layer_activations=["relu", "relu", "linear"],
        initial_unconstrained_scale=None,  # unconstrained noise standard deviation. When None, the noise std is model by a second network output. When float, the noise std is homoscedastic.
        transform_unconstrained_scale_factor=0.05,  # factor to be used in the calculation of the actual noise std.
        l2_weight_lambda=None,  # float or list of floats
        l2_bias_lambda=None,
        preprocess_x=False,
        preprocess_y=False,
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        names=None,
        seed=0,
    ):
        if initial_unconstrained_scale is not None:
            assert layer_units[-1] == 1
        else:
            assert layer_units[-1] == 2
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
        self.names = names
        self.seed = seed
        tf.random.set_seed(self.seed)

        if self.preprocess_y:
            self.y_preprocessor = StandardizePreprocessor()

        self.network = build_keras_model(
            self.input_shape,
            self.layer_units,
            self.layer_activations,
            l2_weight_lambda=self.l2_weight_lambda,
            l2_bias_lambda=self.l2_bias_lambda,
            normalization_layer=self.preprocess_x,
            names=self.names,
        )
        if initial_unconstrained_scale is not None:
            # print("homoscedastic noise")
            self.network.add(AddSigmaLayer(initial_unconstrained_scale))
        self.network.add(
            tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :1],
                    scale=transform_unconstrained_scale(
                        t[..., 1:], transform_unconstrained_scale_factor
                    ),
                )
            )
        )

        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=MapDensityNetwork.negative_log_likelihood,
        )

    def clone(self):
        if isinstance(
            self.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule
        ):
            print(
                "Cloning will likely not work as desired. Check wether the learning rate schedule is reset."
            )
        return MapDensityNetwork(
            input_shape=self.input_shape,
            layer_units=self.layer_units,
            layer_activations=self.layer_activations,
            initial_unconstrained_scale=self.initial_unconstrained_scale,
            transform_unconstrained_scale_factor=self.transform_unconstrained_scale_factor,
            l2_weight_lambda=self.l2_weight_lambda,
            l2_bias_lambda=self.l2_bias_lambda,
            preprocess_x=self.preprocess_x,
            preprocess_y=self.preprocess_y,
            learning_rate=self.learning_rate,
            names=self.names,
            seed=self.seed,
        )

    def negative_log_prior(self):
        params = self.get_weights()
        log_prob = 0
        weight_priors, bias_priors = self._get_priors()
        for w, b, w_prior, b_prior in zip(
            params[::2], params[1::2], weight_priors, bias_priors
        ):
            log_prob += tf.reduce_sum(w_prior.log_prob(w))
            log_prob += tf.reduce_sum(b_prior.log_prob(b))
        return -log_prob

    def negative_log_likelihood(y, p_y):
        return -p_y.log_prob(y)

    def negative_log_posterior(y, p_y):
        # log_prior = self.negative_log_prior()
        log_likelihood = MapDensityNetwork.log_likelihood(y, p_y)
        return log_likelihood  # + log_prior

    def _get_priors(self):
        assert self.l2_weight_lambda is not None and self.l2_bias_lambda is not None
        l2_weight_lambda = self.l2_weight_lambda
        l2_bias_lambda = self.l2_bias_lambda
        if not isinstance(l2_weight_lambda, Iterable):
            l2_weight_lambda = [l2_weight_lambda] * len(self.layer_units)
        if not isinstance(l2_bias_lambda, Iterable):
            l2_bias_lambda = [l2_bias_lambda] * len(self.layer_units)
        w_priors = []
        b_priors = []
        for w_lambda, b_lambda, u1, u2 in zip(
            l2_weight_lambda,
            l2_bias_lambda,
            self.input_shape + self.layer_units[:-1],
            self.layer_units,
        ):
            w_scale = regularization_lambda_to_prior_scale(w_lambda)
            b_scale = regularization_lambda_to_prior_scale(b_lambda)
            w_priors.append(tfd.Normal(0, tf.ones((u1, u2)) * w_scale))
            b_priors.append(tfd.Normal(0, tf.ones(u2) * b_scale))
        return w_priors, b_priors

    def sample_prior_state(self, n_states=1, seed=0):
        weight_priors, bias_priors = self._get_priors()
        tf.random.set_seed(seed)
        state = []
        for w_prior, b_prior in zip(weight_priors, bias_priors):
            w = w_prior.sample()
            b = b_prior.sample()
            state.extend([w, b])
        return state

    def fit_preprocessing(self, y_train):
        if self.preprocess_y:
            self.y_preprocessor.fit(y_train)

    def fit(
        self,
        x_train,
        y_train,
        n_train_data=None,
        batch_size=None,
        epochs=1,
        early_stop_callback=None,
        validation_split=0.0,
        validation_data=None,
        verbose=1,
    ):
        if self.preprocess_x:
            self.network.get_layer("normalization").adapt(x_train, reset_state=True)
        self.fit_preprocessing(y_train)
        if self.preprocess_y:
            y_train = self.y_preprocessor.transform(y_train)
            if validation_data:
                y_validation = self.y_preprocessor.transform(validation_data[1])
                validation_data = (validation_data[0], y_validation)
        if n_train_data is None:
            n_train_data = y_train.shape[0]
        if batch_size is None:
            batch_size = n_train_data
        callbacks = []
        if early_stop_callback:
            callbacks.append(early_stop_callback)
        tf.random.set_seed(self.seed)
        history = self.network.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            verbose=verbose,
        )
        self.history = history
        self.total_epochs = len(history.history["loss"])
        # if restore_best_weights is True, then the number of total epochs needs to be
        # adjusted by the patience
        if early_stop_callback is not None:
            if early_stop_callback.restore_best_weights:
                self.total_epochs -= early_stop_callback.patience
        return self

    def evaluate(self, x, y, batch_size=None, verbose=0):
        if self.preprocess_y:
            y = self.y_preprocessor.transform(y)
        return self.network.evaluate(x, y, batch_size=batch_size, verbose=verbose)

    def rmse(self, x, y):
        mean_prediction = self.predict(x).mean()
        return tf.keras.metrics.RootMeanSquaredError()(mean_prediction, y)

    def predict(self, x):
        prediction = self.network(x)
        if self.preprocess_y:
            mean = prediction.mean()
            std = prediction.stddev()
            mean = self.y_preprocessor.inverse_transform(mean)
            if self.y_preprocessor.std is not None:
                std *= self.y_preprocessor.std
            prediction = tfd.Normal(mean, std)
        return prediction

    def predict_with_prior_samples(self, x_test, n_samples=5, seed=0):
        saved_weights = self.get_weights()
        predictive_distributions = []
        for sample in range(n_samples):
            prior_sample = self.sample_prior_state(seed=seed + sample * 0.01)
            self.set_weights(prior_sample)
            prediction = self.predict(x_test)
            predictive_distributions.append(prediction)
        self.set_weights(saved_weights)
        return predictive_distributions

    def __call__(self, x_test):
        return self.predict(x_test)

    def get_weights(self):
        weights = self.network.get_weights()
        weights[-1] = weights[-1][0]
        return weights

    def set_weights(self, weights_list):
        weights_list[-1] = weights_list[-1].reshape(1, 1)
        self.network.set_weights(weights_list)
