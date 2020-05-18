from collections import Iterable

import tensorflow as tf
import tensorflow_probability as tfp

from ..network_utils import (
    PriorRegularizer,
    build_keras_model,
    transform_unconstrained_scale,
)

tfd = tfp.distributions


class MapDensityEnsemble:
    def __init__(
        self,
        n_networks=5,
        input_shape=[1],
        layer_units=[200, 100, 2],
        layer_activations=["relu", "relu", "linear"],
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        seed=0,
    ):
        """
        Class that represents an ensemble of Map (currently ML) trained neural networks.
        Each network is trained to minimize the (gaussian) log likelihood of the data
        and learns both the mean and the variance of this gaussian. Each network thereby
        represents aleatoric uncertainty. Taken together as an ensemble, epistemic
        uncertainty is also represented in the different mean functions that are
        learned.
        """
        assert layer_units[-1] == 2
        self.n_networks = n_networks
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.learning_rate = learning_rate
        self.seed = seed
        tf.random.set_seed(self.seed)
        self.networks = [
            MapDensityNetwork(
                self.input_shape,
                self.layer_units,
                self.layer_activations,
                learning_rate=self.learning_rate,
                seed=self.seed + i,
            )
            for i in range(self.n_networks)
        ]

    def fit(self, x_train, y_train, batch_size, epochs=200, verbose=1):
        tf.random.set_seed(self.seed)
        if not isinstance(epochs, Iterable):
            epochs = [epochs] * self.n_networks
        for network, epoch in zip(self.networks, epochs):
            network.fit(
                x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=verbose
            )
        return self

    def predict_list_of_gaussians(self, x_test, n_predictions=None):
        if n_predictions is None:
            loop_indices = tf.range(0, self.n_networks)
        else:
            # think: tf.linspace(0, self.n_networks - 1, n_predictions).
            # The rest is just annoying tensorflow issues
            loop_indices = tf.cast(
                tf.math.ceil(
                    tf.linspace(
                        0,
                        tf.constant(self.n_networks - 1, dtype=tf.float32),
                        n_predictions,
                    )
                ),
                tf.int32,
            )
        predictive_distributions = []
        for i in loop_indices:
            predictive_distributions.append(self.networks[i](x_test))
        return predictive_distributions

    def predict_mixture_of_gaussians(self, x_test):
        gaussians = self.predict_list_of_gaussians(x_test)
        cat_probs = tf.ones(x_test.shape + (self.n_networks,)) / self.n_networks
        return tfd.Mixture(cat=tfd.Categorical(probs=cat_probs), components=gaussians)

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
        weight_priors=None,
        bias_priors=None,
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        seed=0,
    ):
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.weight_priors = weight_priors
        self.bias_priors = bias_priors
        self.learning_rate = learning_rate
        self.seed = seed
        tf.random.set_seed(self.seed)
        # There is a problem with this PriorRegularizer
        if weight_priors is not None:
            weight_regularizers = [
                PriorRegularizer(w_prior) for w_prior in weight_priors
            ]
        else:
            weight_regularizers = None
        if bias_priors is not None:
            bias_regularizers = [PriorRegularizer(b_prior) for b_prior in bias_priors]
        else:
            bias_regularizers = None
        self.network = build_keras_model(
            self.input_shape,
            self.layer_units,
            self.layer_activations,
            kernel_regularizers=weight_regularizers,
            bias_regularizers=bias_regularizers,
        )
        self.network.add(
            tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :1], scale=transform_unconstrained_scale(t[..., 1:])
                )
            )
        )

        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=MapDensityNetwork.negative_log_likelihood,
        )

    def negative_log_prior(self):
        params = self.get_weights()
        log_prob = 0
        for w, b, w_prior, b_prior in zip(
            params[::2], params[1::2], self.weight_priors, self.bias_priors
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

    def sample_prior_state(self, n_states=1, seed=0):
        tf.random.set_seed(seed)
        state = []
        for w_prior, b_prior in zip(self.weight_priors, self.bias_priors):
            w = w_prior.sample()
            b = b_prior.sample()
            state.extend([w, b])
        return state

    def fit(
        self,
        x_train,
        y_train,
        n_train_data=None,
        batch_size=None,
        epochs=120,
        verbose=1,
    ):
        if n_train_data is None:
            self.n_train_data = y_train.shape[0]
        if batch_size is None:
            batch_size = n_train_data
        tf.random.set_seed(self.seed)
        self.network.fit(
            x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose
        )
        return self

    # def train(self, x_train, y_train, batch_size, epochs=120):
    #     return self.fit(x_train, y_train, batch_size, epochs=120)

    def predict(self, x_test):
        return self.network(x_test)

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
        return self.network.get_weights()

    def set_weights(self, weights_list):
        self.network.set_weights(weights_list)
