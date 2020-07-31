from collections import Iterable

import tensorflow as tf
import tensorflow_probability as tfp

from ..network_utils import (
    _linspace_network_indices,
    build_keras_model,
    regularization_lambda_to_prior_scale,
    transform_unconstrained_scale,
)

tfd = tfp.distributions


class AddSigmaLayer(tf.keras.layers.Layer):
    """
    Layer that passes its input through unchanged but enlarges the returned tensor in
    the last dimension by adding a constant (but trainable) value. Say the input to the
    layer was of shape (3, 1), i.e.:
    [[0.5],
     [0.7],
     [0.1]]
    This layer would return (before training) an output of shape (3, 2), in this case:
    [[0.5, initial_value]
     [0.7, initial_value]
     [0.1, initial_value]]
    This is useful to model homoscedastic noise variance explicitly.
    """

    def __init__(self, initial_value=0.0):
        super(AddSigmaLayer, self).__init__()
        self.sigma = tf.Variable(
            [[initial_value]], name="sigma", dtype="float32", trainable=True
        )

    def call(self, input):
        # Need the matmul workaround since the commented code below throws error when
        #  tensorflow is generating the graph.
        _ones = tf.ones_like(input)
        sigmas = tf.matmul(_ones, self.sigma)
        # sigmas = tf.repeat(self.sigma, input.shape[0])
        # sigmas = tf.expand_dims(sigmas, axis=-1)
        return tf.concat([input, sigmas], axis=-1)


class MapDensityEnsemble:
    def __init__(
        self,
        n_networks=5,
        input_shape=[1],
        layer_units=[200, 100, 2],
        initial_unconstrained_scales=None,  # None, float or Iterable
        transform_unconstrained_scale_factor=0.05,
        l2_weight_lambda=None,
        l2_bias_lambda=None,
        layer_activations=["relu", "relu", "linear"],
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
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
        assert layer_units[-1] == 2
        self.n_networks = n_networks
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.initial_unconstrained_scales = initial_unconstrained_scales
        if not isinstance(initial_unconstrained_scales, Iterable):
            self.initial_unconstrained_scales = [
                initial_unconstrained_scales
            ] * self.n_networks
        self.transform_unconstrained_scale_factor = transform_unconstrained_scale_factor
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
            MapDensityNetwork(
                self.input_shape,
                self.layer_units,
                self.layer_activations,
                initial_unconstrained_scale=initial_unconstrained_scale,
                transform_unconstrained_scale_factor=transform_unconstrained_scale_factor,
                l2_weight_lambda=l2_weight_lambda,
                l2_bias_lambda=l2_bias_lambda,
                learning_rate=self.learning_rate,
                seed=seed,
            )
            for initial_unconstrained_scale, l2_weight_lambda, l2_bias_lambda, seed in zip(
                self.initial_unconstrained_scales,
                self.l2_weight_lambda,
                self.l2_bias_lambda,
                self.seed,
            )
        ]

    def fit(self, x_train, y_train, batch_size, epochs=200, verbose=1):
        tf.random.set_seed(self.seed[0])
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
            loop_indices = _linspace_network_indices(self.n_networks, n_predictions)
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
        initial_unconstrained_scale=None,  # unconstrained noise standard deviation. When None, the noise std is model by a second network output. When float, the noise std is homoscedastic.
        transform_unconstrained_scale_factor=0.05,  # factor to be used in the calculation of the actual noise std.
        l2_weight_lambda=None,  # float or list of floats
        l2_bias_lambda=None,
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
        self.transform_unconstrained_scale_factor = transform_unconstrained_scale_factor
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
            n_train_data = y_train.shape[0]
        if batch_size is None:
            batch_size = n_train_data
        tf.random.set_seed(self.seed)
        self.network.fit(
            x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose
        )
        return self

    def evaluate(self, x, y, batch_size=None, verbose=0):
        return self.network.evaluate(x, y, batch_size=batch_size, verbose=verbose)

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
