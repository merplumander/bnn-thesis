import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp

from .network_utils import build_model

tfd = tfp.distributions


class MapDensityEnsemble:
    def __init__(
        self,
        n_networks=5,
        input_shape=[1],
        layer_units=[200, 100, 2],
        layer_activations=["relu", "relu", "linear"],
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
        self.seed = seed
        tf.random.set_seed(self.seed)
        self.networks = []
        for i in range(self.n_networks):
            model = build_model(
                self.input_shape, self.layer_units, self.layer_activations
            )
            model.add(
                tfp.layers.DistributionLambda(
                    lambda t: tfd.Normal(
                        loc=t[..., :1], scale=1e-6 + tf.math.softplus(0.05 * t[..., 1:])
                    )
                )
            )
            self.networks.append(model)

        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
        )
        for network in self.networks:
            network.compile(
                optimizer=tf.keras.optimizers.Adam(lr_schedule),
                loss=MapDensityEnsemble.negative_log_likelihood,
            )

    def negative_log_likelihood(y, p_y):
        return -p_y.log_prob(y)

    def train(self, x_train, y_train, batchsize_train, epochs=200):
        tf.random.set_seed(self.seed)
        for network in self.networks:
            network.fit(x_train, y_train, epochs=epochs, batch_size=batchsize_train)

    def predict_gaussian(self, x_test):
        means = np.zeros((self.n_networks,) + x_test.shape)
        stds = np.zeros((self.n_networks,) + x_test.shape)
        for i, network in enumerate(self.networks):
            prediction = network(x_test)
            means[i] = prediction.mean()
            stds[i] = prediction.stddev()
        gaussian_mean = 1 / self.n_networks * np.sum(means, axis=0)
        gaussian_var = (
            1 / self.n_networks * np.sum(stds ** 2 + means ** 2, axis=0)
            - gaussian_mean ** 2
        )
        gaussian_std = np.sqrt(gaussian_var)
        return gaussian_mean, gaussian_std

    def predict(self, x_test):
        return self.predict_gaussian(x_test)

    def predict_mixture_of_gaussian(self, x_test):
        means = []
        stds = []
        for network in self.networks:
            prediction = network(x_test)
            means.append(prediction.mean().numpy())
            stds.append(prediction.stddev().numpy())
        return means, stds
