import tensorflow as tf
import tensorflow_probability as tfp

from ..network_utils import build_keras_model, transform_unconstrained_scale

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
                self.learning_rate,
                self.seed + i,
            )
            for i in range(self.n_networks)
        ]

    def fit(self, x_train, y_train, batch_size, epochs=200):
        tf.random.set_seed(self.seed)
        for network in self.networks:
            network.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        return self

    def predict_list_of_gaussians(self, x_test):
        predictive_distributions = []
        for network in self.networks:
            predictive_distributions.append(network(x_test))
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
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        seed=0,
    ):
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.learning_rate = learning_rate
        self.seed = seed
        tf.random.set_seed(self.seed)
        self.network = build_keras_model(
            self.input_shape, self.layer_units, self.layer_activations
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

    def negative_log_likelihood(y, p_y):
        return -p_y.log_prob(y)

    def fit(self, x_train, y_train, batch_size, epochs=120):
        tf.random.set_seed(self.seed)
        self.network.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        return self

    # def train(self, x_train, y_train, batch_size, epochs=120):
    #     return self.fit(x_train, y_train, batch_size, epochs=120)

    def predict(self, x_test):
        return self.network(x_test)

    def __call__(self, x_test):
        return self.predict(x_test)

    def get_weights(self):
        return self.network.get_weights()

    def set_weights(self, weights_list):
        self.network.set_weights(weights_list)
