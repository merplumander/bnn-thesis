import numpy as np
import tensorflow as tf

from .network_utils import build_model


class MapEnsemble:
    def __init__(
        self,
        n_networks=5,
        input_shape=[1],
        layer_units=[200, 100, 1],
        layer_activations=["relu", "relu", "linear"],
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        seed=0,
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
        self.learning_rate = learning_rate
        self.seed = seed
        tf.random.set_seed(self.seed)
        self.networks = [
            MapNetwork(
                self.input_shape,
                self.layer_units,
                self.layer_activations,
                self.learning_rate,
                seed=self.seed + i,
            )
            for i in range(self.n_networks)
        ]

    def train(self, x_train, y_train, batch_size, epochs=120):
        tf.random.set_seed(self.seed)
        for network in self.networks:
            network.train(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def predict(self, x_test):
        predictions = []
        for network in self.networks:
            prediction = network(x_test)
            predictions.append(prediction)
        return predictions


class MapNetwork:
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
        self.network = build_model(
            self.input_shape, self.layer_units, self.layer_activations
        )

        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="mean_squared_error",
        )

    def train(self, x_train, y_train, batch_size, epochs=120):
        tf.random.set_seed(self.seed)
        self.network.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x_test):
        prediction = self.network(x_test)
        return prediction

    def __call__(self, x_test):
        return self.predict(x_test)
