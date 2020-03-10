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
        self.seed = seed
        tf.random.set_seed(self.seed)
        self.networks = [
            build_model(self.input_shape, self.layer_units, self.layer_activations)
            for i in range(self.n_networks)
        ]

        initial_learning_rate = 0.01
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
        )
        for network in self.networks:
            network.compile(
                optimizer=tf.keras.optimizers.Adam(lr_schedule),
                loss="mean_squared_error",
            )

    def train(self, x_train, y_train, batchsize_train, epochs=120):
        tf.random.set_seed(self.seed)
        for network in self.networks:
            network.fit(x_train, y_train, epochs=epochs, batch_size=batchsize_train)

    def predict(self, x_test):
        predictions = []
        for network in self.networks:
            prediction = network(x_test)
            predictions.append(prediction)
        return predictions
