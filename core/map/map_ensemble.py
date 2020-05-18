import tensorflow as tf

from ..network_utils import build_keras_model


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
                learning_rate=self.learning_rate,
                seed=self.seed + i,
            )
            for i in range(self.n_networks)
        ]

    def fit(self, x_train, y_train, batch_size, epochs=120, verbose=1):
        tf.random.set_seed(self.seed)
        for network in self.networks:
            network.fit(
                x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose
            )
        return self

    def predict(self, x_test):
        predictions = []
        for network in self.networks:
            prediction = network(x_test)
            predictions.append(prediction)
        return predictions

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
        names=None,
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        seed=0,
    ):
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.names = names
        self.learning_rate = learning_rate
        self.seed = seed
        tf.random.set_seed(self.seed)
        self.network = build_keras_model(
            self.input_shape, self.layer_units, self.layer_activations, names=self.names
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
        return self

    def predict(self, x_test):
        prediction = self.network(x_test)
        return prediction

    def __call__(self, x_test):
        return self.predict(x_test)

    def get_weights(self):
        return self.network.get_weights()

    def set_weights(self, weights_list):
        self.network.set_weights(weights_list)
