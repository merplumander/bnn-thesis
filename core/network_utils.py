from collections.abc import Iterable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def _linspace_network_indices(n_networks, n_predictions):
    """
    Function to compute the indices of the networks or samples to use for prediction
    (since for HMC you want the network samples used for prediction to be as far apart
    as possible).
    think: tf.linspace(0, self.n_networks - 1, n_predictions).
    The rest is just annoying tensorflow issues
    """
    indices = tf.cast(
        tf.math.ceil(
            tf.linspace(
                0, tf.constant(n_networks - 1, dtype=tf.float32), n_predictions,
            )
        ),
        tf.int32,
    )
    return indices


def regularization_lambda_to_prior_scale(regularization_lambda):
    """
    Takes the regularization parameter lambda of L2 regularization and outputs the scale
    of the corresponding normal distribution.
    """
    scale = np.sqrt(1 / (2 * regularization_lambda))
    return scale


def prior_scale_to_regularization_lambda(prior_scale):
    """
    Takes the scale of the prior normal distribution and outputs the corresponding L2
    regularization lambda.
    """
    regularization_lambda = 1 / (2 * prior_scale ** 2)
    return regularization_lambda


def transform_unconstrained_scale(unconstrained_scale, stability_addition=1e-6):
    """
    Density networks output an unconstrained standard deviation (a.k.a. scale).
    This function transforms the unconstrained standard deviation to the actual value.
    """
    return stability_addition + tf.math.softplus(0.05 * unconstrained_scale)


def activation_strings_to_activation_functions(layer_activations):
    """
    The list of layer activations is usually a list of strings.
    built_scratch_model needs tensorflow activation functions and not strings
    (as opposed to keras, which can deal with the strings). This function transforms the
    list of strings to a list of activation functions.
    """
    layer_activation_functions = []
    for activation_str in layer_activations:
        if activation_str == "relu":
            activation_function = tf.nn.relu
        elif activation_str == "tanh":
            activation_function = tf.nn.tanh
        elif activation_str == "linear":
            activation_function = tf.identity
        else:
            raise NotImplementedError(
                f"Activation function {activation_str} not yet implemented"
            )
        layer_activation_functions.append(activation_function)
    return layer_activation_functions


def dense_layer(inputs, w, b, activation=tf.identity):
    return activation(tf.matmul(inputs, w) + b)


def build_scratch_density_model(weights_list, layer_activation_functions):
    """
    Building a tensorflow model from scratch (i.e. without keras and tf.layers).
    This is needed for HMC, since sampling cannot deal with keras objects
    for some reason. Probably this way it is also faster, since keras layers have a lot
    of additional functionality.

    Args:
        weights_list (list):      Flat list of weights and biases of the network
                                  (must have an even number of elements)
        layer_activations (list): List of tensorflow activation functions
                                  (must have half as many elements as weights_list)
    """

    def model(X):
        for w, b, activation in zip(
            weights_list[::2], weights_list[1::2], layer_activation_functions
        ):
            X = dense_layer(X, w, b, activation)
        return tfd.Normal(
            loc=X[..., :1], scale=transform_unconstrained_scale(X[..., 1:])
        )

    return model


def build_scratch_model(weights_list, noise_std, layer_activation_functions):
    """
    Building a tensorflow model from scratch (i.e. without keras and tf.layers).
    This is needed for HMC, since sampling cannot deal with keras objects
    for some reason. Probably this way it is also faster, since keras layers have a lot
    of additional functionality.

    Args:
        weights_list (list):      Flat list of weights and biases of the network
                                  (must have an even number of elements)
        noise_std (float):        Noise standard deviation
        layer_activations (list): List of tensorflow activation functions
                                  (must have half as many elements as weights_list)
    """

    def model(X):
        for w, b, activation in zip(
            weights_list[::2], weights_list[1::2], layer_activation_functions
        ):
            X = dense_layer(X, w, b, activation)
        return tfd.Normal(loc=X[..., :1], scale=noise_std)

    return model


def build_keras_model(
    input_shape=[1],
    layer_units=[200, 100, 1],
    layer_activations=["relu", "relu", "linear"],
    l2_weight_lambda=None,
    l2_bias_lambda=None,
    names=None,
):
    """Building an uncompiled keras tensorflow model with the architecture given"""
    if names is None:
        names = [None] * len(layer_units)
    if not isinstance(l2_weight_lambda, Iterable):
        if l2_weight_lambda is None:
            kernel_regularizers = [None] * len(layer_units)
        else:
            kernel_regularizers = [tf.keras.regularizers.l2(l2_weight_lambda)] * len(
                layer_units
            )
    if not isinstance(l2_bias_lambda, Iterable):
        if l2_bias_lambda is None:
            bias_regularizers = [None] * len(layer_units)
        else:
            bias_regularizers = [tf.keras.regularizers.l2(l2_bias_lambda)] * len(
                layer_units
            )

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(
            layer_units[0],
            input_shape=input_shape,
            activation=layer_activations[0],
            name=names[0],
            kernel_regularizer=kernel_regularizers[0],
            bias_regularizer=bias_regularizers[0],
        )
    )
    for units, activation, name, kernel_regularizer, bias_regularizer in zip(
        layer_units[1:],
        layer_activations[1:],
        names[1:],
        kernel_regularizers[1:],
        bias_regularizers[1:],
    ):
        model.add(
            tf.keras.layers.Dense(
                units,
                activation=activation,
                name=name,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
        )
    return model


# @tf.keras.utils.register_keras_serializable(package="Custom", name="l2")
# class PriorRegularizer(tf.keras.regularizers.Regularizer):
#     def __init__(self, prior_distribution):
#         self.prior_distribution = prior_distribution
#
#     def __call__(self, weight_matrix):
#         return tf.math.reduce_sum(self.prior_distribution.log_prob(weight_matrix))
#
#     def get_config(self):
#         return {"prior_distribution": self.prior_distribution}


def train_ml_model(
    train_x, train_y, layer_units, layer_activations, batchsize_train, seed=0
):
    print("This function is deprecated!")
    tf.random.set_seed(seed)
    model = build_keras_model(
        layer_units=layer_units, layer_activations=layer_activations
    )
    # return model
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule), loss="mean_squared_error"
    )
    model.fit(train_x, train_y, epochs=120, batch_size=batchsize_train)
    return model
