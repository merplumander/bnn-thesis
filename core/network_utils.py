import copy
from collections.abc import Iterable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


# @tf.keras.utils.register_keras_serializable()
class NormalizationFix(tf.keras.layers.experimental.preprocessing.Normalization):
    """
    tf.keras.layers.experimental.preprocessing.Normalization cannot deal with training
    data where one feature is constant. It will assign a variance of 0, resulting in
    NaNs being created when data is passed through the layer (since it divides the data
    by 0 then). This is an adhoc fix to circumvent that problem.
    """

    def adapt(self, data, reset_state=True):
        super().adapt(data, reset_state)
        # akward tf way of saying self.variance[self.variance == 0] = 1
        bmask = tf.equal(self.variance, 0.0)
        imask = tf.where(
            bmask, tf.ones_like(self.variance), tf.zeros_like(self.variance)
        )

        self.variance.assign(self.variance + imask)

    def get_config(self):
        config = super().get_config()

        # config.update({"variance": self.variance})
        return config


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

    def __init__(self, initial_value=0.0, **kwargs):
        super().__init__(**kwargs)
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

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({"sigma": self.sigma})
    #     #config.update({"units": self.units})
    #     return config


def map_to_hmc_weights(weights):
    """
    Takes the output of MapDensityEnsemble .get_weights() and transforms it to weights
    that are usable as multi-chain HMC starting points.
    """
    _weights = copy.deepcopy(weights)
    for network_weights in _weights:
        for i in np.arange(len(network_weights))[1::2]:
            # i only indexes biases
            network_weights[i] = tf.expand_dims(network_weights[i], axis=0)
        network_weights[-1] = tf.expand_dims(network_weights[-1], axis=0)
    hmc_weights = []
    for i in range(len(_weights[0])):
        hmc_weights.append(tf.stack((list(map(lambda x: x[i], _weights))), axis=0))
    return hmc_weights


def hmc_to_map_weights(weights):
    """
    Takes hmc samples, e.g. HMCDensityNetwork .sample_prior_state(n_samples=2) or
    tf.nest.map_structure(lambda x: x[i], hmc_net.samples) and transforms it to a list
    of weights usable by a MapDensityEnsemble.
    """
    _weights = copy.deepcopy(weights)
    for i in np.arange(len(_weights))[1::2]:
        _weights[i] = tf.squeeze(_weights[i], axis=1)
    _weights[-1] = tf.squeeze(_weights[-1], axis=1)
    n_networks = _weights[0].shape[0]
    map_weights = []
    for i_networks in range(n_networks):
        map_weights.append(
            [
                _weights[i_weights][i_networks].numpy()
                for i_weights in range(len(_weights))
            ]
        )

    return map_weights


def make_independent_gaussian_network_prior(
    input_shape, layer_units, loc=0.0, scale=1.0
):
    """
    Takes the loc and scale of a Gaussian distribution and produces a list of
    independent multi-D Gaussians that define a prior over the whole network.
    Intended to be used for HMCDensityNetwork.
    """
    full_shape = input_shape + layer_units
    network_prior = []
    for i in range(len(full_shape) - 1):
        weight_prior = tfd.Independent(
            tfd.Normal(
                loc=loc, scale=tf.fill((full_shape[i], full_shape[i + 1]), scale)
            ),
            reinterpreted_batch_ndims=2,
        )
        bias_prior = tfd.Independent(
            tfd.Normal(loc=loc, scale=tf.fill((1, full_shape[i + 1]), scale)),
            reinterpreted_batch_ndims=2,
        )
        network_prior.append(weight_prior)
        network_prior.append(bias_prior)
    return network_prior


def batch_repeat_matrix(m, repeats):
    """
    Adds leading dimension to a matrix and repeats matrix along this dimension.
    If m is a 1D vector, then an additional empty middle dimension is added.
    Example:
        m.shape: (a, b)
            Returns tensor with shape (repeats, a, b)
        m.shape: (a)
            Returns tensor with shape (repeats, 1, a)
    This is especially useful for getting the weights of a MapDensityNetwork to be
    repeated such that multiple HMC chains can be run starting there.
    """
    m = tf.expand_dims(m, axis=0)
    m = tf.repeat(m, repeats=repeats, axis=0)
    if len(m.shape) == 2:
        m = tf.expand_dims(m, axis=1)
    return m


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
            tf.linspace(0, tf.constant(n_networks - 1, dtype=tf.int32), n_predictions)
        ),
        tf.int32,
    )
    return indices


def regularization_lambda_to_prior_scale(regularization_lambda, n_train):
    """
    Takes the regularization parameter lambda of L2 regularization and outputs the scale
    of the corresponding normal distribution.
    """
    scale = np.sqrt(1 / (2 * regularization_lambda * n_train))  # * n_train
    return scale


def prior_scale_to_regularization_lambda(prior_scale, n_train):
    """
    Takes the scale of the prior normal distribution and outputs the corresponding L2
    regularization lambda.
    """
    regularization_lambda = (1 / (2 * prior_scale ** 2)) / n_train
    return regularization_lambda


def transform_unconstrained_scale(
    unconstrained_scale, factor=0.05, stability_addition=1e-6
):
    """
    Density networks output an unconstrained standard deviation (a.k.a. scale).
    This function transforms the unconstrained standard deviation to the actual value.
    """
    return stability_addition + tf.math.softplus(factor * unconstrained_scale)


def backtransform_constrained_scale(
    constrained_scale, factor=0.05, stability_addition=1e-6
):
    """
    Backtransform the actual scale (i.e constrained scale) to the unconstrained space.
    """
    unconstrained = tfp.math.softplus_inverse(constrained_scale)
    unconstrained /= factor
    unconstrained -= stability_addition
    return unconstrained


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


def build_scratch_model(weights_list, noise_std, layer_activation_functions):
    """
    Deprecated. Only to be used with deprecated HMCNetwork.
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
        # print("output", X[..., :1].shape)
        # print("noise", noise_std.shape)
        return tfd.Normal(loc=X[..., :1], scale=noise_std)

    return model


def build_scratch_density_model(
    weights_list,
    layer_activation_functions,
    transform_unconstrained_scale_factor,
    unconstrained_noise_scale=None,
):
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
        transform_unconstrained_scale_factor (float): Factor for the scale
                                                      transformation
        unconstrained_noise_scale (float, optional) : Unconstrained noise standard
                                                      deviation. If not passed it is
                                                      assumed that the network has two
                                                      outputs and the second one models
                                                      the unconstrained_noise_scale.
    """

    def model(X):
        """
        X must be a two dimensional numpy array or tensorflow tensor with shape
        (n_samples, n_features).
        """
        for w, b, activation in zip(
            weights_list[::2], weights_list[1::2], layer_activation_functions
        ):
            X = dense_layer(X, w, b, activation)

        if unconstrained_noise_scale is None:
            noise_scale = transform_unconstrained_scale(
                X[..., 1:], factor=transform_unconstrained_scale_factor
            )
        else:
            noise_scale = transform_unconstrained_scale(
                unconstrained_noise_scale, factor=transform_unconstrained_scale_factor
            )
        return tfd.Independent(
            tfd.Normal(loc=X[..., :1], scale=noise_scale), reinterpreted_batch_ndims=2
        )

    return model


def build_keras_model(
    input_shape=[1],
    layer_units=[200, 100, 1],
    layer_activations=["relu", "relu", "linear"],
    weight_prior=None,
    bias_prior=None,
    n_train=None,
    l2_weight_lambda=None,
    l2_bias_lambda=None,
    normalization_layer=False,
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
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    if normalization_layer:
        model.add(NormalizationFix(input_shape=input_shape, name="normalization"))

    # The code below for creating the layers is really stupid. But it is necessary,
    # because the following code does not work because of tensorflow bug:
    # https://github.com/tensorflow/tensorflow/issues/44590

    # for i, (units, activation, name, kernel_regularizer, bias_regularizer) in enumerate(zip(
    #     layer_units, layer_activations, names, kernel_regularizers, bias_regularizers
    # )):
    #     layer = tf.keras.layers.Dense(
    #         units,
    #         activation=activation,
    #         name=name,
    #         kernel_regularizer=kernel_regularizer,
    #         bias_regularizer=bias_regularizer,
    #     )
    #     # if weight_prior:
    #     #     model.add_loss(lambda :-tf.reduce_sum(weight_prior.log_prob(layer.kernel)))
    #     # if bias_prior:
    #     #     layer.add_loss(lambda :-tf.reduce_sum(bias_prior.log_prob(layer.bias)))

    #     model.add(layer)
    layer0 = tf.keras.layers.Dense(
        layer_units[0],
        activation=layer_activations[0],
        name=names[0],
        kernel_regularizer=kernel_regularizers[0],
        bias_regularizer=bias_regularizers[0],
    )
    if weight_prior:
        layer0.add_loss(
            lambda: -tf.reduce_sum(weight_prior.log_prob(layer0.kernel)) / n_train
        )
    if bias_prior:
        layer0.add_loss(
            lambda: -tf.reduce_sum(bias_prior.log_prob(layer0.bias)) / n_train
        )
    model.add(layer0)
    if len(layer_units) > 1:
        layer1 = tf.keras.layers.Dense(
            layer_units[1],
            activation=layer_activations[1],
            name=names[1],
            kernel_regularizer=kernel_regularizers[1],
            bias_regularizer=bias_regularizers[1],
        )
        if weight_prior:
            layer1.add_loss(
                lambda: -tf.reduce_sum(weight_prior.log_prob(layer1.kernel)) / n_train
            )
        if bias_prior:
            layer1.add_loss(
                lambda: -tf.reduce_sum(bias_prior.log_prob(layer1.bias)) / n_train
            )
        model.add(layer1)
    if len(layer_units) > 2:
        layer2 = tf.keras.layers.Dense(
            layer_units[2],
            activation=layer_activations[2],
            name=names[2],
            kernel_regularizer=kernel_regularizers[2],
            bias_regularizer=bias_regularizers[2],
        )
        if weight_prior:
            layer2.add_loss(
                lambda: -tf.reduce_sum(weight_prior.log_prob(layer2.kernel)) / n_train
            )
        if bias_prior:
            layer2.add_loss(
                lambda: -tf.reduce_sum(bias_prior.log_prob(layer2.bias)) / n_train
            )
        model.add(layer2)
    if len(layer_units) > 3:
        layer3 = tf.keras.layers.Dense(
            layer_units[3],
            activation=layer_activations[3],
            name=names[3],
            kernel_regularizer=kernel_regularizers[3],
            bias_regularizer=bias_regularizers[3],
        )
        if weight_prior:
            layer3.add_loss(
                lambda: -tf.reduce_sum(weight_prior.log_prob(layer3.kernel)) / n_train
            )
        if bias_prior:
            layer3.add_loss(
                lambda: -tf.reduce_sum(bias_prior.log_prob(layer3.bias)) / n_train
            )
        model.add(layer3)
    if len(layer_units) > 4:
        layer4 = tf.keras.layers.Dense(
            layer_units[4],
            activation=layer_activations[4],
            name=names[4],
            kernel_regularizer=kernel_regularizers[4],
            bias_regularizer=bias_regularizers[4],
        )
        if weight_prior:
            layer4.add_loss(
                lambda: -tf.reduce_sum(weight_prior.log_prob(layer4.kernel)) / n_train
            )
        if bias_prior:
            layer4.add_loss(
                lambda: -tf.reduce_sum(bias_prior.log_prob(layer4.bias)) / n_train
            )
        model.add(layer4)
    if len(layer_units) > 5:
        raise ValueError("Larger then depth five networks are currently not supported.")

    return model


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
