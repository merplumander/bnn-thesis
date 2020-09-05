import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..network_utils import (
    AddSigmaLayer,
    NormalizationFix,
    transform_unconstrained_scale,
)
from ..preprocessing import StandardizePreprocessor

tfd = tfp.distributions


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1e-5))
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(
                        loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])
                    ),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


def prior_fn_factory(scale_identity_multiplier=1):
    def prior(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        mvn = tfd.MultivariateNormalDiag(
            loc=[0] * n, scale_identity_multiplier=[scale_identity_multiplier]
        )
        return lambda input: mvn

    return prior


class ValidationModel(tf.keras.Model):
    """
    This class inherits from keras.Model and only changes the test_step function and
    with that the behavior during evaluation. With the standard keras.Model
    model.evaluate() will compute the loss from the loss function and add to it the
    regularization losses. This might sometimes be intended behaviour, but for early
    stopping on a validation set, I would argue that you want to ignore regularization
    losses. This model does exactly that. It makes sure that model.evaluate() ignores
    regularization losses and therefore early stopping via
    tf.keras.callbacks.EarlyStopping also ignores regularization losses.
    """

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred)  # , regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def build_variational_keras_model(
    input_shape=[1],
    layer_units=[200, 100, 1],
    layer_activations=["relu", "relu", "linear"],
    initial_unconstrained_scale=None,
    transform_unconstrained_scale_factor=0.05,
    normalization_layer=False,
    prior_scale_identity_multiplier=1,
    kl_weight=None,
    names=None,
):
    """
    Building an uncompiled keras tensorflow variational model with the architecture
    given.
    """
    if names is None:
        names = [None] * len(layer_units)

    # model = tf.keras.Sequential()
    input = tf.keras.Input(shape=input_shape)
    if normalization_layer:
        x = NormalizationFix(input_shape=input_shape, name="normalization")(input)

    prior = prior_fn_factory(prior_scale_identity_multiplier)

    for i, units, activation, name in zip(
        range(len(layer_units)), layer_units, layer_activations, names,
    ):
        layer = tfp.layers.DenseVariational(
            units,
            activation=activation,
            name=name,
            make_posterior_fn=posterior_mean_field,
            make_prior_fn=prior,
            kl_weight=kl_weight,
            use_bias=True,
        )
        if i == 0 and not normalization_layer:
            x = layer(input)
        else:
            x = layer(x)
    if initial_unconstrained_scale is not None:
        x = AddSigmaLayer(initial_unconstrained_scale)(x)
    x = tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(
            loc=t[..., :1],
            scale=transform_unconstrained_scale(
                t[..., 1:], transform_unconstrained_scale_factor
            ),
        )
    )(x)
    model = ValidationModel(inputs=input, outputs=x)
    return model


class VariationalDensityNetwork:
    def __init__(
        self,
        input_shape=[1],
        layer_units=[50, 1],
        layer_activations=["relu", "linear"],
        initial_unconstrained_scale=None,  # unconstrained noise standard deviation. When None, the noise std is model by a second network output. When float, the noise std is homoscedastic.
        transform_unconstrained_scale_factor=0.05,  # factor to be used in the calculation of the actual noise std.
        kl_weight=None,  # should be 1 / n_train
        prior_scale_identity_multiplier=1,
        preprocess_x=False,
        preprocess_y=False,
        learning_rate=0.1,
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
        self.kl_weight = kl_weight
        self.prior_scale_identity_multiplier = prior_scale_identity_multiplier
        self.preprocess_x = preprocess_x
        self.preprocess_y = preprocess_y
        self.learning_rate = learning_rate
        self.names = names
        self.seed = seed

        tf.random.set_seed(self.seed)

        if self.preprocess_y:
            self.y_preprocessor = StandardizePreprocessor()

        #         self.network = tf.keras.Sequential()
        #         self.network.add(tf.keras.layers.InputLayer(input_shape=[1]))

        #         for units, activation in zip(self.layer_units, self.layer_activations):
        #             self.network.add(
        #                 tfp.layers.DenseVariational(
        #                     input_shape=[1],
        #                     units=units,
        #                     make_posterior_fn=posterior_mean_field,
        #                     make_prior_fn=prior,
        #                     use_bias=True,
        #                     kl_weight=1 / x_train.shape[0],
        #                     activation=activation,
        #                 )
        #             )

        self.network = build_variational_keras_model(
            self.input_shape,
            self.layer_units,
            self.layer_activations,
            initial_unconstrained_scale=self.initial_unconstrained_scale,
            transform_unconstrained_scale_factor=self.transform_unconstrained_scale_factor,
            prior_scale_identity_multiplier=self.prior_scale_identity_multiplier,
            normalization_layer=self.preprocess_x,
            kl_weight=self.kl_weight,
            names=self.names,
        )

        # if self.initial_unconstrained_scale is not None:
        #     self.network.add(AddSigmaLayer(self.initial_unconstrained_scale))

        # self.network.add(
        #     tfp.layers.DistributionLambda(
        #         lambda t: tfd.Normal(
        #             loc=t[..., :1],
        #             scale=transform_unconstrained_scale(
        #                 t[..., 1:], transform_unconstrained_scale_factor
        #             ),
        #         )
        #     )
        # )

        self.network.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss=VariationalDensityNetwork.negative_log_likelihood,
        )

    def negative_log_likelihood(y, p_y):
        return -p_y.log_prob(y)

    def fit_preprocessing(self, y_train):
        if self.preprocess_y:
            self.y_preprocessor.fit(y_train)

    def fit(
        self,
        x_train,
        y_train,
        batch_size=1,
        epochs=1,
        early_stop_callback=None,
        validation_split=0.0,
        validation_data=None,
        verbose=0,
    ):
        if self.preprocess_x:
            self.network.get_layer("normalization").adapt(x_train, reset_state=True)
        self.fit_preprocessing(y_train)
        if self.preprocess_y:
            y_train = self.y_preprocessor.transform(y_train)
            if validation_data:
                y_validation = self.y_preprocessor.transform(validation_data[1])
                validation_data = (validation_data[0], y_validation)
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

    def predict_list_of_gaussians(self, x, n_predictions=10):
        predictions = []
        for i in range(n_predictions):
            prediction = self.network(x)
            if self.preprocess_y:
                mean = prediction.mean()
                std = prediction.stddev()
                mean = self.y_preprocessor.inverse_transform(mean)
                if self.y_preprocessor.std is not None:
                    std *= self.y_preprocessor.std
                prediction = tfd.Normal(mean, std)
            predictions.append(prediction)
        return predictions

    def predict_mixture_of_gaussians(self, x, n_predictions=10):
        gaussians = self.predict_list_of_gaussians(x, n_predictions)
        cat_probs = tf.ones((x.shape[0],) + (1, len(gaussians))) / len(gaussians)
        return tfd.Mixture(cat=tfd.Categorical(probs=cat_probs), components=gaussians)

    def predict(self, x, n_predictions=10):
        return self.predict_mixture_of_gaussians(x, n_predictions)

    def get_weights(self):
        return self.network.get_weights()
