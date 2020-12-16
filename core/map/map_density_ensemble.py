import pickle
from collections import Iterable
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp

from ..network_utils import (
    AddSigmaLayer,
    _linspace_network_indices,
    build_keras_model,
    regularization_lambda_to_prior_scale,
    transform_unconstrained_scale,
)
from ..preprocessing import StandardizePreprocessor

tfd = tfp.distributions


class MapDensityEnsemble:
    def __init__(
        self,
        n_networks=5,
        input_shape=[1],
        layer_units=[200, 100, 2],
        layer_activations=["relu", "relu", "linear"],
        initial_unconstrained_scale=None,  # None, float or Iterable
        transform_unconstrained_scale_factor=0.05,
        weight_prior=None,
        bias_prior=None,
        noise_scale_prior=None,
        n_train=None,
        l2_weight_lambda=None,
        l2_bias_lambda=None,
        preprocess_x=False,
        preprocess_y=False,
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        names=None,
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
        self._n_networks = n_networks
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.initial_unconstrained_scale = initial_unconstrained_scale
        if not isinstance(initial_unconstrained_scale, Iterable):
            self.initial_unconstrained_scale = [
                initial_unconstrained_scale
            ] * self._n_networks
        self.transform_unconstrained_scale_factor = transform_unconstrained_scale_factor
        self.weight_prior = weight_prior
        self.bias_prior = bias_prior
        self.noise_scale_prior = noise_scale_prior
        self.n_train = n_train
        self.l2_weight_lambda = l2_weight_lambda
        self.l2_bias_lambda = l2_bias_lambda
        self.preprocess_x = preprocess_x
        self.preprocess_y = preprocess_y
        self.learning_rate = learning_rate
        if not isinstance(self.l2_weight_lambda, Iterable):
            self.l2_weight_lambda = [self.l2_weight_lambda] * self._n_networks
        if not isinstance(self.l2_bias_lambda, Iterable):
            self.l2_bias_lambda = [self.l2_bias_lambda] * self._n_networks
        if not isinstance(seed, Iterable):
            delta = (
                1000000  # earliest seed that has almost the same trained models as 0.
            )
            seed = [i for i in range(seed, seed + self._n_networks * delta, delta)]
        self.names = names
        self.seed = seed

        self.total_epochs = None

        tf.random.set_seed(self.seed[0])

        self.networks = [
            MapDensityNetwork(
                self.input_shape,
                self.layer_units,
                self.layer_activations,
                initial_unconstrained_scale=initial_unconstrained_scale,
                transform_unconstrained_scale_factor=self.transform_unconstrained_scale_factor,
                weight_prior=self.weight_prior,
                bias_prior=self.bias_prior,
                noise_scale_prior=self.noise_scale_prior,
                n_train=self.n_train,
                l2_weight_lambda=l2_weight_lambda,
                l2_bias_lambda=l2_bias_lambda,
                preprocess_x=self.preprocess_x,
                preprocess_y=self.preprocess_y,
                learning_rate=self.learning_rate,
                names=self.names,
                seed=seed,
            )
            for initial_unconstrained_scale, l2_weight_lambda, l2_bias_lambda, seed in zip(
                self.initial_unconstrained_scale,
                self.l2_weight_lambda,
                self.l2_bias_lambda,
                self.seed,
            )
        ]

    @property
    def n_networks(self):
        return len(self.networks)

    def fit_additional_memebers(
        self,
        x_train,
        y_train,
        batch_size,
        n_new_members=5,
        epochs=200,
        early_stop_callback=None,
        validation_split=0.0,
        validation_data=None,
        verbose=1,
    ):
        delta = 100
        seeds = [
            i
            for i in range(
                self.seed[-1] + delta,
                self.seed[-1] + delta + n_new_members * delta,
                delta,
            )
        ]
        new_networks = [
            MapDensityNetwork(
                self.input_shape,
                self.layer_units,
                self.layer_activations,
                initial_unconstrained_scale=self.initial_unconstrained_scale[0],
                transform_unconstrained_scale_factor=self.transform_unconstrained_scale_factor,
                weight_prior=self.weight_prior,
                bias_prior=self.bias_prior,
                noise_scale_prior=self.noise_scale_prior,
                n_train=self.n_train,
                l2_weight_lambda=self.l2_weight_lambda[0],
                l2_bias_lambda=self.l2_bias_lambda[0],
                preprocess_x=self.preprocess_x,
                preprocess_y=self.preprocess_y,
                learning_rate=self.learning_rate,
                names=self.names,
                seed=seed,
            )
            for seed in seeds
        ]
        self._n_networks += n_new_members
        self.fit_member_networks(
            new_networks,
            x_train,
            y_train,
            batch_size,
            epochs=epochs,
            early_stop_callback=early_stop_callback,
            validation_split=validation_split,
            validation_data=validation_data,
            verbose=verbose,
        )
        self.networks += new_networks
        return self

    def fit_member_networks(
        self,
        member_networks,
        x_train,
        y_train,
        batch_size,
        epochs=200,
        early_stop_callback=None,
        validation_split=0.0,
        validation_data=None,
        verbose=1,
    ):
        tf.random.set_seed(self.seed[0])
        if not isinstance(epochs, Iterable):
            epochs = [epochs] * self.n_networks
        if self.total_epochs is None:
            self.total_epochs = []
        for network, epoch in zip(member_networks, epochs):
            network.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epoch,
                early_stop_callback=early_stop_callback,
                validation_split=validation_split,
                validation_data=validation_data,
                verbose=verbose,
            )
            self.total_epochs.append(network.total_epochs)
        return self

    def fit(
        self,
        x_train,
        y_train,
        batch_size,
        epochs=200,
        early_stop_callback=None,
        validation_split=0.0,
        validation_data=None,
        verbose=1,
    ):
        self.fit_member_networks(
            self.networks,
            x_train,
            y_train,
            batch_size,
            epochs=epochs,
            early_stop_callback=early_stop_callback,
            validation_split=validation_split,
            validation_data=validation_data,
            verbose=verbose,
        )

    def rmse(self, x, y):
        mean_prediction = self.predict(x).mean()
        return tf.keras.metrics.RootMeanSquaredError()(mean_prediction, y)

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
        cat_probs = tf.ones((x_test.shape[0],) + (1, self.n_networks)) / len(
            self.networks
        )
        return tfd.Mixture(cat=tfd.Categorical(probs=cat_probs), components=gaussians)

    def predict_moment_matched_gaussian(self, x_test):
        predictive_mixture = self.predict_mixture_of_gaussians(x_test)
        std = predictive_mixture.stddev()
        # When the std of the mixture is below ~0.0001 it just returns 0. (although much
        # smaller numbers are representable by float32). I think that is due to the way
        # the stddev is computed (seems unstable for such small numbers). When that
        # happens, just put the stddev to 0.0001.
        condition = tf.math.logical_or(std <= 0.0, tf.math.is_nan(std))
        std = tf.where(condition, 0.0001, std)
        return tfd.Normal(loc=predictive_mixture.mean(), scale=std)

    def predict(self, x_test):
        return self.predict_mixture_of_gaussians(x_test)

    def predict_unnormalized_space(self, x_test, mean, scale):
        """
        Predict and transform the predictive distribution into the unnormalized space

        Args:
            mean (float):  Mean of the unnormalized training data. I.e. mean of the
                           z-transformation that transformed the data into the normalized
                           space.
            scale (float): Scale of the unnormalized training data.
        """
        gaussians = self.predict_list_of_gaussians(x_test)
        unnormalized_gaussians = []
        for prediction in gaussians:
            predictive_mean = prediction.mean()
            predictive_std = prediction.stddev()
            predictive_mean = (predictive_mean * scale) + mean
            predictive_std = predictive_std * scale
            unnormalized_gaussians.append(tfd.Normal(predictive_mean, predictive_std))
        cat_probs = tf.ones((x_test.shape[0],) + (1, self.n_networks)) / len(
            self.networks
        )
        return tfd.Mixture(
            cat=tfd.Categorical(probs=cat_probs), components=unnormalized_gaussians
        )

    def get_weights(self):
        networks_weights_list = []
        for network in self.networks:
            networks_weights_list.append(network.get_weights())
        return networks_weights_list

    def set_weights(self, networks_weights_list):
        for network, network_weights in zip(self.networks, networks_weights_list):
            network.set_weights(network_weights)

    def save(self, save_path):
        save_path = Path(save_path)
        for i, net in enumerate(self.networks):
            net.save(save_path.joinpath(f"_{i}"))
        networks = self.networks
        self.networks = None
        with open(save_path.joinpath("pickle_class"), "wb") as f:
            pickle.dump(self, f, -1)
        self.networks = networks


class MapDensityNetwork:
    def __init__(
        self,
        input_shape=[1],
        layer_units=[200, 100, 2],
        layer_activations=["relu", "relu", "linear"],
        initial_unconstrained_scale=None,  # unconstrained noise standard deviation. When None, the noise std is model by a second network output. When float, the noise std is homoscedastic.
        transform_unconstrained_scale_factor=0.05,  # factor to be used in the calculation of the actual noise std.
        weight_prior=None,
        bias_prior=None,
        noise_scale_prior=None,
        n_train=None,
        l2_weight_lambda=None,  # float or list of floats
        l2_bias_lambda=None,
        preprocess_x=False,
        preprocess_y=False,
        learning_rate=0.01,  # can be float or an instance of tf.keras.optimizers.schedules
        names=None,
        seed=0,
    ):
        if initial_unconstrained_scale is not None:
            assert layer_units[-1] == 1
        else:
            assert layer_units[-1] == 2
        # This is a very typical mistake that throws a quite cryptically error message.
        # So it is better to catch it with this assertion.
        if (
            weight_prior is not None
            or bias_prior is not None
            or noise_scale_prior is not None
        ):
            assert n_train is not None
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.initial_unconstrained_scale = initial_unconstrained_scale
        self.transform_unconstrained_scale_factor = transform_unconstrained_scale_factor
        self.weight_prior = weight_prior
        self.bias_prior = bias_prior
        self.noise_scale_prior = noise_scale_prior
        self.n_train = n_train
        self.l2_weight_lambda = l2_weight_lambda
        self.l2_bias_lambda = l2_bias_lambda
        self.preprocess_x = preprocess_x
        self.preprocess_y = preprocess_y
        self.learning_rate = learning_rate
        self.names = names
        self.seed = seed
        tf.random.set_seed(self.seed)

        if self.preprocess_y:
            self.y_preprocessor = StandardizePreprocessor()

        self.network = build_keras_model(
            self.input_shape,
            self.layer_units,
            self.layer_activations,
            weight_prior=self.weight_prior,
            bias_prior=self.bias_prior,
            n_train=self.n_train,
            l2_weight_lambda=self.l2_weight_lambda,
            l2_bias_lambda=self.l2_bias_lambda,
            normalization_layer=self.preprocess_x,
            names=self.names,
        )
        if self.initial_unconstrained_scale is not None:
            # print("homoscedastic noise")
            layer = AddSigmaLayer(
                self.initial_unconstrained_scale, name="aleatoric_noise"
            )
            if self.noise_scale_prior is not None:
                self.network.add_loss(
                    lambda: -tf.reduce_sum(
                        self.noise_scale_prior.log_prob(
                            transform_unconstrained_scale(
                                layer.sigma, self.transform_unconstrained_scale_factor
                            )
                        )
                    )
                    / self.n_train
                )
            self.network.add(layer)
        self.network.add(
            tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :1],
                    scale=transform_unconstrained_scale(
                        t[..., 1:], self.transform_unconstrained_scale_factor
                    ),
                )
            )
        )

        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=MapDensityNetwork.negative_log_likelihood,
        )

    @property
    def noise_sigma(self):
        if self.initial_unconstrained_scale is not None:
            unconstrained_sigma = self.network.get_layer("aleatoric_noise").sigma[0, 0]
            return transform_unconstrained_scale(
                unconstrained_sigma, self.transform_unconstrained_scale_factor
            )
        else:
            return None  # in this case the noise sigma depends on the input

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

    def fit_preprocessing(self, y_train):
        if self.preprocess_y:
            self.y_preprocessor.fit(y_train)

    def fit(
        self,
        x_train,
        y_train,
        n_train_data=None,
        batch_size=None,
        epochs=1,
        early_stop_callback=None,
        validation_split=0.0,
        validation_data=None,
        verbose=1,
    ):
        if self.preprocess_x:
            self.network.get_layer("normalization").adapt(x_train, reset_state=True)
        self.fit_preprocessing(y_train)
        if self.preprocess_y:
            y_train = self.y_preprocessor.transform(y_train)
            if validation_data:
                y_validation = self.y_preprocessor.transform(validation_data[1])
                validation_data = (validation_data[0], y_validation)
        if n_train_data is None:
            n_train_data = y_train.shape[0]
        if batch_size is None:
            batch_size = n_train_data
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
        if tf.math.reduce_any(tf.math.is_inf(self.history.history["loss"])):
            raise RuntimeError(
                "Loss is infinite, possibly because the prior assigns probability 0 to some values."
            )
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

    def rmse(self, x, y):
        mean_prediction = self.predict(x).mean()
        return tf.keras.metrics.RootMeanSquaredError()(mean_prediction, y)

    def predict(self, x):
        prediction = self.network(x)
        if self.preprocess_y:
            mean = prediction.mean()
            std = prediction.stddev()
            mean = self.y_preprocessor.inverse_transform(mean)
            if self.y_preprocessor.std is not None:
                std *= self.y_preprocessor.std
            prediction = tfd.Normal(mean, std)
        return prediction

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
        weights = self.network.get_weights()
        if self.initial_unconstrained_scale is not None:
            # Then the last element is not a bias vector but the noise sigma
            weights[-1] = weights[-1][0]
        return weights

    def set_weights(self, weights_list):
        if self.initial_unconstrained_scale is not None:
            weights_list[-1] = weights_list[-1].reshape(1, 1)
        self.network.set_weights(weights_list)

    def save(self, save_path):
        self.network.save(save_path)
        network = self.network
        history = self.history
        self.network = None
        self.history = None
        save_path = Path(save_path)
        with open(save_path.joinpath("pickle_class"), "wb") as f:
            pickle.dump(self, f, -1)
        self.network = network
        self.history = history


def map_density_ensemble_from_save_path(save_path):
    save_path = Path(save_path)
    with open(save_path.joinpath("pickle_class"), "rb") as f:
        ensemble = pickle.load(f)
    networks = [
        map_density_network_from_save_path(save_path.joinpath(f"_{i}"))
        for i in range(ensemble._n_networks)
    ]
    ensemble.networks = networks
    return ensemble


def map_density_network_from_save_path(save_path):
    save_path = Path(save_path)
    with open(save_path.joinpath("pickle_class"), "rb") as f:
        net = pickle.load(f)
    net.network = tf.keras.models.load_model(save_path, compile=False)
    net.network.compile(
        optimizer=tf.keras.optimizers.Adam(net.learning_rate),
        loss=MapDensityNetwork.negative_log_likelihood,
    )
    print(
        "When resuming training on this model, check wether the optimizers state is set correctly. Likely it starts from step 0 again. Which might be a problem for LearningRateSchedules."
    )
    return net
