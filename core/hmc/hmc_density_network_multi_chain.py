# with inspiration from https://janosh.io/blog/hmc-bnn
import os
import pickle
from collections import Iterable
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp

from ..map import MapDensityNetwork
from ..network_utils import (
    _linspace_network_indices,
    activation_strings_to_activation_functions,
    backtransform_constrained_scale,
    build_scratch_density_model,
    transform_unconstrained_scale,
)

tfd = tfp.distributions


class MixtureSameFamilySampleFix(tfd.MixtureSameFamily):
    """
    Sampling from MixtureSameFamily is currently really inefficient in terms of
    computation time and memory. See tensorflow probability issue:
    https://github.com/tensorflow/probability/issues/1208 .
    This is a quick and dirty work around fix for that.
    """

    def _sample_n(self, n, seed):
        # only for MixtureSameFamilySampleFix
        import warnings
        from tensorflow_probability.python.distributions import independent
        from tensorflow_probability.python.internal import dtype_util
        from tensorflow_probability.python.internal import prefer_static
        from tensorflow_probability.python.internal import samplers
        from tensorflow_probability.python.internal import tensorshape_util
        from tensorflow_probability.python.util.seed_stream import SeedStream
        from tensorflow_probability.python.util.seed_stream import (
            TENSOR_SEED_MSG_PREFIX,
        )

        components_seed, mix_seed = samplers.split_seed(seed, salt="MixtureSameFamily")
        try:
            seed_stream = SeedStream(seed, salt="MixtureSameFamily")
        except TypeError as e:  # Can happen for Tensor seeds.
            seed_stream = None
            seed_stream_err = e
        try:
            mix_sample = self.mixture_distribution.sample(
                n, seed=mix_seed
            )  # [n, B] or [n]
        except TypeError as e:
            if "Expected int for argument" not in str(
                e
            ) and TENSOR_SEED_MSG_PREFIX not in str(e):
                raise
            if seed_stream is None:
                raise seed_stream_err
            msg = (
                "Falling back to stateful sampling for `mixture_distribution` "
                "{} of type `{}`. Please update to use `tf.random.stateless_*` "
                "RNGs. This fallback may be removed after 20-Aug-2020. ({})"
            )
            warnings.warn(
                msg.format(
                    self.mixture_distribution.name,
                    type(self.mixture_distribution),
                    str(e),
                )
            )
            mix_sample = self.mixture_distribution.sample(
                n, seed=seed_stream()
            )  # [n, B] or [n]
        _seed = int(components_seed[0].numpy())
        ret = tf.stack(
            [
                self.components_distribution[i_component.numpy()].sample(seed=_seed + i)
                for i, i_component in enumerate(mix_sample)
            ],
            axis=0,
        )
        return ret
        # try:
        #   x = self.components_distribution.sample(  # [n, B, k, E]
        #       n, seed=components_seed)
        #   if seed_stream is not None:
        #     seed_stream()  # Advance even if unused.
        # except TypeError as e:
        #   if ('Expected int for argument' not in str(e) and
        #       TENSOR_SEED_MSG_PREFIX not in str(e)):
        #     raise
        #   if seed_stream is None:
        #     raise seed_stream_err
        #   msg = ('Falling back to stateful sampling for `components_distribution` '
        #          '{} of type `{}`. Please update to use `tf.random.stateless_*` '
        #          'RNGs. This fallback may be removed after 20-Aug-2020. {}')
        #   warnings.warn(msg.format(self.components_distribution.name,
        #                            type(self.components_distribution),
        #                            str(e)))
        #   x = self.components_distribution.sample(  # [n, B, k, E]
        #       n, seed=seed_stream())
        #
        # event_shape = None
        # event_ndims = tensorshape_util.rank(self.event_shape)
        # if event_ndims is None:
        #   event_shape = self.components_distribution.event_shape_tensor()
        #   event_ndims = prefer_static.rank_from_shape(event_shape)
        # event_ndims_static = tf.get_static_value(event_ndims)
        #
        # num_components = None
        # if event_ndims_static is not None:
        #   num_components = tf.compat.dimension_value(
        #       x.shape[-1 - event_ndims_static])
        # # We could also check if num_components can be computed statically from
        # # self.mixture_distribution's logits or probs.
        # if num_components is None:
        #   num_components = tf.shape(x)[-1 - event_ndims]
        #
        # # TODO(jvdillon): Consider using tf.gather (by way of index unrolling).
        # npdt = dtype_util.as_numpy_dtype(x.dtype)
        #
        # mask = tf.one_hot(
        #     indices=mix_sample,  # [n, B] or [n]
        #     depth=num_components,
        #     on_value=npdt(1),
        #     off_value=npdt(0))    # [n, B, k] or [n, k]
        #
        # # Pad `mask` to [n, B, k, [1]*e] or [n, [1]*b, k, [1]*e] .
        # batch_ndims = prefer_static.rank(x) - event_ndims - 1
        # mask_batch_ndims = prefer_static.rank(mask) - 1
        # pad_ndims = batch_ndims - mask_batch_ndims
        # mask_shape = prefer_static.shape(mask)
        # mask = tf.reshape(
        #     mask,
        #     shape=prefer_static.concat([
        #         mask_shape[:-1],
        #         prefer_static.ones([pad_ndims], dtype=tf.int32),
        #         mask_shape[-1:],
        #         prefer_static.ones([event_ndims], dtype=tf.int32),
        #     ], axis=0))
        #
        # if x.dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64,
        #                tf.complex64, tf.complex128]:
        #   masked = tf.math.multiply_no_nan(x, mask)
        # else:
        #   masked = x * mask
        # ret = tf.reduce_sum(masked, axis=-1 - event_ndims)  # [n, B, E]
        #
        # if self._reparameterize:
        #   if event_shape is None:
        #     event_shape = self.components_distribution.event_shape_tensor()
        #   ret = self._reparameterize_sample(ret, event_shape=event_shape)
        # return ret


class HMCDensityNetwork:
    def __init__(
        self,
        input_shape=[1],
        layer_units=[20, 10, 1],
        layer_activations=["relu", "relu", "linear"],
        transform_unconstrained_scale_factor=0.5,
        network_prior=None,
        noise_scale_prior=None,
        sampler="hmc",  # choices are "hmc" and "nuts"
        step_size_adapter="dual_averaging",  # choices are "simple" and "dual_averaging"
        num_burnin_steps=1000,
        discard_burnin_samples=False,  # mainly relevant for RAM limitations
        step_size=0.01,
        num_leapfrog_steps=100,  # only relevant for HMC
        max_tree_depth=10,  # only relevant for NUTS
        num_steps_between_results=0,
        seed=0,
    ):
        """
        A Bayesian neural network whose posterior distribution parameters are estimated
        by HMC. Each individual network sample models a function and a homoscedastic or
        heteroscedasticity standard deviation around that function in the form of a
        Normal distribution. Whether the noise is modelled heteroscedastically or
        homoscedastically is controlled by layer_units and noise_scale_prior. The class
        can very efficiently run several MCMC chains in parallel. Often running 128
        chains in parallel is faster than running one chain twice in a row.

        Params:
            network_prior:
                    list of tfp distributions with event shapes fiiting the shapes of
                    the weight and bias vectors.
            noise_scale_prior (optional):
                    tfp distribution that defines a prior for the noise the scale.
                    If this is not provided, layer_units[-1] must equal 2 so that the
                    noise scale is modelled as a second output of the network.
        """
        if noise_scale_prior is not None:
            assert layer_units[-1] == 1
        else:
            assert layer_units[-1] == 2
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.transform_unconstrained_scale_factor = transform_unconstrained_scale_factor
        self.network_prior = network_prior
        self.noise_scale_prior = noise_scale_prior
        self.sampler = sampler
        self._step_size_adapter = step_size_adapter
        self.num_burnin_steps = num_burnin_steps
        self.discard_burnin_samples = discard_burnin_samples
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.max_tree_depth = max_tree_depth
        self.num_steps_between_results = num_steps_between_results
        self.seed = seed

        self.layer_activation_functions = activation_strings_to_activation_functions(
            self.layer_activations
        )

        self.step_size_adapter = {
            "simple": tfp.mcmc.SimpleStepSizeAdaptation,
            "dual_averaging": tfp.mcmc.DualAveragingStepSizeAdaptation,
        }[self._step_size_adapter]

        self._chain = None
        self._trace = None
        self.chain_mask = None
        self.kernel_results = None

    @property
    def n_chains(self):
        return self._chain[0].shape[1]

    @property
    def n_used_chains(self):
        return self.samples[0].shape[1]

    @property
    def n_samples(self):
        return self.samples[0].shape[0]

    def _apply_chain_mask(self, samples):
        if self.chain_mask is not None:
            samples = tf.nest.map_structure(
                lambda x: tf.boolean_mask(x, self.chain_mask, axis=1), samples
            )
        return samples

    @property
    def samples(self):
        if self.discard_burnin_samples:
            _samples = self._chain
        else:
            _samples = tf.nest.map_structure(
                lambda x: x[self.num_burnin_steps :], self._chain
            )
        _samples = self._apply_chain_mask(_samples)
        return _samples

    @property
    def trace(self):
        if self.discard_burnin_samples:
            trace = self._trace
        else:
            trace = tf.nest.map_structure(
                lambda x: x[self.num_burnin_steps :], self._trace
            )
        trace = self._apply_chain_mask(trace)
        return trace

    def mask_chains(self, chain_mask):
        """
        Takes a boolean mask that is used to mask out some chains e.g. because they were
        divergent.

        Args:
            chain_mask (boolean tensor): Boolean tensor with shape (n_chains).
        """
        self.chain_mask = chain_mask

    def thinned_samples(self, thinning):
        return tf.nest.map_structure(lambda x: x[::thinning], self.samples)

    def _split_sample_weights_and_noise_scale(self, samples):
        if self.noise_scale_prior is not None:
            weights = samples[:-1]
            unconstrained_noise_scale = samples[-1]
        else:
            weights = samples
            unconstrained_noise_scale = None
        return weights, unconstrained_noise_scale

    def log_prior(self, weights, unconstrained_noise_scale):
        log_prob = 0
        for w, w_prior in zip(weights, self.network_prior):
            log_prob += w_prior.log_prob(w)

        if unconstrained_noise_scale is not None:
            noise_scale = transform_unconstrained_scale(
                unconstrained_noise_scale,
                factor=self.transform_unconstrained_scale_factor,
            )
            log_prob += tf.reduce_sum(
                self.noise_scale_prior.log_prob(noise_scale), axis=(-1, -2)
            )
        return log_prob

    def log_likelihood(self, weights, unconstrained_noise_scale, x, y):
        model = build_scratch_density_model(
            weights,
            self.layer_activation_functions,
            transform_unconstrained_scale_factor=self.transform_unconstrained_scale_factor,
            unconstrained_noise_scale=unconstrained_noise_scale,
        )
        return model(x).log_prob(y)

    def _target_log_prob_fn_factory(self, x, y):
        def log_posterior(*current_state):
            (
                weights,
                unconstrained_noise_scale,
            ) = self._split_sample_weights_and_noise_scale(current_state)
            log_prob = self.log_prior(weights, unconstrained_noise_scale)
            log_prob += self.log_likelihood(weights, unconstrained_noise_scale, x, y)
            return log_prob

        return log_posterior

    def sample_prior_state(self, n_samples=(), overdisp=1.0, seed=0):
        """
        Draw random samples for weights and biases of a NN according to some
        specified prior distributions. This set of parameter values can serve as a
        starting point for MCMC or gradient descent training.
        """
        prior_state = []
        for w_prior in self.network_prior:
            w_sample = w_prior.sample(n_samples, seed=seed) * overdisp
            prior_state.append(w_sample)
        if self.noise_scale_prior is not None:
            noise_scale = self.noise_scale_prior.sample(n_samples, seed=seed)
            unconstrained_noise_scale = backtransform_constrained_scale(
                noise_scale, factor=self.transform_unconstrained_scale_factor
            )
            prior_state.append(unconstrained_noise_scale)
        return prior_state

    @tf.function  # (experimental_compile=True)
    def _sample_chain(
        self, num_burnin_steps, num_results, current_state, adaptive_kernel
    ):
        if self.sampler == "nuts":
            trace_fn = lambda _, pkr: [
                pkr.inner_results.is_accepted,
                pkr.inner_results.leapfrogs_taken,
            ]
        else:
            trace_fn = lambda _, pkr: [pkr.inner_results.is_accepted]
        chain, trace, final_kernel_results = tfp.mcmc.sample_chain(
            num_results=num_burnin_steps + num_results,
            current_state=current_state,
            previous_kernel_results=self.kernel_results,
            kernel=adaptive_kernel,
            return_final_kernel_results=True,
            trace_fn=trace_fn,
            num_steps_between_results=self.num_steps_between_results,
            seed=self.seed,
        )
        return chain, trace, final_kernel_results

    def fit(self, x_train, y_train, current_state=None, num_results=5000, resume=False):
        if current_state is None and not resume:
            raise ValueError("Must either pass current_state or set resume=True")
        target_log_prob_fn = self._target_log_prob_fn_factory(x_train, y_train)
        num_burnin_steps = 0 if resume else self.num_burnin_steps

        if self.sampler == "nuts":
            kernel = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn,
                max_tree_depth=self.max_tree_depth,
                step_size=self.step_size,
            )
            adaptive_kernel = self.step_size_adapter(
                kernel,
                num_adaptation_steps=int(num_burnin_steps * 0.8),
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                    step_size=new_step_size
                ),
                step_size_getter_fn=lambda pkr: pkr.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
            )
        elif self.sampler == "hmc":
            kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn,
                step_size=self.step_size,
                num_leapfrog_steps=self.num_leapfrog_steps,
            )
            adaptive_kernel = self.step_size_adapter(
                kernel, num_adaptation_steps=int(num_burnin_steps * 0.8)
            )
        if resume:
            current_state = tf.nest.map_structure(lambda x: x[-1], self._chain)
        else:
            self.kernel_results = adaptive_kernel.bootstrap_results(current_state)

        chain, trace, self.kernel_results = self._sample_chain(
            num_burnin_steps=num_burnin_steps,
            num_results=num_results,
            current_state=current_state,
            adaptive_kernel=adaptive_kernel,
        )
        if not resume:
            if self.discard_burnin_samples:
                self._chain = tf.nest.map_structure(
                    lambda x: x[self.num_burnin_steps :], chain
                )
                self._trace = tf.nest.map_structure(
                    lambda x: x[self.num_burnin_steps :], trace
                )
            else:
                self._chain = chain
                self._trace = trace
        else:
            self._chain = tf.nest.map_structure(
                lambda *parts: tf.concat(parts, axis=0), *[self._chain, chain]
            )
            self._trace = tf.nest.map_structure(
                lambda *parts: tf.concat(parts, axis=0), *[self._trace, trace]
            )
        return self

    def potential_scale_reduction(self):
        return tfp.mcmc.potential_scale_reduction(self.samples)

    def effective_sample_size(self, thinning=1):
        samples = self.thinned_samples(thinning=thinning)
        cross_chain_dims = None if self.n_used_chains == 1 else [1] * len(samples)
        ess = tfp.mcmc.effective_sample_size(samples, cross_chain_dims=cross_chain_dims)
        return ess

    def acceptance_ratio(self):
        return tf.reduce_mean(tf.cast(self.trace[0], tf.float32), axis=0)

    def leapfrog_steps_taken(self):
        """
        Returns the means and standard deviations of the leapfrog steps taken of the
        individual chains.
        """
        if self.sampler == "nuts":
            mean = tf.reduce_mean(tf.cast(self.trace[1], "float32"), axis=0)
            std = tf.math.reduce_std(tf.cast(self.trace[1], "float32"), axis=0)
            return mean, std
        else:
            return self.num_leapfrog_steps, 0

    def _base_predict(self, x, thinning):
        """
        Produces a predictive normal distribution for every parameter sample.

        Returns:
            prediction: IndependentNormal with event shape of x.shape and batch shape of
                        (n_samples / thinning, n_chains)
        """
        samples = self.thinned_samples(thinning=thinning)
        weights, unconstrained_noise_scale = self._split_sample_weights_and_noise_scale(
            samples
        )
        model = build_scratch_density_model(
            weights,
            self.layer_activation_functions,
            self.transform_unconstrained_scale_factor,
            unconstrained_noise_scale,
        )
        prediction = model(x)
        return prediction

    def predict(self, x, thinning=1):
        """
        Produces the networks prediction as a mixture of the individual predictions of
        every parameter sample in every chain (possibly skipping some dependent on
        thinning).

        Args:
            x (numpy array or tensorflow tensor): The locations at which to predict.
                                                  Shape (n_samples, n_features)
            thinning (int):                       Parameter to control how many samples
                                                  to skip. E.g.
                                                  thinning=1 => use all samples
                                                  thinning=10 => use every tenth sample

        Returns:
            prediction: MixtureSameFamily of IndependentNormals with event shape of
                        x.shape and batch shape of []
        """
        return self.predict_mixture_of_gaussians(x, thinning=thinning)

    def predict_mixture_of_gaussians(self, x, thinning=1):
        samples = self.thinned_samples(thinning=thinning)
        samples = tf.nest.map_structure(
            lambda x: tf.reshape(x, shape=(x.shape[0] * x.shape[1],) + x.shape[2:]),
            samples,
        )
        n_samples = samples[0].shape[0]
        weights, unconstrained_noise_scale = self._split_sample_weights_and_noise_scale(
            samples
        )
        model = build_scratch_density_model(
            weights,
            self.layer_activation_functions,
            self.transform_unconstrained_scale_factor,
            unconstrained_noise_scale,
        )
        prediction = model(x)
        cat = tfp.distributions.Categorical(probs=tf.ones((n_samples,)) / n_samples)
        predictive_mixture = MixtureSameFamilySampleFix(cat, prediction)
        # predictive_mixture = tfd.MixtureSameFamily(cat, prediction)

        return predictive_mixture
        # The following way of doing the prediction does not work in some cases due to tensorflow issue https://github.com/tensorflow/probability/issues/1206
        # prediction = self._base_predict(x, thinning=thinning)
        # n_samples = prediction.batch_shape[0]
        # prediction = tfp.distributions.BatchReshape(
        #     prediction, (n_samples * self.n_used_chains,)
        # )
        #
        # cat = tfp.distributions.Categorical(
        #     probs=tf.ones((n_samples * self.n_used_chains,))
        #     / (n_samples * self.n_used_chains)
        # )
        # predictive_mixture = tfp.distributions.MixtureSameFamily(cat, prediction)
        # return predictive_mixture

    def predict_chains(self, x, thinning=1):
        prediction = self._base_predict(x, thinning=thinning)
        n_samples = prediction.batch_shape[0]
        prediction = tfp.distributions.BatchReshape(
            prediction, (self.n_used_chains, n_samples)
        )

        cat = tfp.distributions.Categorical(probs=tf.ones((n_samples,)) / n_samples)
        predictive_mixture = MixtureSameFamilySampleFix(cat, prediction)
        # predictive_mixture = tfd.MixtureSameFamily(cat, prediction)
        return predictive_mixture

    def predict_individual_chain(self, x, i_chain, thinning=1):
        samples = self.thinned_samples(thinning=thinning)
        n_samples = samples[0].shape[0]
        samples = tf.nest.map_structure(lambda x: x[:, i_chain], samples)
        weights, unconstrained_noise_scale = self._split_sample_weights_and_noise_scale(
            samples
        )
        model = build_scratch_density_model(
            weights,
            self.layer_activation_functions,
            self.transform_unconstrained_scale_factor,
            unconstrained_noise_scale,
        )
        prediction = model(x)
        cat = tfp.distributions.Categorical(probs=tf.ones((n_samples,)) / n_samples)
        predictive_mixture = MixtureSameFamilySampleFix(cat, prediction)
        # predictive_mixture = tfd.MixtureSameFamily(cat, prediction)
        return predictive_mixture

    def predict_random_sample(self, x, seed=0):
        i_chain = tf.random.uniform(
            (), maxval=self.n_used_chains, dtype=tf.dtypes.int32, seed=seed
        )
        i_sample = tf.random.uniform(
            (), maxval=self.n_samples, dtype=tf.dtypes.int32, seed=seed
        )
        sample = tf.nest.map_structure(lambda x: x[i_sample, i_chain], self.samples)
        return self.predict_from_sample_parameters(x, sample)

    def predict_individual_chain_start_stop_indices(
        self, x, i_chain, start_index, stop_index, thinning=1
    ):
        samples = self.thinned_samples(thinning=thinning)
        n_samples = samples[0].shape[0]
        samples = tf.nest.map_structure(lambda x: x[:, i_chain], samples)
        samples = tf.nest.map_structure(
            lambda x: x[int(start_index / thinning) : int(stop_index / thinning)],
            samples,
        )
        n_samples = samples[0].shape[0]
        weights, unconstrained_noise_scale = self._split_sample_weights_and_noise_scale(
            samples
        )
        model = build_scratch_density_model(
            weights,
            self.layer_activation_functions,
            self.transform_unconstrained_scale_factor,
            unconstrained_noise_scale,
        )
        prediction = model(x)
        cat = tfp.distributions.Categorical(probs=tf.ones((n_samples,)) / n_samples)
        predictive_mixture = MixtureSameFamilySampleFix(cat, prediction)
        # predictive_mixture = tfd.MixtureSameFamily(cat, prediction)
        return predictive_mixture

    def predict_from_indices(self, x, indices):
        samples = tf.nest.map_structure(
            lambda x: tf.gather_nd(x, indices), self.samples
        )
        n_samples = samples[0].shape[0]
        weights, unconstrained_noise_scale = self._split_sample_weights_and_noise_scale(
            samples
        )
        model = build_scratch_density_model(
            weights,
            self.layer_activation_functions,
            self.transform_unconstrained_scale_factor,
            unconstrained_noise_scale,
        )
        prediction = model(x)
        cat = tfp.distributions.Categorical(probs=tf.ones((n_samples,)) / n_samples)
        # predictive_mixture = MixtureSameFamilySampleFix(cat, prediction)
        predictive_mixture = tfd.MixtureSameFamily(cat, prediction)
        return predictive_mixture

    # very useful for debugging
    def predict_from_sample_parameters(self, x, sample_parameters):
        """
        sample_parameters is a list of tensors or arrays specifying the network
        parameters.
        """
        weights, unconstrained_noise_scale = self._split_sample_weights_and_noise_scale(
            sample_parameters
        )
        model = build_scratch_density_model(
            weights,
            self.layer_activation_functions,
            self.transform_unconstrained_scale_factor,
            unconstrained_noise_scale=unconstrained_noise_scale,
        )
        prediction = model(x)
        return prediction

    def save(self, save_path):
        init_dict = dict(
            input_shape=self.input_shape,
            layer_units=self.layer_units,
            layer_activations=self.layer_activations,
            transform_unconstrained_scale_factor=self.transform_unconstrained_scale_factor,
            # network_prior=self.network_prior,
            # noise_scale_prior=self.noise_scale_prior,
            sampler=self.sampler,
            num_burnin_steps=self.num_burnin_steps,
            discard_burnin_samples=self.discard_burnin_samples,
            step_size=self.step_size,
            num_leapfrog_steps=self.num_leapfrog_steps,
            max_tree_depth=self.max_tree_depth,
            num_steps_between_results=self.num_steps_between_results,
            seed=self.seed,
        )
        fit_dict = dict(
            _chain=self._chain,
            _trace=self._trace,
            chain_mask=self.chain_mask,
            kernel_results=self.kernel_results,
        )
        dic = dict(init=init_dict, fit=fit_dict)
        pickle_large_object(save_path, dic)


def pickle_large_object(save_path, object):
    """
    Due to an error of python on OS X objects larger than 4GB cannot be pickled and unpickled directly
    (see bug https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb). This function is a workaround to pickle larger objects.
    """
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(object, protocol=4)  # not sure if this is needed
    with open(save_path, "wb") as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx : idx + max_bytes])


def unpickle_large_object(save_path):
    """
    Due to an error of python on OS X objects larger than 4GB cannot be pickled and unpickled directly
    (see bug https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb). This function is a workaround to unpickle larger objects.
    """
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(save_path)
    with open(save_path, "rb") as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    object = pickle.loads(bytes_in)
    return object


def hmc_density_network_from_save_path(
    save_path, network_prior=None, noise_scale_prior=None
):
    """
    Recreates a hmc density network from a save-path. Unfortunately, tensorflow
    probability distributions cannot reliably be pickled (specifically it doesn't work
    for transformed distributions). Since the noise_scale_prior is often a transformed
    distribution I chose not to save it to disk. Therefore it needs to be provided here
    as an additional argument if you want to e.g. resume training or sample from the
    prior. Don't worry about providing it if you only care about the saved network's
    prediction.
    """
    dic = unpickle_large_object(save_path)
    # with open(save_path, "rb") as f:
    #     dic = pickle.load(f)
    dic["init"]["network_prior"] = network_prior
    dic["init"]["noise_scale_prior"] = noise_scale_prior
    hmc_net = HMCDensityNetwork(**dic["init"])
    fit_d = dic["fit"]
    hmc_net._chain = fit_d["_chain"]
    hmc_net._trace = fit_d["_trace"]
    hmc_net.chain_mask = fit_d["chain_mask"]
    hmc_net.kernel_results = fit_d["kernel_results"]
    return hmc_net


def undo_masking(hmc_net):
    # Undo masking that was done so far
    undo_mask = tf.cast(tf.ones((hmc_net.n_chains,)), "bool")
    hmc_net.mask_chains(undo_mask)
    return hmc_net


def mask_nonsense_chains(
    hmc_net,
    median_scale_cutter=None,
    lowest_acceptance_ratio=None,
    highest_acceptance_ratio=None,
    x_train=None,
    y_train=None,
    thinning=1,
):
    """
    Function that takes an HMC network and masks some MCMC chains based on nonsensical
    samples.

    Args:
        median_scale_cutter (float, optional):
                Chains with a median scale above this value are masked. Useful since we
                are often predicting in a normalized y-space and therefore know that
                noise scales cannot be above 1.
        lowest_acceptance_ratio (float, optional):
                Chains with a lower acceptance ratio are masked.
        highest_acceptance_ratio (float, optional):
                Chains with a higher acceptance ratio are masked.
        x_train (numpy array or tensorflow tensor, optional):
                Actual data. If both x_train and y_train are provided, then chains are
                masked out if the chain's sample with highest posterior probability has
                a lower probability then the highest of the minimal posterior
                probabilities of samples across chains.
    """
    undo_masking(hmc_net)
    chain_mask = tf.cast(tf.ones((hmc_net.n_chains,)), "bool")
    if median_scale_cutter is not None:
        # Mask by nonsense scales
        scales = transform_unconstrained_scale(
            hmc_net.samples[-1], factor=hmc_net.transform_unconstrained_scale_factor
        )
        median_scale_per_chain = tfp.stats.percentile(
            scales, 50.0, interpolation="midpoint", axis=(0, 2, 3)
        )
        scale_mask = tf.greater_equal(median_scale_cutter, median_scale_per_chain)
        chain_mask = tf.math.logical_and(chain_mask, scale_mask)
        print(
            "Chains removed because of nonsense scales:",
            hmc_net.n_chains
            - tf.reduce_sum(tf.cast(scale_mask, dtype="int32")).numpy(),
        )
        print(
            "Rejected median scales:",
            median_scale_per_chain[tf.math.logical_not(scale_mask)].numpy(),
        )
        print()

    if lowest_acceptance_ratio is not None:
        # Mask by acceptance ratios
        acceptance_ratio = hmc_net.acceptance_ratio()
        high_enough = tf.less_equal(lowest_acceptance_ratio, acceptance_ratio)
        low_enough = tf.greater_equal(highest_acceptance_ratio, acceptance_ratio)
        acceptance_mask = tf.math.logical_and(high_enough, low_enough)
        chain_mask = tf.math.logical_and(chain_mask, acceptance_mask)
        print(
            "Chains removed because of nonsense acceptance ratio:",
            hmc_net.n_chains
            - tf.reduce_sum(tf.cast(acceptance_mask, dtype="int32")).numpy(),
        )
        print(
            "Rejected acceptance ratios:",
            acceptance_ratio[tf.math.logical_not(acceptance_mask)].numpy(),
        )
        print()

    if x_train is not None:
        log_posterior = hmc_net._target_log_prob_fn_factory(x_train, y_train)
        log_posteriors = log_posterior(
            *tf.nest.map_structure(lambda x: x[::thinning], hmc_net.samples)
        )
        largest_min = tf.reduce_max(tf.reduce_min(log_posteriors, axis=0))
        highest_log_posterior_by_chain = tf.reduce_max(log_posteriors, axis=0)
        highest_log_posterior = tf.reduce_max(log_posteriors)
        posterior_mask = tf.greater(
            highest_log_posterior_by_chain, largest_min - tf.math.log(1.0)
        )
        chain_mask = tf.math.logical_and(chain_mask, posterior_mask)
        print(
            "Chains removed because all samples had very low posterior probability:",
            hmc_net.n_chains
            - tf.reduce_sum(tf.cast(posterior_mask, dtype="int32")).numpy(),
        )
        print("The highest log posterior:", highest_log_posterior.numpy())
        print("The highest minimal log posterior over chains:", largest_min.numpy())
        print(
            "The highest log posteriors of the removed chains:",
            highest_log_posterior_by_chain[tf.math.logical_not(posterior_mask)].numpy(),
        )

    hmc_net.mask_chains(chain_mask)
    return hmc_net
