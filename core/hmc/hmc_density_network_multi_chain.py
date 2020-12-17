# with inspiration from https://janosh.io/blog/hmc-bnn
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


class HMCDensityNetwork:
    def __init__(
        self,
        input_shape=[1],
        layer_units=[20, 10, 1],
        layer_activations=["relu", "relu", "linear"],
        transform_unconstrained_scale_factor=0.5,
        network_prior=None,
        noise_scale_prior=None,
        sampler="hmc",
        step_size_adapter="dual_averaging",
        num_burnin_steps=1000,
        step_size=0.01,
        num_leapfrog_steps=100,  # only relevant for HMC
        max_tree_depth=10,  # only relevant for NUTS
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
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.max_tree_depth = max_tree_depth
        self.seed = seed

        self.layer_activation_functions = activation_strings_to_activation_functions(
            self.layer_activations
        )

        self.step_size_adapter = {
            "simple": tfp.mcmc.SimpleStepSizeAdaptation,
            "dual_averaging": tfp.mcmc.DualAveragingStepSizeAdaptation,
        }[self._step_size_adapter]

        self._samples = None
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

    @property
    def _non_burnin_samples(self):
        return tf.nest.map_structure(lambda x: x[self.num_burnin_steps :], self._chain)

    def _apply_chain_mask(self, samples):
        if self.chain_mask is not None:
            samples = tf.nest.map_structure(
                lambda x: tf.boolean_mask(x, self.chain_mask, axis=1), samples
            )
        return samples

    def _assign_used_samples(self):
        _samples = self._non_burnin_samples
        self._samples = self._apply_chain_mask(_samples)

    @property
    def samples(self):
        if self._samples is None:
            self._assign_used_samples()
        return self._samples

    @property
    def trace(self):
        trace = tf.nest.map_structure(lambda x: x[self.num_burnin_steps :], self._trace)
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
        self._assign_used_samples()

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
        """Draw random samples for weights and biases of a NN according to some
        specified prior distributions. This set of parameter values can serve as a
        starting point for MCMC or gradient descent training.
        """
        tf.random.set_seed(seed)
        prior_state = []
        for w_prior in self.network_prior:
            w_sample = w_prior.sample(n_samples) * overdisp
            prior_state.append(w_sample)
        if self.noise_scale_prior is not None:
            noise_scale = self.noise_scale_prior.sample(n_samples)
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
            seed=self.seed,
        )
        return chain, trace, final_kernel_results

    def fit(self, x_train, y_train, current_state=None, num_results=5000, resume=False):
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
            self._chain = chain
            self._trace = trace

        else:
            self._chain = tf.nest.map_structure(
                lambda *parts: tf.concat(parts, axis=0), *[self._chain, chain]
            )
            self._trace = tf.nest.map_structure(
                lambda *parts: tf.concat(parts, axis=0), *[self._trace, trace]
            )
        # After changing self._chain we always need to call _assign_used_samples to ensure that all relevant samples from _chain show up in self.samples.
        self._assign_used_samples()
        return self

    def potential_scale_reduction(self):
        print(self.samples[0].shape)
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
        prediction = self._base_predict(x, thinning=thinning)
        n_samples = prediction.batch_shape[0]
        prediction = tfp.distributions.BatchReshape(
            prediction, (n_samples * self.n_used_chains,)
        )

        cat = tfp.distributions.Categorical(
            probs=tf.ones((n_samples * self.n_used_chains,))
            / (n_samples * self.n_used_chains)
        )
        predictive_mixture = tfp.distributions.MixtureSameFamily(cat, prediction)
        return predictive_mixture

    def predict_chains(self, x, thinning=1):
        prediction = self._base_predict(x, thinning=thinning)
        n_samples = prediction.batch_shape[0]
        prediction = tfp.distributions.BatchReshape(
            prediction, (self.n_used_chains, n_samples)
        )

        cat = tfp.distributions.Categorical(probs=tf.ones((n_samples,)) / n_samples)
        predictive_mixture = tfp.distributions.MixtureSameFamily(cat, prediction)
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
        predictive_mixture = tfp.distributions.MixtureSameFamily(cat, prediction)
        return predictive_mixture

    def predict_random_sample(self, x, seed=0):
        tf.random.set_seed(seed)
        i_chain = tf.random.uniform(
            (), maxval=self.n_used_chains, dtype=tf.dtypes.int32, seed=seed
        )
        i_sample = tf.random.uniform(
            (), maxval=self.n_samples, dtype=tf.dtypes.int32, seed=seed
        )
        sample = tf.nest.map_structure(lambda x: x[i_sample, i_chain], self.samples)
        return self.predict_from_sample_parameters(x, sample)

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
            network_prior=self.network_prior,
            noise_scale_prior=self.noise_scale_prior,
            sampler=self.sampler,
            num_burnin_steps=self.num_burnin_steps,
            step_size=self.step_size,
            num_leapfrog_steps=self.num_leapfrog_steps,
            max_tree_depth=self.max_tree_depth,
            seed=self.seed,
        )
        fit_dict = dict(
            _samples=self._samples,
            _chain=self._chain,
            _trace=self._trace,
            chain_mask=self.chain_mask,
            kernel_results=self.kernel_results,
        )
        dic = dict(init=init_dict, fit=fit_dict)
        with open(save_path, "wb") as f:
            pickle.dump(dic, f, -1)


def hmc_density_network_from_save_path(save_path):
    with open(save_path, "rb") as f:
        dic = pickle.load(f)
    hmc_net = HMCDensityNetwork(**dic["init"])
    fit_d = dic["fit"]
    hmc_net._samples = fit_d["_samples"]
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
):
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
            *tf.nest.map_structure(lambda x: x, hmc_net.samples)
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
