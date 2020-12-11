# with lots of inspiration from https://janosh.io/blog/hmc-bnn
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
        n_chains=1,
        seed=0,
    ):
        """
        A neural network whose posterior distribution parameters are estimated by HMC.
        Each individual network sample models a function and a homoscedastic standard
        deviation is put around that function in the form of a Normal distribution.
        The class can run several MCMC chains in parallel.

        Params:
            weight_priors (list of tfp distributions):
                Must have same length as layer_units. Each distribution must be scalar
                and is applied to the respective layer weights.
                In some future version, it might be possible to pass a distribution
                with the right shape.
            bias_priors (list of tfp distributions):
                Same as weight_prior but for biases.
            noise_scale_prior
        """
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.transform_unconstrained_scale_factor = transform_unconstrained_scale_factor
        self.network_prior = network_prior
        self.noise_scale_prior = noise_scale_prior
        self.sampler = sampler
        self.num_burnin_steps = num_burnin_steps
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.max_tree_depth = max_tree_depth
        self.n_chains = n_chains
        self.seed = seed

        tf.random.set_seed(self.seed)

        self.layer_activation_functions = activation_strings_to_activation_functions(
            self.layer_activations
        )

        self.step_size_adapter = {
            "simple": tfp.mcmc.SimpleStepSizeAdaptation,
            "dual_averaging": tfp.mcmc.DualAveragingStepSizeAdaptation,
        }[step_size_adapter]

        self._samples = None

    @property
    def samples(self):
        if self._samples is None:
            self._samples = tf.nest.map_structure(
                lambda x: x[self.num_burnin_steps :], self.chain
            )
        return self._samples

    def log_prior(self, weights, unconstrained_noise_scale):
        log_prob = 0
        for w, w_prior in zip(weights, self.network_prior):
            log_prob += w_prior.log_prob(w)

        noise_scale = transform_unconstrained_scale(
            unconstrained_noise_scale, factor=self.transform_unconstrained_scale_factor
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
            unconstrained_noise_scale = current_state[-1]
            weights = current_state[0:-1]  # includes biases
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
        noise_scale = self.noise_scale_prior.sample(n_samples)
        unconstrained_noise_scale = backtransform_constrained_scale(
            noise_scale, factor=self.transform_unconstrained_scale_factor
        )
        prior_state.append(unconstrained_noise_scale)
        return prior_state

    @tf.function  # (experimental_compile=True)
    def _sample_chain(
        self, num_burnin_steps, num_results, current_state, previous_kernel_results=None
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
            previous_kernel_results=previous_kernel_results,
            kernel=self.adaptive_kernel,
            return_final_kernel_results=True,
            trace_fn=trace_fn,
        )
        return chain, trace, final_kernel_results

    def fit(self, x_train, y_train, current_state=None, num_results=5000, resume=False):
        assert current_state[0].shape[0] == self.n_chains
        target_log_prob_fn = self._target_log_prob_fn_factory(x_train, y_train)

        if resume:
            num_burnin_steps = 0
            # initial_states = [
            #     list(tf.nest.map_structure(lambda param: param[-1], samples))
            #     for samples in self.samples
            # ]
            # previous_kernel_results = self.final_kernel_results

            # not adapted yet!
            current_state = tf.nest.map_structure(lambda x: x[-1], self.chain)
        else:
            num_burnin_steps = self.num_burnin_steps
            if self.sampler == "nuts":
                kernel = tfp.mcmc.NoUTurnSampler(
                    target_log_prob_fn,
                    max_tree_depth=self.max_tree_depth,
                    step_size=self.step_size,
                )
                self.adaptive_kernel = self.step_size_adapter(
                    kernel,
                    num_adaptation_steps=int(self.num_burnin_steps * 0.8),
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
                self.adaptive_kernel = self.step_size_adapter(
                    kernel, num_adaptation_steps=int(self.num_burnin_steps * 0.8)
                )

            self.kernel_results = self.adaptive_kernel.bootstrap_results(current_state)

            chain, trace, self.kernel_results = self._sample_chain(
                num_burnin_steps=num_burnin_steps,
                num_results=num_results,
                current_state=current_state,
                previous_kernel_results=self.kernel_results,
            )
            if not resume:
                self.chain = chain
                self.trace = trace

            else:
                self.chain = tf.nest.map_structure(
                    lambda *parts: tf.concat(parts, axis=0), *[self.chain, chain]
                )
                self.trace = tf.nest.map_structure(
                    lambda *parts: tf.concat(parts, axis=0), *[self.trace, trace]
                )
        return self

    def potential_scale_reduction(self):
        return tfp.mcmc.potential_scale_reduction(self.samples)

    ## OLD ##
    # def ess(self, **kwargs):
    #     """
    #     Estimate effective sample size of Markov chain(s).
    #     """
    #     mean_esss = []
    #     esss = []
    #     for i in range(self.n_chains):
    #         ess = tfp.mcmc.effective_sample_size(self.samples[i], **kwargs)
    #         esss.append(ess)
    #         flat_ess = tf.concat([tf.reshape(t, (-1)) for t in ess], axis=0)
    #         mean_ess = tf.reduce_mean(flat_ess)
    #         mean_esss.append(mean_ess)
    #         print(tf.math.reduce_min(flat_ess))
    #         print(tf.math.reduce_max(flat_ess))
    #
    #     return esss, mean_esss
    #
    ## OLD ##
    # def acceptance_ratio(self):
    #     acceptance_ratios = []
    #     for i in range(self.n_chains):
    #         acceptance_ratios.append(
    #             tf.reduce_sum(tf.cast(self.trace[i][0], tf.float32))
    #             / len(self.trace[i][0])
    #         )
    #     return tf.convert_to_tensor(acceptance_ratios)
    #
    ## OLD ##
    # def leapfrog_steps_taken(self):
    #     """
    #     Returns the means and standard deviations of the leapfrog steps taken of the
    #     individual chains.
    #     """
    #     if self.sampler == "nuts":
    #         means = []
    #         stds = []
    #         for i in range(self.n_chains):
    #             means.append(tf.reduce_mean(self.trace[i][1]))
    #             stds.append(tf.math.reduce_std(tf.cast(self.trace[i][1], tf.float32)))
    #         return tf.convert_to_tensor(means), tf.convert_to_tensor(stds)
    #     else:
    #         return self.num_leapfrog_steps, 0
    #

    def predict(self, x):
        return self.predict_mixture_of_gaussians(x)

    def predict_mixture_of_gaussians(self, x):
        n_samples = self.samples[0].shape[0]
        model = build_scratch_density_model(
            self.samples[:-1],
            self.layer_activation_functions,
            self.transform_unconstrained_scale_factor,
            self.samples[-1],
        )
        prediction = model(x)
        prediction = tfp.distributions.BatchReshape(
            prediction, (n_samples * self.n_chains,)
        )

        cat = tfp.distributions.Categorical(
            probs=tf.ones((n_samples * self.n_chains,)) / (n_samples * self.n_chains)
        )
        predictive_mixture = tfp.distributions.MixtureSameFamily(cat, prediction)
        return predictive_mixture

    ## OLD ##
    # def predict_list_of_gaussians(self, x_test, thinning=1, n_predictions=None):
    #     """
    #     Produces a list of aleatoric Gaussian predictions from the sampled parameters
    #     out of the markov chain. Together they also represent epistemic uncertainty.
    #     Args:
    #         thinning (int):      (thinning - 1) samples will be skipped in between two
    #                              predictions.
    #         n_predictions (int): How many samples should be used to make predictions.
    #                              If n_predictions is specified, thinning will be
    #                              ignored.
    #     """
    #     n_samples = len(self.combined_samples[0])
    #     if n_predictions is None:
    #         loop_over = tf.range(0, n_samples, thinning)
    #     else:
    #         loop_over = _linspace_network_indices(n_samples, n_predictions)
    #     # A network parameterized by a sample from the chain produces a (aleatoric)
    #     # predictive normal distribution for the x_test. They are all accumulated in
    #     # this list
    #     predictive_distributions = []
    #     for i_sample in loop_over:
    #         params_list = [param[i_sample] for param in self.combined_samples]
    #         noise_std = transform_unconstrained_scale(
    #             params_list[-1], factor=self.transform_unconstrained_scale_factor
    #         )
    #         weights_list = params_list[:-1]
    #         model = build_scratch_model(
    #             weights_list, noise_std, self.layer_activation_functions
    #         )
    #         prediction = model(x_test)
    #         predictive_distributions.append(prediction)
    #     return predictive_distributions
    #
    ## OLD ##
    # def predict_mixture_of_gaussians(self, cat_probs, gaussians):
    #     return tfd.Mixture(cat=tfd.Categorical(probs=cat_probs), components=gaussians)
    #
    ## OLD ##
    # def predict(self, x_test, thinning=1):
    #     gaussians = self.predict_list_of_gaussians(x_test, thinning=thinning)
    #     cat_probs = tf.ones((x_test.shape[0],) + (1, len(gaussians))) / len(gaussians)
    #     return self.predict_mixture_of_gaussians(cat_probs, gaussians)
    #
    ## OLD ##
    # def predict_unnormalized_space(self, x_test, mean, scale, thinning=1):
    #     """
    #     Predict and transform the predictive distribution into the unnormalized space
    #
    #     Args:
    #         mean (float):  Mean of the unnormalized training data. I.e. mean of the
    #                        z-transformation that transformed the data into the normalized
    #                        space.
    #         scale (float): Scale of the unnormalized training data.
    #     """
    #     gaussians = self.predict_list_of_gaussians(x_test, thinning=thinning)
    #     unnormalized_gaussians = []
    #     for prediction in gaussians:
    #         predictive_mean = prediction.mean()
    #         predictive_std = prediction.stddev()
    #         predictive_mean = (predictive_mean * scale) + mean
    #         predictive_std = predictive_std * scale
    #         unnormalized_gaussians.append(tfd.Normal(predictive_mean, predictive_std))
    #     cat_probs = tf.ones((x_test.shape[0],) + (1, len(gaussians))) / len(gaussians)
    #     return self.predict_mixture_of_gaussians(cat_probs, unnormalized_gaussians)

    ## OLD ##
    # # very useful for debugging
    # def predict_list_from_sample_indices(self, x_test, burnin=False, indices=[0]):
    #     predictive_distributions = []
    #     for i_sample in indices:
    #         params_list = [param[i_sample] for param in self.samples]
    #         noise_std = transform_unconstrained_scale(
    #             params_list[-1], factor=self.transform_unconstrained_scale_factor
    #         )
    #         weights_list = params_list[:-1]
    #         model = build_scratch_model(
    #             weights_list, noise_std, self.layer_activation_functions
    #         )
    #         prediction = model(x_test)
    #         predictive_distributions.append(prediction)
    #     return predictive_distributions

    # very useful for debugging
    def predict_from_sample_parameters(self, x_test, sample_parameters):
        """
        sample_parameters is a list of tensors or arrays specifying the network
        parameters.
        """
        unconstrained_noise_scale = sample_parameters[-1]
        weights = sample_parameters[:-1]
        model = build_scratch_density_model(
            weights,
            self.layer_activation_functions,
            self.transform_unconstrained_scale_factor,
            unconstrained_noise_scale=unconstrained_noise_scale,
        )
        prediction = model(x_test)
        return prediction

    ## OLD ##
    # # very useful for debugging
    # def predict_mixture_from_sample_indices(self, x_test, burnin=False, indices=[0]):
    #     gaussians = self.predict_list_from_sample_indices(
    #         x_test, burnin=burnin, indices=indices
    #     )
    #     cat_probs = tf.ones(x_test.shape + (len(gaussians),)) / len(gaussians)
    #     return self.predict_mixture_of_gaussians(cat_probs, gaussians)

    ## OLD ##
    # def predict_with_prior_samples(self, x_test, n_samples=5, seed=0):
    #     predictive_distributions = []
    #     for sample in range(n_samples):
    #         prior_sample = self.sample_prior_state(seed=seed + sample * 0.01)
    #         prediction = self.predict_from_sample_parameters(x_test, prior_sample)
    #         predictive_distributions.append(prediction)
    #     return predictive_distributions

    ## OLD ##
    # def save(self, save_path):
    #     with open(save_path, "wb") as f:
    #         pickle.dump(self, f, -1)


## OLD ##
# def hmc_network_from_save_path(save_path):
#     with open(save_path, "rb") as f:
#         hmc_net = pickle.load(f)
#     return hmc_net
