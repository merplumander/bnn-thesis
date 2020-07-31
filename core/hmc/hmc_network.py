# with lots of inspiration from https://janosh.io/blog/hmc-bnn

import pickle
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp

from ..map import MapNetwork
from ..network_utils import (
    _linspace_network_indices,
    activation_strings_to_activation_functions,
    backtransform_constrained_scale,
    build_scratch_model,
    transform_unconstrained_scale,
)

tfd = tfp.distributions

# def transform_unconstrained_scale(scale, factor=1):
#     return scale
#
# def backtransform_constrained_scale(scale, factor=1):
#     return scale


class HMCNetwork:
    def __init__(
        self,
        input_shape=[1],
        layer_units=[200, 100, 2],
        layer_activations=["relu", "relu", "linear"],
        transform_unconstrained_scale_factor=0.05,
        weight_priors=[tfd.Normal(0, 0.2)] * 3,
        bias_priors=[tfd.Normal(0, 0.2)] * 3,
        std_prior=tfd.Normal(2, 0.01),  # currently does nothing
        sampler="nuts",
        seed=0,
    ):
        """
        A neural network whose posterior distribution parameters is estimated by HMC.
        Each individual network sample models a function and a homoscedastic standard
        deviation is put around that function in the form of a Normal distribution.

        Params:
            weight_priors (list of tfp distributions):
                Must have same length as layer_units. Each distribution must be scalar
                is applied to the respective layer weights.
                In some future version, it might be possible to pass a distribution
                with the right shape.
            bias_priors (list of tfp distributions):
                Same as weight_prior but for biases.
            std_prior
        """
        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
        self.transform_unconstrained_scale_factor = transform_unconstrained_scale_factor
        self.weight_priors = weight_priors
        self.bias_priors = bias_priors
        self.std_prior = std_prior
        self.sampler = sampler
        self.seed = seed

        self.initial_state = None

        self.burnin = None
        self.samples = None
        self.trace = None
        self.final_kernel_results = None

        self.layer_activation_functions = activation_strings_to_activation_functions(
            self.layer_activations
        )

    def _initial_state_through_map_estimation(
        self, x_train, y_train, learning_rate, batch_size, epochs, verbose=1
    ):
        net = MapNetwork(
            input_shape=self.input_shape,
            layer_units=self.layer_units,
            layer_activations=self.layer_activations,
            learning_rate=learning_rate,
            seed=self.seed,
        )
        net.fit(
            x_train=x_train,
            y_train=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
        )
        self.initial_state = net.get_weights()
        self.initial_state.append(
            backtransform_constrained_scale(
                net.sample_std, factor=self.transform_unconstrained_scale_factor
            ),
        )
        return self.initial_state

    def log_prior(self, current_state):
        log_prob = 0
        for w, b, w_prior, b_prior in zip(
            current_state[::2],
            current_state[1::2],
            self.weight_priors,
            self.bias_priors,
        ):
            log_prob += tf.reduce_sum(w_prior.log_prob(w))
            log_prob += tf.reduce_sum(b_prior.log_prob(b))
        return log_prob

    def log_likelihood(self, weights, noise_std, x, y):
        model = build_scratch_model(weights, noise_std, self.layer_activation_functions)
        return tf.reduce_sum(model(x).log_prob(y))

    def _target_log_prob_fn_factory(self, x, y):
        def log_posterior(*current_state):
            noise_std = transform_unconstrained_scale(
                current_state[-1], factor=self.transform_unconstrained_scale_factor
            )
            weights = current_state[0:-1]  # includes biases
            log_prob = self.log_prior(weights)
            log_prob += self.log_likelihood(weights, noise_std, x, y)
            return log_prob
            # model = build_scratch_model(current_state, self.layer_activation_functions)
            # return tf.reduce_sum(model(x_train).log_prob(y_train))

        return log_posterior

    def sample_prior_state(self, overdisp=1.0, seed=0):
        """Draw random samples for weights and biases of a NN according to some
        specified prior distributions. This set of parameter values can serve as a
        starting point for MCMC or gradient descent training.
        """
        tf.random.set_seed(seed)
        init_state = []
        for n1, n2, w_prior, b_prior in zip(
            self.input_shape + self.layer_units,
            self.layer_units,
            self.weight_priors,
            self.bias_priors,
        ):
            w_shape, b_shape = [n1, n2], n2
            # Use overdispersion > 1 for better R-hat statistics.
            w = w_prior.sample(w_shape) * overdisp
            b = b_prior.sample(b_shape) * overdisp
            init_state.extend([tf.Variable(w), tf.Variable(b)])
        return init_state

    @tf.function  # (experimental_compile=True)
    def _sample_chain(
        self,
        num_burnin_steps,
        num_results,
        current_state,
        previous_kernel_results,
        adaptive_kernel,
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
            kernel=adaptive_kernel,
            return_final_kernel_results=True,
            trace_fn=trace_fn,
        )
        return chain, trace, final_kernel_results

    def fit(
        self,
        x_train,
        y_train,
        initial_state=None,
        step_size_adapter="dual_averaging",
        num_burnin_steps=1000,
        num_results=5000,
        num_leapfrog_steps=25,
        step_size=0.1,
        resume=False,
        learning_rate=0.01,
        batch_size=20,
        epochs=100,
        verbose=1,
    ):
        self.num_leapfrog_steps = num_leapfrog_steps
        if self.initial_state is None and initial_state is None:
            initial_state = self._initial_state_through_map_estimation(
                x_train,
                y_train,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
            )
        current_state = initial_state

        target_log_prob_fn = self._target_log_prob_fn_factory(x_train, y_train)

        tf.random.set_seed(self.seed)

        step_size_adapter = {
            "simple": tfp.mcmc.SimpleStepSizeAdaptation,
            "dual_averaging": tfp.mcmc.DualAveragingStepSizeAdaptation,
        }[step_size_adapter]
        if self.sampler == "nuts":
            kernel = tfp.mcmc.NoUTurnSampler(target_log_prob_fn, step_size=step_size)
            adaptive_kernel = step_size_adapter(
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
                step_size=step_size,
                num_leapfrog_steps=self.num_leapfrog_steps,
            )
            adaptive_kernel = step_size_adapter(
                kernel, num_adaptation_steps=int(num_burnin_steps * 0.8)
            )

        if resume:
            prev_chain, prev_trace, previous_kernel_results = (
                nest_concat(self.burnin, self.samples),
                self.trace,
                self.final_kernel_results,
            )
            # step = len(prev_chain)
            current_state = tf.nest.map_structure(lambda chain: chain[-1], prev_chain)
        else:
            previous_kernel_results = adaptive_kernel.bootstrap_results(current_state)
            # step = 0
        # Run the chain (with burn-in).
        # print("current state:", len(current_state), current_state)
        # print("previous_kernel_results:", len(previous_kernel_results), previous_kernel_results[0])
        chain, trace, final_kernel_results = self._sample_chain(
            num_burnin_steps=num_burnin_steps,
            num_results=num_results,
            current_state=current_state,
            previous_kernel_results=previous_kernel_results,
            adaptive_kernel=adaptive_kernel,
        )

        if resume:
            chain = nest_concat(prev_chain, chain)
            trace = nest_concat(prev_trace, trace)
        burnin, samples = zip(*[(t[:-num_results], t[-num_results:]) for t in chain])

        # print("number of chains:", tf.size(target_log_prob_fn(*current_state)))
        self.burnin = burnin
        self.samples = samples
        self.trace = trace
        self.final_kernel_results = final_kernel_results
        return self

    def ess(self, **kwargs):
        """
        Estimate effective sample size of Markov chain(s).
        """
        return tfp.mcmc.effective_sample_size(self.samples, **kwargs)

    def acceptance_ratio(self):
        return tf.reduce_sum(tf.cast(self.trace[0], tf.float32)) / len(self.trace[0])

    def leapfrog_steps_taken(self):
        """
        Returns the mean and standard deviation of the leapfrog steps taken.
        """
        if self.sampler == "nuts":
            return (
                tf.reduce_mean(self.trace[1]),
                tf.math.reduce_std(tf.cast(self.trace[1], tf.float32)),
            )
        else:
            return self.num_leapfrog_steps, 0

    def predict_list_of_gaussians(self, x_test, thinning=1, n_predictions=None):
        """
        Produces a list of aleatoric Gaussian predictions from the sampled parameters
        out of the markov chain. Together they also represent epistemic uncertainty.
        Args:
            thinning (int):      (thinning - 1) samples will be skipped in between two
                                 predictions.
            n_predictions (int): How many samples should be used to make predictions.
                                 If n_predictions is specified, thinning will be
                                 ignored.
        """
        n_samples = len(self.samples[0])
        if n_predictions is None:
            loop_over = tf.range(0, n_samples, thinning)
        else:
            loop_over = _linspace_network_indices(n_samples, n_predictions)
        # A network parameterized by a sample from the chain produces a (aleatoric)
        # predictive normal distribution for the x_test. They are all accumulated in
        # this list
        predictive_distributions = []
        for i_sample in loop_over:
            params_list = [param[i_sample] for param in self.samples]
            noise_std = transform_unconstrained_scale(
                params_list[-1], factor=self.transform_unconstrained_scale_factor
            )
            weights_list = params_list[:-1]
            model = build_scratch_model(
                weights_list, noise_std, self.layer_activation_functions
            )
            prediction = model(x_test)
            predictive_distributions.append(prediction)
        return predictive_distributions

    def predict_mixture_of_gaussians(self, cat_probs, gaussians):
        return tfd.Mixture(cat=tfd.Categorical(probs=cat_probs), components=gaussians)

    def predict(self, x_test, thinning=1):
        gaussians = self.predict_list_of_gaussians(x_test, thinning=thinning)
        cat_probs = tf.ones(x_test.shape + (len(gaussians),)) / len(gaussians)
        print(cat_probs.shape)
        print(gaussians[0])
        return self.predict_mixture_of_gaussians(cat_probs, gaussians)

    def predict_epistemic(self, x_test, thinning=1):
        gaussians = self.predict_list_of_gaussians(x_test, thinning=thinning)
        dirac_deltas = []
        for gaussian in gaussians:
            delta = tfp.distributions.Deterministic(
                loc=gaussian.mean()
            )  # tfd.Normal(gaussian.mean(), 1e-8)
            dirac_deltas.append(delta)
        cat_probs = tf.ones(x_test.shape + (len(gaussians),)) / len(gaussians)
        return self.predict_mixture_of_gaussians(cat_probs, dirac_deltas)

    # very useful for debugging
    def predict_list_from_sample_indices(self, x_test, burnin=False, indices=[0]):
        predictive_distributions = []
        for i_sample in indices:
            params_list = [param[i_sample] for param in self.samples]
            noise_std = transform_unconstrained_scale(
                params_list[-1], factor=self.transform_unconstrained_scale_factor
            )
            weights_list = params_list[:-1]
            model = build_scratch_model(
                weights_list, noise_std, self.layer_activation_functions
            )
            prediction = model(x_test)
            predictive_distributions.append(prediction)
        return predictive_distributions

    # very useful for debugging
    def predict_from_sample_parameters(self, x_test, sample_parameters):
        """
        sample_parameters is a list of tensors or arrays specifying the network
        parameters.
        """
        noise_std = transform_unconstrained_scale(
            sample_parameters[-1], factor=self.transform_unconstrained_scale_factor
        )
        weights_list = sample_parameters[:-1]
        model = build_scratch_model(
            weights_list, noise_std, self.layer_activation_functions
        )
        prediction = model(x_test)
        return prediction

    # very useful for debugging
    def predict_mixture_from_sample_indices(self, x_test, burnin=False, indices=[0]):
        gaussians = self.predict_list_from_sample_indices(
            x_test, burnin=burnin, indices=indices
        )
        cat_probs = tf.ones(x_test.shape + (len(gaussians),)) / len(gaussians)
        return self.predict_mixture_of_gaussians(cat_probs, gaussians)

    def predict_with_prior_samples(self, x_test, n_samples=5, seed=0):
        predictive_distributions = []
        for sample in range(n_samples):
            prior_sample = self.sample_prior_state(seed=seed + sample * 0.01)
            prediction = self.predict_from_sample_parameters(x_test, prior_sample)
            predictive_distributions.append(prediction)
        return predictive_distributions

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(self, f, -1)


def hmc_network_from_save_path(save_path):
    with open(save_path, "rb") as f:
        hmc_net = pickle.load(f)
    return hmc_net


def nest_concat(*args, axis=0):
    """Utility function for concatenating a new Markov chain or trace with
    older ones when resuming a previous calculation.
    """
    return tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=axis), *args)
