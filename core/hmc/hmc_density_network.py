# with lots of inspiration from https://janosh.io/blog/hmc-bnn

import tensorflow as tf
import tensorflow_probability as tfp

from ..map import MapDensityNetwork
from ..network_utils import (
    activation_strings_to_activation_functions,
    build_scratch_model,
)

tfd = tfp.distributions


class HMCDensityNetwork:
    def __init__(
        self,
        input_shape=[1],
        layer_units=[200, 100, 2],
        layer_activations=["relu", "relu", "linear"],
        seed=0,
    ):

        self.input_shape = input_shape
        self.layer_units = layer_units
        self.layer_activations = layer_activations
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
        net = MapDensityNetwork(
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
        return self.initial_state

    def _target_log_prob_fn_factory(self, x_train, y_train):
        def target_log_prob_fn(*current_state):
            model = build_scratch_model(current_state, self.layer_activation_functions)
            return tf.reduce_sum(model(x_train).log_prob(y_train))

        return target_log_prob_fn

    @tf.function
    def _sample_chain(
        self,
        num_burnin_steps,
        num_results,
        current_state,
        previous_kernel_results,
        adaptive_kernel,
    ):
        chain, trace, final_kernel_results = tfp.mcmc.sample_chain(
            num_results=num_burnin_steps + num_results,
            current_state=current_state,
            previous_kernel_results=previous_kernel_results,
            kernel=adaptive_kernel,
            return_final_kernel_results=True,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        )
        return chain, trace, final_kernel_results

    def fit(
        self,
        x_train,
        y_train,
        initial_state=None,
        step_size_adapter="dual_averaging",
        sampler="hmc",
        num_burnin_steps=1000,
        num_results=5000,
        num_leapfrog_steps=5,
        step_size=0.1,
        resume=False,
        learning_rate=0.01,
        batch_size=20,
        epochs=100,
        verbose=1,
    ):

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
        if sampler == "nuts":
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
        elif sampler == "hmc":
            kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
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
            loop_over = tf.cast(
                tf.math.ceil(
                    tf.linspace(
                        0, tf.constant(n_samples - 1, dtype=tf.float32), n_predictions,
                    )
                ),
                tf.int32,
            )
            # interval = int(n_samples / n_predictions)
            # loop_over = tf.range(0, n_samples, interval) + interval - 1
        # A network parameterized by a sample from the chain produces a (aleatoric)
        # predictive normal distribution for the x_test. They are all accumulated in
        # this list
        predictive_distributions = []
        for i_sample in loop_over:
            weights_list = [param[i_sample] for param in self.samples]
            model = build_scratch_model(weights_list, self.layer_activation_functions)
            prediction = model(x_test)
            predictive_distributions.append(prediction)
        return predictive_distributions

    def predict_mixture_of_gaussians(self, x_test, thinning=1):
        gaussians = self.predict_list_of_gaussians(x_test, thinning=thinning)
        cat_probs = tf.ones(x_test.shape + (len(gaussians),)) / len(gaussians)
        return tfd.Mixture(cat=tfd.Categorical(probs=cat_probs), components=gaussians)

    def predict(self, x_test, thinning=1):
        return self.predict_mixture_of_gaussians(x_test=x_test, thinning=thinning)

    # very useful for debugging
    def predict_by_sample_indices(self, x_test, burnin=False, indices=[0]):
        predictive_distributions = []
        for i_sample in indices:
            weights_list = [param[i_sample] for param in self.samples]
            model = build_scratch_model(weights_list, self.layer_activation_functions)
            prediction = model(x_test)
            predictive_distributions.append(prediction)
        return predictive_distributions

    # very useful for debugging
    def predict_from_sample_parameters(self, x_test, sample_parameters):
        """
        sample_parameters is a list of tensors or arrays specifying the network
        parameters.
        """
        model = build_scratch_model(sample_parameters, self.layer_activation_functions)
        prediction = model(x_test)
        return prediction

    # # this probably makes no sense
    # def predict_aleatoric_only(self, x_test):
    #     post_point_params = [tf.reduce_mean(t, axis=0) for t in self.samples]
    #     post_point_model = build_scratch_model(post_point_params, self.layer_activation_functions)
    #     return post_point_model(x_test)


def nest_concat(*args, axis=0):
    """Utility function for concatenating a new Markov chain or trace with
    older ones when resuming a previous calculation.
    """
    return tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=axis), *args)
