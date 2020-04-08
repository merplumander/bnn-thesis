# with lots of inspiration from https://janosh.io/blog/hmc-bnn

import tensorflow as tf
import tensorflow_probability as tfp

from ..network_utils import (
    activation_strings_to_activation_functions,
    build_scratch_model,
)

tfd = tfp.distributions


def target_log_prob_fn_factory(
    x_train, y_train, layer_activations
):  # this is not done yet. you need to finish the factory
    layer_activation_functions = activation_strings_to_activation_functions(
        layer_activations
    )

    def target_log_prob_fn(*current_state):

        # # print("current_state\n", current_state)
        # # print(w)
        # # print(b)
        # w = current_state[0]
        # b = current_state[1]
        # # _w = tf.expand_dims(wb[0, 0], axis=0)#tf.reshape(tf.stack((w, tf.constant(0., dtype='float32')), axis=0), (1, -1))
        # # _b = wb[0, 1]
        # # print(_w.shape)
        # net = tf.matmul(x_train, w) + b
        # y_pred, y_unconstrained_std = tf.unstack(net, axis=1)
        # prediction = tfp.distributions.Normal(
        #     loc=y_pred, scale=1e-6 + tf.math.softplus(0.05 * y_unconstrained_std)
        # )
        model = build_scratch_model(current_state, layer_activation_functions)

        return tf.reduce_sum(model(x_train).log_prob(y_train))

    return target_log_prob_fn


@tf.function
def run_hmc(
    target_log_prob_fn,
    current_state=None,
    resume=None,
    num_results=500,
    num_burnin_steps=500,
    step_size=1.0,
    num_leapfrog_steps=2,
    step_size_adapter="dual_averaging",
    sampler="hmc",
):
    err = "Either current_state or resume is required when calling run_hmc"
    assert current_state is not None or resume is not None, err

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
        prev_chain, prev_trace, prev_kernel_results = resume
        # step = len(prev_chain)
        current_state = tf.nest.map_structure(lambda chain: chain[-1], prev_chain)
    else:
        prev_kernel_results = adaptive_kernel.bootstrap_results(current_state)
        # step = 0

    # Run the chain (with burn-in).
    chain, trace, final_kernel_results = tfp.mcmc.sample_chain(
        num_results=num_burnin_steps + num_results,
        current_state=current_state,
        previous_kernel_results=prev_kernel_results,
        kernel=adaptive_kernel,
        return_final_kernel_results=True,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )
    # print("chain",chain)
    # print()
    # print("trace",trace)
    # print()
    # print("final kernel results", final_kernel_results)

    if resume:
        chain = nest_concat(prev_chain, chain)
        trace = nest_concat(prev_trace, trace)
    burnin, samples = zip(*[(t[:-num_results], t[-num_results:]) for t in chain])
    return burnin, samples, trace, final_kernel_results


def nest_concat(*args, axis=0):
    """Utility function for concatenating a new Markov chain or trace with
    older ones when resuming a previous calculation.
    """
    return tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=axis), *args)
