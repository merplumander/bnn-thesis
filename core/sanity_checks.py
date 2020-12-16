import numpy as np


def check_posterior_equivalence(map_net, hmc_net, x, y):
    """
    Takes a MapDensityNetwork, an HMCDensityNetwork and some x and y values and checks
    whether the map loss is equivalent to the unnormalized log posterior probability.
    """
    n_train = x.shape[0]
    map_loss = map_net.evaluate(x, y)
    map_prob = -map_loss * n_train
    weights = map_net.get_weights()
    log_posterior_fn = hmc_net._target_log_prob_fn_factory(x, y)
    log_prob = log_posterior_fn(*weights).numpy()
    return np.isclose(map_prob, log_prob)
