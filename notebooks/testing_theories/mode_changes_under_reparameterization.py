import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


# %% markdown
# One problem with MAP estimation is that modes change when we apply a reparameterization to a parameter (but leaving the probabilistic model the same). For example look at the distribution of a variance parameter below and the corresponding distribution on the standard deviation parameter. The mode of the variance distribution is at 0.5, implying a mode of the standard deviation distribution at 0.5^2, if modes were stable under a reparameterization. But the mode of the standard deviation distribution is at ~1.22.

# %%
low = 0.5
high = 1.5
triangular = tfd.Triangular(low=low, high=high, peak=low)
uniform = tfd.Uniform(low, high)
uniform_p = 0.9
var_d = tfd.Mixture(
    cat=tfd.Categorical(probs=[uniform_p, 1 - uniform_p]),
    components=[uniform, triangular],
)
scale_d = tfd.TransformedDistribution(
    distribution=var_d, bijector=tfb.Invert(tfb.Square())
)
var_samples = var_d.sample(40000)
x = np.linspace(low - 0.2, high + 0.2, 200)
fig, ax = plt.subplots()
ax.plot(x, var_d.prob(x), label="var")
ax.plot(x, scale_d.prob(x), label="scale")
ax.hist(var_samples, density=True, bins=30, alpha=0.2)
ax.hist(np.sqrt(var_samples), density=True, bins=30, alpha=0.2)
