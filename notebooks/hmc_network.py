# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from core.hmc.hmc_network import run_hmc, target_log_prob_fn_factory
from core.map import MapDensityNetwork
from core.network_utils import (
    activation_strings_to_activation_functions,
    build_scratch_model,
)
from core.preprocessing import preprocess_create_x_train_test
from data.toy_regression import create_linear_data, ground_truth_linear_function

tfd = tfp.distributions


# %% codecell
n_train = 20
m = 1
b = 3
_x_train, y_train = create_linear_data(n_train, m=m, b=b, sigma=0.5)
x_train, _x_test, x_test = preprocess_create_x_train_test(_x_train)
y_test = ground_truth_linear_function(_x_test, m=m, b=b)


# %% codecell
input_shape = [1]
layer_units = [3, 2]
layer_activations = ["relu", "linear"]

# %%
initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
)

net = MapDensityNetwork(
    input_shape=input_shape,
    layer_units=layer_units,
    layer_activations=layer_activations,
    learning_rate=lr_schedule,
)

net.fit(x_train=x_train, y_train=y_train, batch_size=20, epochs=10)

# %%
prediction = net.predict(x_test)
mean = prediction.mean().numpy()
std = prediction.stddev().numpy()

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.1)
ax.plot(_x_test, mean, label=f"Mean prediction", alpha=0.8)
ax.fill_between(
    _x_test.flatten(),
    mean.flatten() - 2 * std.flatten(),
    mean.flatten() + 2 * std.flatten(),
    alpha=0.2,
    label="95% HDR prediction",
)
ax.scatter(_x_train, y_train, c="k", marker="x", s=100, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
ax.legend()

# %%
initial_state = net.get_weights()
initial_state

# %%
# w_0 = np.array([[1, 1, 1]], dtype="float32")
# b_0 = np.array([0, 0, 0], dtype="float32")
# w_1 = np.random.normal(size=(3, 2)).astype("float32")
# b_1 = np.zeros(2, dtype="float32")
# initial_state = [w_0, b_0, w_1, b_1]
num_results = 1000
num_burnin_steps = 1000
step_size = 0.1
num_leapfrog_steps = 5
step_size_adapter = "dual_averaging"
sampler = "hmc"

target_log_prob_fn = target_log_prob_fn_factory(x_train, y_train, layer_activations)


burnin, chain, is_accepted, final_kernel_results = run_hmc(
    target_log_prob_fn,
    step_size=step_size,
    num_leapfrog_steps=num_leapfrog_steps,
    current_state=initial_state,
    num_burnin_steps=num_burnin_steps,
    num_results=num_results,
    step_size_adapter=step_size_adapter,
    sampler=sampler,
)
# sample_mean = tf.reduce_mean(samples)
# sample_stddev = tf.math.reduce_std(samples)
is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))

print(f"acceptance:{is_accepted:.4f}")


layer_activation_functions = activation_strings_to_activation_functions(
    layer_activations
)

# %%
n_predictions = 3
n_samples = chain[0].shape[0]
interval = n_samples / n_predictions
fig, ax = plt.subplots(figsize=(8, 8))
for i, sample_i in enumerate(np.arange(0, n_samples, interval) + interval - 1):
    sample_i = int(sample_i)
    c = sns.color_palette()[i]
    weights_list = [param[sample_i] for param in chain]
    model = build_scratch_model(weights_list, layer_activation_functions)
    # mean, std = model(x_test)
    # print(mean.shape)
    prediction = model(x_test)
    mean = prediction.mean().numpy()
    std = prediction.stddev().numpy()
    ax.plot(_x_test, mean, label=f"Mean {sample_i} prediction", c=c, alpha=0.8)
    ax.fill_between(
        _x_test.flatten(),
        mean.flatten() - 2 * std.flatten(),
        mean.flatten() + 2 * std.flatten(),
        color=c,
        alpha=0.1,
        label=f"95% HDR {sample_i} prediction",
    )
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.1)
ax.scatter(_x_train, y_train, c="k", marker="x", s=100, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
ax.legend()


# %%
target_log_prob_fn(*initial_state)
# %%
sample_i = 165
weights_list = [param[sample_i] for param in chain]
target_log_prob_fn(*weights_list)
# %%
print(initial_state)
print(weights_list)
# %%
model = build_scratch_model(initial_state, layer_activation_functions)
# print(y_train.flatten().shape)
tf.reduce_sum(model(x_train).log_prob(y_train.flatten()))
# %%
for p, i in zip(burnin, initial_state):
    print(p[0].numpy())
    print(i)


# %%
w = chain[0]
b = chain[1]
w = tf.reshape(w, (1, 2, -1))
b = tf.reshape(b, (2, -1))
model = build_scratch_model([w, b], layer_activation_functions)
model(x_test).shape
# for p in chain:
#    print(p.shape)

# %% codecell
n_models = 20
n_samples = chain[0].shape[0]
jump_over = n_samples / n_models
network_samples = []
for sample in chain[::jump_over]:
    print()
    network = []
    for param_samples in chain:
        network.append(param_samples[0])  # print(t.shape)
    # print(t[0].shape)


model = build_scratch_model(network, layer_activation_functions)

mean, std = model(x_test, training=False)
mean = mean.numpy()
std = std.numpy()

fig, ax = plt.subplots(figsize=(8, 8))
c = sns.color_palette()[0]
ax.plot(_x_test, mean, label=f"Mean prediction", c=c, alpha=0.8)
ax.fill_between(
    _x_test.flatten(),
    mean.flatten() - 2 * std.flatten(),
    mean.flatten() + 2 * std.flatten(),
    color=c,
    alpha=0.2,
    label="95% HDR prediction",
)
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.1)
ax.scatter(_x_train, y_train, c="k", marker="x", s=100, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim([-5, 5])
ax.legend()


# %%
post_params = [tf.reduce_mean(t, axis=0) for t in chain]
post_model = build_scratch_model(post_params, layer_activation_functions)


mean, std = post_model(x_test, training=False)
mean = mean.numpy()
std = std.numpy()

fig, ax = plt.subplots(figsize=(8, 8))
c = sns.color_palette()[0]
ax.plot(_x_test, mean, label=f"Mean prediction", c=c, alpha=0.8)
ax.fill_between(
    _x_test.flatten(),
    mean.flatten() - 2 * std.flatten(),
    mean.flatten() + 2 * std.flatten(),
    color=c,
    alpha=0.2,
    label="95% HDR prediction",
)
ax.plot(_x_test, y_test, label="Ground truth", alpha=0.1)
ax.scatter(_x_train, y_train, c="k", marker="x", s=100, label="Train data")
ax.set_xlabel("")
ax.set_ylabel("")
# ax.set_ylim([-5, 5])
ax.legend()
