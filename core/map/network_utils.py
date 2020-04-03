import numpy as np
import tensorflow as tf


def build_keras_model(
    input_shape=[1],
    layer_units=[200, 100, 1],
    layer_activations=["relu", "relu", "linear"],
):
    """Building an uncompiled keras tensorflow model with the architecture given"""
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(
            layer_units[0], activation=layer_activations[0], input_shape=input_shape
        )
    )
    for units, activation in zip(layer_units[1:], layer_activations[1:]):
        model.add(tf.keras.layers.Dense(units, activation=activation))
    return model


def train_ml_model(
    train_x, train_y, layer_units, layer_activations, batchsize_train, seed=0
):
    print("This function is deprecated!")
    tf.random.set_seed(seed)
    model = build_keras_model(
        layer_units=layer_units, layer_activations=layer_activations
    )
    # return model
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=20, decay_rate=0.9, staircase=True
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule), loss="mean_squared_error"
    )
    model.fit(train_x, train_y, epochs=120, batch_size=batchsize_train)
    return model
