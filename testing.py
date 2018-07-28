from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split

x_data, y_data = datasets.load_boston(return_X_y=True)
x_split, y_split, x_test, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
x_train, y_train, x_val, y_val = train_test_split(x_split, y_split, test_size=0.33, random_state=42)


def lstm_model(num_units, rnn_layers, dense_layers=None, learning_rate=0.1, optimizer='Adagrad'):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param num_units: the size of the cells.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """

    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(
                    layer['num_units'], state_is_tuple=True
                ),
                layer['keep_prob']
            ) if layer.get('keep_prob') else tf.contrib.rnn.BasicLSTMCell(
                    layer['num_units'],
                    state_is_tuple=True
                ) for layer in layers
            ]
        return [tf.contrib.rnn.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = tf.unstack(X, axis=1, num=num_units)
        output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = dnn_layers(output[-1], dense_layers)
        prediction, loss = tflearn.models.linear_regression(output, y)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
            learning_rate=learning_rate)
        return prediction, loss, train_op

    return _lstm_model

log_dir = './ops_logs/sin'
time_steps = 3
rnn_layers = [{'num_units': 5}]
dense_layers = None
training_steps = 10000
print_steps = training_steps / 10
batch_size = 100

regressor = learn.SKCompat(learn.Estimator(model_fn=lstm_model(time_steps, rnn_layers, dense_layers), model_dir=log_dir))
