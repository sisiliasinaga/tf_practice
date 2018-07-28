import tensorflow as tf
import numpy as np
import pandas as pd
import hyperengine as hype
from sklearn import datasets
from sklearn.model_selection import train_test_split

x_data, y_data = datasets.load_boston(return_X_y=True)
x_split, x_test, y_split, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_split, y_split, test_size=0.33, random_state=7)

data = hype.Data(train=hype.DataSet(x_train, y_train),
                 validation=hype.DataSet(x_val, y_val),
                 test=hype.DataSet(x_test, y_test))


def dnn_model(params):
    x = tf.placeholder(shape=[None, 13], dtype=tf.float32, name='input')
    y = tf.placeholder(shape=None, dtype=tf.float32, name='label')
    mode = tf.placeholder(tf.string, name='mode')
    training = tf.equal(mode, 'train')

    weights = {
        1: tf.Variable(tf.random_normal([13, 6], stddev=0.1)),
        2: tf.Variable(tf.random_normal([6, 6], stddev=0.1)),
        'output': tf.Variable(tf.random_normal([6, 1], stddev=0.1))
    }
    biases = {
        1: tf.Variable(tf.random_normal([6])),
        2: tf.Variable(tf.random_normal([6])),
        'output': tf.Variable(tf.random_normal([1]))
    }

    layer = tf.add(tf.matmul(tf.sigmoid(x), weights[1]), biases[1])
    layer = tf.add(tf.matmul(tf.sigmoid(layer), weights[2]), biases[2])
    output = tf.add(tf.matmul(tf.sigmoid(layer), weights['output']), biases['output'])

    cost = tf.reduce_mean(tf.squared_difference(output, y), name='loss')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params.learning_rate).minimize(cost, name='minimize')
    tf.reduce_mean(tf.cast(tf.abs(output - y) < 0.5, tf.float32), name='accuracy')


def solver_generator(params):
    solver_params = {
        'batch_size': 167,
        'epochs': 50,
        'evaluate_test': True,
        'eval_flexible': True,
    }
    dnn_model(params)
    solver = hype.TensorflowSolver(data=data, hyper_params=params, **solver_params)
    return solver


hyper_params_spec = hype.spec.new(
    learning_rate=10**hype.spec.uniform(-1, -3),
)

tuner = hype.HyperTuner(hyper_params_spec, solver_generator)
tuner.tune()
