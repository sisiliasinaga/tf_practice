import tensorflow as tf
import numpy as np
import pandas as pd
import hyperengine as hype
from tensorflow.python.framework import ops
ops.reset_default_graph()

readCSV = pd.read_csv('C:/Users/yuiichiiros/Downloads/adjusted_file_na2o.csv')

x_data = readCSV[['312.979', '334.54599', '335.79401', '818.99371', '819.80206', '246.847', '279.89401', '280.09601',
                  '416.60229', '429.45056', '549.03223', '590.69543', '620.13879', '819.60004']]
x_data = x_data.astype(np.float32)
y_data = readCSV['Na2O']
y_data = y_data.astype(np.float32)

x_train = x_data[:697]
x_test = x_data[697:1392]
x_val = x_data[1392:]

y_train = y_data[:697]
y_test = y_data[697:1392]
y_val = y_data[1392:]

data = hype.Data(train=hype.DataSet(x_train, y_train),
                 validation=hype.DataSet(x_val, y_val),
                 test=hype.DataSet(x_test, y_test))


def dnn_model(params):
    x = tf.placeholder(shape=[None, 14], dtype=tf.float32, name='input')
    y = tf.placeholder(shape=None, dtype=tf.float32, name='label')
    mode = tf.placeholder(tf.string, name='mode')
    training = tf.equal(mode, 'train')

    weights = {
        1: tf.Variable(tf.random_normal([14, params.hidden_size], stddev=0.1))
    }
    biases = {
        1: tf.Variable(tf.random_normal([params.hidden_size]))
    }
    for i in range(params.hidden_layers-1):
        weights[i+2] = tf.Variable(tf.random_normal([params.hidden_size, params.hidden_size], stddev=0.1))
        biases[i+2] = tf.Variable(tf.random_normal([params.hidden_size]))
    weights['output'] = tf.Variable(tf.random_normal([params.hidden_size, 1], stddev=0.1))
    biases['output'] = tf.Variable(tf.random_normal([1]))

    layer = tf.add(tf.matmul(tf.sigmoid(x), weights[1]), biases[1])
    for hidden_size in range(params.hidden_layers-1):
        layer = tf.add(tf.matmul(tf.sigmoid(layer), weights[hidden_size+2]), biases[hidden_size+2])
    output = tf.add(tf.matmul(tf.sigmoid(layer), weights['output']), biases['output'])

    cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(output, y)), name='loss')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params.learning_rate).minimize(cost, name='minimize')
    tf.reduce_mean(tf.cast(tf.abs(output - y) < 0.5, tf.float32), name='accuracy')


def solver_generator(params):
    solver_params = {
        'batch_size': 694,
        'epochs': 1000,
        'evaluate_test': True,
        'eval_flexible': True,
    }
    dnn_model(params)
    solver = hype.TensorflowSolver(data=data, hyper_params=params, **solver_params)
    return solver


hyper_params_spec = hype.spec.new(
    hidden_size=hype.spec.choice([12, 14, 16, 18, 20]),
    hidden_layers=hype.spec.choice([1, 2, 3]),
    learning_rate=10**hype.spec.uniform(-1, -3),
)
strategy_params = {
    'io_load_dir': 'C:/Users/yuiichiiros/Documents/MLP',
    'io_save_dir': 'C:/Users/yuiichiiros/Documents/MLP'
}

tuner = hype.HyperTuner(hyper_params_spec, solver_generator, **strategy_params)
tuner.tune()
