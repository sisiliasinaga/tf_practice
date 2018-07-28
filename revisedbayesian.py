import tensorflow as tf
import pandas as pd
import edward as ed
import hyperengine as hype
from edward.models import Normal

readCSV = pd.read_csv('C:/Users/yuiichiiros/Downloads/adjusted_file.csv')

x_data = readCSV[['257.90399', '256.80801', '634.87054', '576.64545', '252.13901', '421.67935', '333.56', '247.96201',
                  '390.64609', '635.08667', '453.17175', '251.823', '390.69318', '285.28', '288.34799', '249.81599',
                  '252.086', '453.67944', '281.69601', '390.74026', '389.18384', '443.09866', '317.92499', '453.13266',
                  '246.209', '635.3028', '254.084', '455.54559', '777.38055', '334.233', '252.19099', '276.27399',
                  '777.79205', '275.616']]
y_data = readCSV[['SiO2']]

x_data_np = x_data.as_matrix()
y_data_np = y_data.as_matrix()

x_train = x_data_np[:695]
x_test = x_data_np[695:1388]
x_val = x_data_np[1388:2083]
y_train = y_data_np[:695]
y_test = y_data_np[695:1388]
y_val = y_data_np[1388:2082]

data = hype.Data(train=hype.DataSet(x_train, y_train),
                 validation=hype.DataSet(x_val, y_val),
                 test=hype.DataSet(x_test, y_test))

n_samples = len(readCSV['275.616']) / 5
n_samples = int(n_samples)

len_train = len(x_train)
len_test = len(x_test)
features = 34


def bayesian_ann(params):
    x = tf.placeholder(shape=[None, features], dtype=tf.float32, name='input')
    y = tf.placeholder(shape=None, dtype=tf.float32, name='label')
    mode = tf.placeholder(tf.string, name='mode')
    training = tf.equal(mode, 'train')

    weights = {
        1: Normal(loc=tf.zeros([features, params.hidden_size]), scale=tf.ones([features, params.hidden_size]))
    }
    biases = {
        1: Normal(loc=tf.zeros(params.hidden_size), scale=tf.ones(params.hidden_size))
    }
    for i in range(params.hidden_layers - 1):
        weights[i+2] = Normal(loc=tf.zeros([params.hidden_size, params.hidden_size]),
                              scale=tf.ones([params.hidden_size]))
        biases[i+2] = Normal(loc=tf.zeros(params.hidden_size), scale=tf.ones(params.hidden_size))
    weights['output'] = Normal(loc=tf.zeros([params.hidden_size, 1]), scale=tf.ones([params.hidden_size, 1]))
    biases['output'] = Normal(loc=tf.zeros(1), scale=tf.ones(1))

    layer = tf.tanh(tf.matmul(x, weights[1]) + biases[1])
    for hidden_size in range(params.hidden_layers - 1):
        layer = tf.tanh(tf.matmul(layer, weights[hidden_size+2]) + biases[hidden_size+2])
    output = tf.reshape(tf.tanh(tf.matmul(layer, weights['output']) + biases['output']), [694, 1])

    qweights = {
        1: Normal(loc=tf.get_variable("loc", [features, params.hidden_size]),
                  scale=tf.nn.softplus(tf.get_variable("scale", [features, params.hidden_size])))
    }
    qbiases = {
        1: Normal(loc=tf.get_variable("loc", [params.hidden_size]),
                  scale=tf.nn.softplus(tf.get_variable("scale", [params.hidden_size])))
    }
    for j in range(len(weights)):
        qweights[j+2] = Normal(loc=tf.get_variable("loc", [features, params.hidden_size]),
                               scale=tf.nn.softplus(tf.get_variable("scale", [features, params.hidden_size])))
        qbiases[j+2] = Normal(loc=tf.get_variable("loc", [params.hidden_size]),
                              scale=tf.nn.softplus(tf.get_variable("scale", [params.hidden_size])))
    qweights['output'] = Normal(loc=tf.get_variable("loc", [params.hidden_size, 1]),
                                scale=tf.nn.softplus(tf.get_variable("scale", [params.hidden_size])))
    qbiases['output'] = Normal(loc=tf.get_variable("loc", [1]), scale=tf.nn.softplus(tf.get_variable("scale", [1])))

    inference =
