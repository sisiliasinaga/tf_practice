import tensorflow as tf
import numpy as np
import math
from tensorflow.python.framework import ops
from sklearn import datasets
from sklearn.model_selection import train_test_split

x_data, y_data = datasets.load_boston(return_X_y=True)
x_split, x_test, y_split, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_split, y_split, test_size=0.33, random_state=42)

learning_rate = 0.001
epochs = 2000
batch_size = 3

x = tf.placeholder(shape=[None, 13], dtype=tf.float32)
y = tf.placeholder(shape=None, dtype=tf.float32)


# creates activation function
def gaussian_function(input_layer):
    initial = math.exp(-2*math.pow(input_layer, 2))
    return initial


np_gaussian_function = np.vectorize(gaussian_function)


def d_gaussian_function(input_layer):
    initial = -4 * input_layer * math.exp(-2*math.pow(input_layer, 2))
    return initial


np_d_gaussian_function = np.vectorize(d_gaussian_function)

np_d_gaussian_function_32 = lambda input_layer: np_d_gaussian_function(input_layer).astype(np.float32)


def tf_d_gaussian_function(input_layer, name=None):
    with ops.name_scope(name, "d_gaussian_function", [input_layer]) as name:
        y = tf.py_func(np_d_gaussian_function_32, [input_layer], [tf.float32], name=name, stateful=False)
    return y[0]


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFunGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def gaussian_function_grad(op, grad):
    input_variable = op.inputs[0]
    n_gr = tf_d_gaussian_function(input_variable)
    return grad * n_gr


np_gaussian_function_32 = lambda input_layer: np_gaussian_function(input_layer).astype(np.float32)


def tf_gaussian_function(input_layer, name=None):
    with ops.name_scope(name, "gaussian_function", [input_layer]) as name:
        y = py_func(np_gaussian_function_32, [input_layer], [tf.float32],
                    name=name, grad=gaussian_function_grad)
    return y[0]
# end of defining activation function

def rbf_network(input_layer, weight_arr, bias_arr):
    layer1 = tf.add(tf.matmul(tf_gaussian_function(input_layer), weight_arr['h1']), bias_arr['b1'])
    layer2 = tf.add(tf.matmul(tf_gaussian_function(layer1), weight_arr['h2']), bias_arr['b2'])
    output = tf.matmul(tf_gaussian_function(layer2), weight_arr['output']) + bias_arr['output']
    return output


weights = {
    'h1': tf.Variable(tf.random_normal([13, 256], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([256, 256], stddev=0.1)),
    'output': tf.Variable(tf.random_normal([256, 1], stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.random_normal([256])),
    'b2': tf.Variable(tf.random_normal([256])),
    'output': tf.Variable(tf.random_normal([1]))
}

pred = rbf_network(x, weights, biases)

cost = tf.reduce_mean(tf.squared_difference(y, pred))
my_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

# Training loop
total_batch = int(len(y_train) / batch_size)
for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_x, batch_y = x_train[i * batch_size:(i + 1) * batch_size], y_train[i * batch_size:(i + 1) * batch_size]
        _, c = sess.run([my_opt, cost], feed_dict={x: batch_x, y: batch_y})
        avg_cost += c / total_batch
    if epoch % 10 == 0:
        print('Epoch:', epoch, 'cost =', '{:.3f}'.format(avg_cost))
print(sess.run(cost, feed_dict={x: x_test, y: y_test}))
