import tensorflow as tf
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()

readCSV = pd.read_csv('C:/Users/yuiichiiros/Downloads/adjusted_file_na2o.csv')

x_data = readCSV[['312.979', '334.54599', '335.79401', '818.99371', '819.80206', '246.847', '279.89401', '280.09601',
                  '416.60229', '429.45056', '549.03223', '590.69543', '620.13879', '819.60004']]
y_data = readCSV['Na2O']

x_train = x_data[:697]
x_test = x_data[697:1392]
x_val = x_data[1392:]

y_train = y_data[:697]
y_test = y_data[697:1392]
y_val = y_data[1392:]

batch_size = 4
training_epochs = 30
learning_rate = 0.001062
display_step = 1
hidden_size = 12
epochs = 10000

x = tf.placeholder(shape=[None, 14], dtype=tf.float32)
y = tf.placeholder(shape=None, dtype=tf.float32)

weights = {
    'h1': tf.Variable(tf.random_normal([14, hidden_size], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1)),
    'h3': tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1)),
    'output': tf.Variable(tf.random_normal([hidden_size, 1], stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.random_normal([hidden_size])),
    'b2': tf.Variable(tf.random_normal([hidden_size])),
    'b3': tf.Variable(tf.random_normal([hidden_size])),
    'output': tf.Variable(tf.random_normal([1]))
}


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


def py_func(func, inp, tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFunGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, tout, stateful=stateful, name=name)


def gaussian_function_grad(op, grad):
    input_variable = op.inputs[0]
    n_gr = tf_d_gaussian_function(input_variable)
    return grad * n_gr


np_gaussian_function_32 = lambda input_layer: np_gaussian_function(input_layer).astype(np.float32)


def tf_gaussian_function(input_layer, name=None):
    with ops.name_scope(name, "gaussian_function", [input_layer]) as name:
        y_var = py_func(np_gaussian_function_32, [input_layer], [tf.float32],
                        name=name, grad=gaussian_function_grad)
    return y_var[0]
# end of defining activation function


def rbf_network(input_layer, weights_arr, biases_arr):
    layer1 = tf.add(tf.matmul(tf_gaussian_function(input_layer), weights_arr['h1']), biases_arr['b1'])
    layer2 = tf.add(tf.matmul(tf_gaussian_function(layer1), weights_arr['h2']), biases_arr['b2'])
    output = tf.matmul(tf_gaussian_function(layer2), weights_arr['output']) + biases_arr['output']
    return output


pred = rbf_network(x, weights, biases)
cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(pred, y)))
my_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

train_cost_list = []
val_cost_list = []

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

total_batch = int(len(y_train) / batch_size)
for epoch in range(epochs):
    avg_cost = 0
    for j in range(total_batch):
        batch_x, batch_y = x_train[j * batch_size:(j + 1) * batch_size], y_train[j * batch_size:(j + 1) * batch_size]
        _, c = sess.run([my_opt, cost], feed_dict={x: batch_x, y: batch_y})
        avg_cost += c / total_batch
    if epoch % 10 == 0:
        print('Epoch:', epoch, 'cost =', '{:.5f}'.format(avg_cost))
        train_cost_list.append(avg_cost)
        if epoch % 100 == 0:
            v = sess.run(cost, feed_dict={x: x_val, y: y_val})
            val_cost_list.append(v)
            print(v)

print(sess.run(cost, feed_dict={x: x_test, y: y_test}))
sess.close()

print(train_cost_list)
print(val_cost_list)

plt.plot(train_cost_list, 'r-', label='Training Accuracy')
plt.plot(val_cost_list, 'g-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.legend(loc='lower right')
plt.show()

plt.plot(x_train, y_train, 'o')
plt.plot(x_train, pred, 'r-', linewidth=3)
plt.title('MLP - Training')
plt.xlabel('Performance')
plt.ylabel('SiO2')
plt.show()

plt.plot(x_val, y_val, 'o')
plt.plot(x_val, pred, 'r-', linewidth=3)
plt.title('MLP - Validation')
plt.xlabel('Performance')
plt.ylabel('SiO2')
plt.show()

plt.plot(x_test, y_test, 'o')
plt.plot(x_test, pred, 'r-', linewidth=3)
plt.title('MLP - Testing')
plt.xlabel('Performance')
plt.ylabel('SiO2')
plt.show()
