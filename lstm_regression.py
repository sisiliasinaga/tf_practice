import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn import datasets
from sklearn.model_selection import train_test_split

boston_data = datasets.load_boston()
x_data, y_data = datasets.load_boston(return_X_y=True)
x_split, x_test, y_split, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_split, y_split, test_size=0.33, random_state=42)

learning_rate = 0.001
time_steps = 28
n_input = 13
num_units = 128
batch_size = 128

x = tf.placeholder("float", [None, time_steps, n_input])
y = tf.placeholder("float", [None, 1])

weights = {
    'output': tf.Variable(tf.random_normal([num_units, 1]))
}

biases = {
    'output': tf.Variable(tf.random_normal([1]))
}

input = tf.unstack(x, time_steps, 1)

lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

pred = tf.matmul(outputs[-1], weights['output']) + biases['output']

cost = tf.reduce_mean(tf.squared_difference(y, pred))
my_opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

# Training
iter = 1
total_batch = int(len(y_train) / batch_size)
while iter < 800:
    avg_cost = 0
    for i in range(total_batch):
        batch_x, batch_y = x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
        batch_x = batch_x.reshape([batch_size, time_steps, n_input])
        sess.run(my_opt, feed_dict={x: batch_x, y: batch_y})
        # avg_cost += c / total_batch
        if iter % 10 == 0:
            costs = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print('Iter', iter, 'cost =', costs)
print(sess.run(cost, feed_dict={x: x_test, y: y_test}))
