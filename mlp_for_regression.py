import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

x_data, y_data = datasets.load_boston(return_X_y=True)
x_split, x_test, y_split, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
x_train, x_validation, y_train, y_validation = train_test_split(x_split, y_split, test_size=0.33, random_state=7)

learning_rate = 0.001
epochs = 2000
batch_size = 3

x = tf.placeholder(shape=[None, 13], dtype=tf.float32)
y = tf.placeholder(shape=None, dtype=tf.float32)


def mlp(input_layer, weight_arr, bias_arr):
    layer1 = tf.add(tf.matmul(tf.sigmoid(input_layer), weight_arr['h1']), bias_arr['b1'])
    layer2 = tf.add(tf.matmul(tf.sigmoid(layer1), weight_arr['h2']), bias_arr['b2'])
    output = tf.matmul(tf.sigmoid(layer2), weight_arr['output']) + bias_arr['output']
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

pred = mlp(x, weights, biases)

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
        batch_x, batch_y = x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
        _, c = sess.run([my_opt, cost], feed_dict={x: batch_x, y: batch_y})
        avg_cost += c / total_batch
    if epoch % 10 == 0:
        print('Epoch:', epoch, 'cost =', '{:.3f}'.format(avg_cost))
print(sess.run(cost, feed_dict={x: x_test, y: y_test}))
