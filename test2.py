######################### import stuff ##########################
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

######################## prepare the data ########################
X, y = load_linnerud(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)

######################## set learning variables ##################
learning_rate = 0.0005
epochs = 2000
batch_size = 3

######################## set some variables #######################
x = tf.placeholder(tf.float32, [None, 3], name='x')  # 3 features
y = tf.placeholder(tf.float32, [None, 3], name='y')  # 3 outputs

# hidden layer 1
W1 = tf.Variable(tf.truncated_normal([3, 10], stddev=0.03), name='W1')
b1 = tf.Variable(tf.truncated_normal([10]), name='b1')

# hidden layer 2
W2 = tf.Variable(tf.truncated_normal([10, 3], stddev=0.03), name='W2')
b2 = tf.Variable(tf.truncated_normal([3]), name='b2')

######################## Activations, outputs ######################
# output hidden layer 1
hidden_out = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

# total output
y_ = tf.nn.relu(tf.add(tf.matmul(hidden_out, W2), b2))

####################### Loss Function  #########################
mse = tf.losses.mean_squared_error(y, y_)

####################### Optimizer      #########################
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)

###################### Initialize, Accuracy and Run #################
# initialize variables
init_op = tf.global_variables_initializer()

# accuracy for the test set
accuracy = tf.reduce_mean(tf.square(tf.subtract(y, y_)))  # or could use tf.losses.mean_squared_error

# run
with tf.Session() as sess:
  sess.run(init_op)
  total_batch = int(len(y_train) / batch_size)
  for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
      batch_x, batch_y = X_train[i * batch_size:min(i * batch_size + batch_size, len(X_train)), :], \
                         y_train[i * batch_size:min(i * batch_size + batch_size, len(y_train)), :]
      _, c = sess.run([optimizer, mse], feed_dict={x: batch_x, y: batch_y})
      avg_cost += c / total_batch
    if epoch % 10 == 0:
      print('Epoch:', (epoch + 1), 'cost =', '{:.3f}'.format(avg_cost))
  print(sess.run(mse, feed_dict={x: X_test, y: y_test}))
