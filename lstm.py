import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
ops.reset_default_graph()

with open('C:/Users/yuiichiiros/Downloads/full_db_mask_norm3_PCA40_folds.csv') as csvfile:
    readCSV = pd.read_csv(csvfile)

pca1 = readCSV.PCA1
pca2 = readCSV.PCA2
pca3 = readCSV.PCA3
pca4 = readCSV.PCA4
pca5 = readCSV.PCA5
pca6 = readCSV.PCA6
pca7 = readCSV.PCA7
pca8 = readCSV.PCA8
pca9 = readCSV.PCA9
pca10 = readCSV.PCA10
pca11 = readCSV.PCA11
pca12 = readCSV.PCA12
pca13 = readCSV.PCA13
pca14 = readCSV.PCA14
pca15 = readCSV.PCA15
pca16 = readCSV.PCA16
pca17 = readCSV.PCA17
pca18 = readCSV.PCA18
pca19 = readCSV.PCA19
pca20 = readCSV.PCA20
pca21 = readCSV.PCA21
pca22 = readCSV.PCA22
pca23 = readCSV.PCA23
pca24 = readCSV.PCA24
pca25 = readCSV.PCA25
pca26 = readCSV.PCA26
pca27 = readCSV.PCA27
pca28 = readCSV.PCA28
pca29 = readCSV.PCA29
pca30 = readCSV.PCA30
pca31 = readCSV.PCA31
pca32 = readCSV.PCA32
pca33 = readCSV.PCA33
pca34 = readCSV.PCA34
pca35 = readCSV.PCA35
pca36 = readCSV.PCA36
pca37 = readCSV.PCA37
pca38 = readCSV.PCA38
pca39 = readCSV.PCA39
pca40 = readCSV.PCA40

data = np.asarray([pca1, pca2, pca3, pca4, pca5, pca6, pca7, pca8, pca9, pca10, pca11, pca12, pca13, pca14, pca15,
                   pca16, pca17, pca18, pca19, pca20, pca21, pca22, pca23, pca24, pca25, pca26, pca27, pca28, pca29,
                   pca30, pca31, pca32, pca33, pca34, pca35, pca36, pca37, pca38, pca39, pca40])

# define constants
time_steps = 28
num_units = 128
# n_input = 40 ??
learning_rate = 0.001
# n_classes = ??
batch_size = 128

out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

x = tf.placeholder("float", [None, time_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

input = tf.unstack(x, time_steps, 1)

lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

# converting last output of dimension [batch_size, num_units] to
# [batch_size, n_classes] by out_weight multiplication
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

mse = tf.losses.mean_squared_error(prediction, y)
my_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse)

init = tf.global_variables_initializer()

accuracy = tf.reduce_mean(tf.square(tf.subtract(prediction, y)))

with tf.Session() as sess:
    sess.run(init)
    iter = 1
    while iter < 800:
