import tensorflow as tf
import pandas as pd
import edward as ed
from edward.models import Normal

readCSV = pd.read_csv('C:/Users/yuiichiiros/Downloads/adjusted_file_na2o.csv')

x_data = readCSV[['312.979', '334.54599', '335.79401', '818.99371', '819.80206']]
y_data = readCSV['Na2O']

x_data_np = x_data.as_matrix()
y_data_np = y_data.as_matrix()

x_train = x_data_np[:1042]
x_test = x_data_np[1042:]
y_train = y_data_np[:1042]
y_test = y_data_np[1042:]

n_samples = len(readCSV['312.979']) / 5
n_samples = int(n_samples)

len_train = len(x_train)
len_test = len(x_test)
features = 5

x = tf.placeholder(tf.float32, [None, features])

w_1 = Normal(loc=tf.zeros([features, features]), scale=tf.ones([features, features]))
w_2 = Normal(loc=tf.zeros([features, features]), scale=tf.ones([features, features]))
w_out = Normal(loc=tf.zeros([features, 1]), scale=tf.ones([features, 1]))

b_1 = Normal(loc=tf.zeros(features), scale=tf.ones(features))
b_2 = Normal(loc=tf.zeros(features), scale=tf.ones(features))
b_out = Normal(loc=tf.zeros(1), scale=tf.ones(1))


def bayesian_1(data):
    layer1 = tf.tanh(tf.matmul(data, w_1) + b_1)
    layer2 = tf.tanh(tf.matmul(layer1, w_2) + b_2)
    output = tf.tanh(tf.matmul(layer2, w_out) + b_out)
    return tf.reshape(output, [1042, 1])


y1 = bayesian_1(x)
qw_1 = Normal(loc=tf.get_variable("qw_1/loc", [features, features]),
              scale=tf.nn.softplus(tf.get_variable("qw_1/scale", [features, features])))
qb_1 = Normal(loc=tf.get_variable("qb_1/loc", [features]),
              scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [features])))
qw_2 = Normal(loc=tf.get_variable("qw_2/loc", [features, features]),
              scale=tf.nn.softplus(tf.get_variable("qw_2/scale", [features, features])))
qb_2 = Normal(loc=tf.get_variable("qb_2/loc", [features]), scale=tf.nn.softplus(tf.get_variable("qb_2/scale",
                                                                                                [features])))
qw_out = Normal(loc=tf.get_variable("qw_out/loc", [features, 1]),
                scale=tf.nn.softplus(tf.get_variable("qw_out/scale", [features, 1])))
qb_out = Normal(loc=tf.get_variable("qb_out/loc", [1]), scale=tf.nn.softplus(tf.get_variable("qb_out/scale", [1])))

inference = ed.KLpq({w_1: qw_1, b_1: qb_1, w_2: qw_2, b_2: qb_2, w_out: qw_out, b_out: qb_out},
                    data={x: x_train, y1: y_train})
inference.run(n_samples=n_samples, n_iter=500)

y_post1 = ed.copy(y1, {w_1: qw_1, b_1: qb_1, w_2: qb_2, b_2: qb_2, w_out: qw_out, b_out: qb_out})

print("Mean squared error on training data:")
print(ed.evaluate('mean_squared_error', data={x: x_train, y_post1: y_train}))
print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={x: x_test, y_post1: y_test}))
