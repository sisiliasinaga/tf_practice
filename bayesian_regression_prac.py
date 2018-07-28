import tensorflow as tf
import numpy as np
import edward as ed
from edward.models import Normal
from sklearn import datasets
from sklearn.model_selection import train_test_split


def build_toy_dataset(N, w, noise_std=0.1):
    D = len(w)
    x = np.random.randn(N, D)
    y = np.dot(x, w) + np.random.normal(0, noise_std, size=N)
    return x, y


N = 40  # number of data points
D = 10  # number of features

w_true = np.random.randn(D)
x_train, y_train = build_toy_dataset(N, w_true)
x_test, y_test = build_toy_dataset(N, w_true)

x = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(x, w) + b, scale=tf.ones(N))

qw = Normal(loc=tf.get_variable("qw/loc", [D]), scale=tf.nn.softplus(tf.get_variable("qw/scale", [D])))
qb = Normal(loc=tf.get_variable("qb/loc", [1]), scale=tf.nn.softplus(tf.get_variable("qb/scale", [1])))

inference = ed.KLpq({w: qw, b: qb}, data={x: x_train, y: y_train})
inference.run(n_samples=5, n_iter=250)

y_post = ed.copy(y, {w: qw, b: qb})

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={x: x_test, y_post: y_test}))
