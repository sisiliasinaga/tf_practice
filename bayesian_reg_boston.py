import tensorflow as tf
import edward as ed
from edward.models import Normal
from sklearn import datasets
from sklearn.model_selection import train_test_split

x_data, y_data = datasets.load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=42)

len_train = len(x_train)
len_test = len(x_test)
features = 13

x = tf.placeholder(tf.float32, [None, features])
w = Normal(loc=tf.zeros(features), scale=tf.ones(features))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(x, w) + b, scale=tf.ones(len_train))

qw = Normal(loc=tf.get_variable("qw/loc", [features]), scale=tf.nn.softplus(tf.get_variable("qw/scale", [features])))
qb = Normal(loc=tf.get_variable("qb/loc", [1]), scale=tf.nn.softplus(tf.get_variable("qb/scale", [1])))

inference = ed.KLpq({w: qw, b: qb}, data={x: x_train, y: y_train})
inference.run(n_samples=506, n_iter=250)

y_post = ed.copy(y, {w: qw, b: qb})

print("Mean squared error on training data:")
print(ed.evaluate('mean_squared_error', data={x: x_train, y_post: y_train}))
print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={x: x_test, y_post: y_test}))
