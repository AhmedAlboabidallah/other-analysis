import tensorflow as tf
import numpy as np


# Teach how to multiply
def generate_data(how_many):
    data = np.random.rand(how_many, 2)
    answers = data[:, 0] * data[:, 1]+8
    return data, answers


sess = tf.InteractiveSession()

input_data = tf.placeholder(tf.float32, shape=[None, 2])
correct_answers = tf.placeholder(tf.float32, shape=[None])

weights_1 = tf.Variable(tf.truncated_normal([2, 1], stddev=.1))
bias_1 = tf.Variable(.0)

output_layer = tf.matmul(input_data, weights_1) + bias_1

mean_squared = tf.reduce_mean(tf.square(correct_answers - tf.squeeze(output_layer)))
optimizer = tf.train.GradientDescentOptimizer(.1).minimize(mean_squared)

sess.run(tf.initialize_all_variables())

for i in range(1000):
    x, y = generate_data(100)
    sess.run(optimizer, feed_dict={input_data: x, correct_answers: y})

error = tf.reduce_sum(tf.abs(tf.squeeze(output_layer) - correct_answers))

x, y = generate_data(100)
print("Total Error: ", error.eval(feed_dict={input_data: x, correct_answers: y}))