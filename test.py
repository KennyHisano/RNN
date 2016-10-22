import tensorflow as tf


lr = 0.001
training_itrs = 100000
batch_size = 128
display_step = 10


n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
#we go through 28 * 28 times
y = tf.placeholder(tf.float32,[None,n_classes])

#print(x)
#print(y)

#https://www.tensorflow.org/versions/r0.11/api_docs/python/array_ops.html
x = tf.reshape(x,[-1,n_inputs])
#print(x)
