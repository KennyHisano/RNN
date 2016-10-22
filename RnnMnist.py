import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

s = tf.set_random_seed(1)

#load input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 0.001
training_iters = 100000
batch_size = 128
display_step = 10


n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
#we go through 28 * 28 times

y = tf.placeholder(tf.float32,[None,n_classes])
#x(,28,28)
#y(,10)
 

weights = {
 	'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
 	'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}

biases = {
	#(128,)
	'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
	#(10,)
	'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))

}


def RNN(X, weights,biases):


	#create input for cell by hidden layer
	#X ==> (128 batch*28 steps, 28 inputs)
	#we need to make 3d to 2d to caluculate
	X = tf.reshape(X,[-1,n_inputs])
	#into hidden
	#X_in = (128 batch * 28 steps, 128 hidden)
	X_in = tf.matmul(X, weights['in']) + biases['in']
	#X_in ==> (128 batch, 29 steps, 128 hidden)
	#make it back to 3d
	X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])


	#not it's time for lstm!
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
	#lstm cell is divided into two parts(c_state,h_state)
	_init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
	#forget_bias = 1.0 : dont forget for first input

	outputs,final_stete = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)
	#steps are in second param. so we set time major false
	#final_state for only the last one, output is for everyontput

	#results = tf.matmul(final_state[1],weights['out']) + biases['out']
	#[1] means the h_state

	#below is the same as results # its for testing
	outputs = tf.unpack(tf.transpose(outputs,[1,0,2]))
	results = tf.matmul(outputs[-1],weights['out']) + biases['out']

	return results








pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1









