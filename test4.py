import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder("float", [None, 784])
#weights
weights = tf.Variable(tf.zeros([784,10]))
#biases
biases = tf.Variable(tf.zeros([10]))
#mulitply values with with weights and the biases
y = tf.nn.softmax(tf.matmul(x,weights)+biases)
#y_ will hold real values we are gonna use to train the NN
y_ = tf.placeholder("float", [None, 10])
#minimising the cross entropy function imporves our model
c_entr = -tf.reduce_sum(y_*tf.log(y))
#low lerning rate to minimise cross entropy
training_step = tf.train.GradientDescentOptimizer(0.02).minimize(c_entr)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#batch our training into 1000 batches of 100 images
for i in range(1000):
    #set the x and y values for this batch
    b_x, b_y = mnist.train.next_batch(100)
    sess.run(training_step, feed_dict={x:b_x,y_:b_y})

correct = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
acc = tf.reduce_mean(tf.cast(correct, "float"))
print( sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
