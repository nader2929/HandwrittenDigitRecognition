import tensorflow as tf
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
import os
from scipy import ndimage
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

n_classes = 10
batch_size = 100

n_nodes_hl1 = 784
n_nodes_hl2 = 784
n_nodes_hl3 = 784

x= tf.placeholder('float', [None, 784] , 'x')
y= tf.placeholder('float')

def neural_network_model(data):

    hidden1Layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden2Layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden3Layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    outputLayer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}
    l1 = tf.add(tf.matmul(data, hidden1Layer['weights']), hidden1Layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden2Layer['weights']), hidden2Layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden3Layer['weights']), hidden3Layer['biases'])
    l3 = tf.nn.relu(l3)
    ol = tf.matmul(l3, outputLayer['weights']) + outputLayer['biases']
    return ol


def train(x):
    prediction = neural_network_model(x)
    cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    opt = tf.train.AdamOptimizer().minimize(cost)
    epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            e_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                e_x,e_y=mnist.train.next_batch(batch_size)
                _, c = sess.run([opt,cost], feed_dict={x: e_x, y: e_y})
                e_loss += c
            print('Epoch', e+1, 'completed out of', epochs,' loss: ', e_loss)
        corr = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        acc = tf.reduce_mean(tf.cast(corr, 'float'))
        print('Acc: ', acc.eval({x:mnist.test.images, y:mnist.test.labels}))


while (True):
    train(x)