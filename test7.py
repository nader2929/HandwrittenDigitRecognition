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

n_nodes_hl1 = 500
n_nodes_hl2 = 800
n_nodes_hl3 = 950

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
        saver=tf.train.Saver()
        saver.save(sess, './nn_test/nnmodel')

def test():
    prediction = neural_network_model(x)
    n_save = tf.train.Saver()
    with tf.Session() as sess3:
        n_save.restore(sess3, './nn_test/nnmodel')
        corr = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        acc = tf.reduce_mean(tf.cast(corr, 'float'))
        print('Acc: ', acc.eval({x:mnist.test.images, y:mnist.test.labels}))

def make_pred():
    Tk().withdraw()
    filename = askopenfilename()
    #the array that will hold the values of image
    image = np.zeros(784)
    #read image
    gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #resize image and set background to black
    gray = cv2.resize(255-gray, (28,28))
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        # first cols than rows
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted



    #making the image a one dimensional array of 784
    flatten = gray.flatten() / 255.0
    #saving the image and the correct value
    image = flatten
    prediction = neural_network_model(x)
    n_save = tf.train.Saver()
    with tf.Session() as sess2:
        n_save.restore(sess2, './nn_test/nnmodel')
        prediction=tf.argmax(prediction,1)
        print ("Value seems to be: ", prediction.eval(feed_dict={x: [image]}, session=sess2), "according to the NN")

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

while (True):
    choice = input("Do you wanna train(1), test(2), make a prediction(3) or exit(4): ")
    if (choice == '1'):
        print("Training model")
        train(x)
    elif (choice == '2'):
        print("Testing model")
        test()
    elif (choice == '3'):
        print("Going to make a prediction on your image")
        make_pred()
    elif (choice == '4'):
        print("Exiting")
        break
    else:
        print ("Invalid entry")
