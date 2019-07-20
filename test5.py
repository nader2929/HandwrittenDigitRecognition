import tensorflow as tf
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#x is a placeholder for each images values
x = tf.placeholder("float", [None, 784])
#weights
weights = tf.Variable(tf.zeros([784,10]))
#biases
biases = tf.Variable(tf.zeros([10]))
#mulitply values with with weights and the biases
y = tf.nn.softmax(tf.matmul(x,weights)+biases)
#y_ will hold real values we are gonna use to train the NN
y_ = tf.placeholder("float", [None, 10])


def train():
    with tf.Session() as sess:
        c_entr = -tf.reduce_sum(y_*tf.log(y))
        training_step = tf.train.AdamOptimizer(0.001).minimize(c_entr)
        init = tf.initialize_all_variables()
        sess.run(init)
        #batch our training into 1000 batches of 100 images
        for i in range(6000):
            #set the x and y values for this batch
            b_x, b_y = mnist.train.next_batch(100)
            sess.run(training_step, feed_dict={x:b_x,y_:b_y})
        saver=tf.train.Saver()
        saver.save(sess, './test_model/testing.ckpt')

def test():
    with tf.Session() as sess1:
        n_save = tf.train.Saver()
        n_save.restore(sess1, './test_model/testing.ckpt')
        #print ("hi", y)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print (sess1.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def make_pred():
    Tk().withdraw()
    filename = askopenfilename()
    #the array that will hold the values of image
    image = np.zeros(784)
    #read image
    gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE )
    #resize image and set background to black
    gray = cv2.resize(gray, (28,28))
    cv2.imwrite(filename, gray)
    #making the image a one dimensional array of 784
    flatten = gray.flatten() / 255.0
    #saving the image and the correct value
    image = flatten

    with tf.Session() as sess2:
        n_save = tf.train.Saver()
        n_save.restore(sess2, './test_model/testing.ckpt')
        prediction=tf.argmax(y,1)
        print (prediction.eval(feed_dict={x: [image]}, session=sess2))


while (True):
    choice = input("Do you wanna train(1), test(2), make a prediction(3) or exit(4): ")
    if (choice == '1'):
        print("Training model")
        train()
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
