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
#classes and size of batches
n_classes = 10
batch_size = 100
#nodes in each layer
n_nodes_hl1 = 500
n_nodes_hl2 = 800
n_nodes_hl3 = 950
#placeholder values in the variables that will be used for the data and labels
x= tf.placeholder('float', [None, 784] , 'x')
y= tf.placeholder('float')

def neural_network_model(data):
    #initiating the layers with random weights and biases
    #1st layer has input size and then its number of nodes as an output size
    hidden1Layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    #2nd layer input size is the output size of the 1st layer and output size as defined above
    hidden2Layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    #3rd layer input size is the output size of the 2nd layer and output size as defined above
    hidden3Layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    #output layer input size is the output size of the 3rd layer and output size 10 (number of classes)
    outputLayer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}
    #multiply the raw input data with their weights and add the biases
    l1 = tf.add(tf.matmul(data, hidden1Layer['weights']), hidden1Layer['biases'])
    #an activation function that will return a value if the output produced by a node for an input is above or equal to 0
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden2Layer['weights']), hidden2Layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden3Layer['weights']), hidden3Layer['biases'])
    l3 = tf.nn.relu(l3)
    ol = tf.matmul(l3, outputLayer['weights']) + outputLayer['biases']
    #return values of possible outputs
    return ol


def train(x):
    #define model we are gonna train
    prediction = neural_network_model(x)
    #cost vairable to track mistakes of model during each epoch
    cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    #define the optimiser to minimise the cost of the model
    opt = tf.train.AdamOptimizer().minimize(cost)
    #number of epochs model needs to complete before training is over
    epochs = 10
    with tf.Session() as sess:
        #initialise variables
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            e_loss=0
            #train on the entire dataset in bathes of previously defined size
            for _ in range(int(mnist.train.num_examples/batch_size)):
                #save batch data and labels in variables
                e_x,e_y=mnist.train.next_batch(batch_size)
                #make predicition and see if it was wrong or write and optimise the cost and reconfigure the model were appropriate
                _, c = sess.run([opt,cost], feed_dict={x: e_x, y: e_y})
                e_loss += c
            print('Epoch', e+1, 'completed out of', epochs,' loss: ', e_loss)
        #save model
        saver=tf.train.Saver()
        saver.save(sess, './nn_test/nnmodel')

def test():
    #deince shape of model to be loaded for testing
    prediction = neural_network_model(x)
    n_save = tf.train.Saver()
    with tf.Session() as sess3:
        n_save.restore(sess3, './nn_test/nnmodel')
        #the model is correct if its prediction matches the label
        corr = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        #calculate the accuracy of the model during testing
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
    gray = cv2.resize(gray, (28,28))
    #making the image a one dimensional array of size 784
    flatten = gray.flatten() / 255.0
    #saving the image and the correct value
    image = flatten
    prediction = neural_network_model(x)
    n_save = tf.train.Saver()
    with tf.Session() as sess2:
        n_save.restore(sess2, './nn_test/nnmodel')
        prediction=tf.argmax(prediction,1)
        print ("Value seems to be: ", prediction.eval(feed_dict={x: [image]}, session=sess2), "according to the NN")



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
