#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy
import csv
from collections import defaultdict

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
H = 10
BATCH_SIZE = 100
#DROP_OUT_RATE = 0.5


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Input: x : 28*28=784
x = tf.placeholder(tf.float32, [None, 784])

# Variable: W, b1
W = weight_variable((784, H))
b1 = bias_variable([H])

# Hidden Layer: h
# softsign(x) = x / (abs(x)+1); https://www.google.co.jp/search?q=x+%2F+(abs(x)%2B1)
h = tf.nn.softmax(tf.matmul(x, W) + b1)
#keep_prob = tf.placeholder(tf.float32)
#h_drop = tf.nn.dropout(h, keep_prob)

# Variable: b2
W2 = tf.transpose(W)  # 転置
b2 = bias_variable([784])
y = tf.nn.relu(tf.matmul(h, W2) + b2)
#y = tf.nn.softmax(tf.matmul(h, W2) + b2)

# Define Loss Function
loss = tf.nn.l2_loss(y - x) / BATCH_SIZE
#loss = tf.nn.l2_loss(x*tf.log(y)) / BATCH_SIZE

#loss=-tf.reduce_sum(x*tf.log(y))

# For tensorboard learning monitoring
tf.scalar_summary("l2_loss", loss)
#tf.scalar_summary("W_loss", y)
# Use Adam Optimizer
#print ','.join(W)
train_step = tf.train.AdamOptimizer().minimize(loss)

# Prepare Session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
summary_writer = tf.train.SummaryWriter('summary/l2_loss', graph_def=sess.graph_def)

# Training
for step in range(40000):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs})
    # Collect Summary
    summary_op = tf.merge_all_summaries()
    summary_str = sess.run(summary_op, feed_dict={x: batch_xs})
    weight_values = sess.run(W)
    numpy.savetxt("weights.csv", weight_values, delimiter=",")
    summary_writer.add_summary(summary_str, step)
    # Print Progress
    if step % 100 == 0:
        print(loss.eval(session=sess, feed_dict={x: batch_xs}))

data = numpy.genfromtxt('weights.csv',delimiter=',', dtype = float)
i=0
while (i <=9) :

    plt.imshow(data[:,i].reshape((28, 28)), clim=(-1, 1.0), origin='upper')
    plt.savefig("Fig"+str(i)+".png")
    plt.show()
    i=i+1    
