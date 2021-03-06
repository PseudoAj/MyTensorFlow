#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
H = 120
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

#Classify
Wo=tf.Variable(tf.zeros([784,10]))
bo=tf.Variable(tf.zeros([10]))

out=tf.nn.softmax(tf.matmul(y,Wo)+bo)
cOut=tf.placeholder(tf.float32,[None,10])

cross_entropy=-tf.reduce_sum(cOut*tf.log(out))
# Define Loss Function
#loss = tf.nn.l2_loss(y - x) / BATCH_SIZE
#loss = tf.nn.l2_loss(x*tf.log(y)) / BATCH_SIZE

#loss=-tf.reduce_sum(x*tf.log(y))

# For tensorboard learning monitoring
tf.scalar_summary("l2_loss",cross_entropy )

# Use Adam Optimizer
#train_step = tf.train.AdamOptimizer().minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Prepare Session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
summary_writer = tf.train.SummaryWriter('summary/l2_loss', graph_def=sess.graph_def)

# Training
for step in range(60000):
    batch_xs, batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,cOut:batch_ys})
#Compare the output 
correct_prediction=tf.equal(tf.argmax(out,1),tf.argmax(cOut,1))
#Derive the accuracy
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
#Print the results
print sess.run(accuracy, feed_dict={x: mnist.test.images, cOut: mnist.test.labels})
"""
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs})
    # Collect Summary
    summary_op = tf.merge_all_summaries()
    summary_str = sess.run(summary_op, feed_dict={x: batch_xs})
    summary_writer.add_summary(summary_str, step)
    # Print Progress
    if step % 100 == 0:
        print(loss.eval(session=sess, feed_dict={x: batch_xs}))
#orrect_prediction=tf.equal(tf.argmax(y,1),tf.argmax(x,1))
#Derive the accuracy
#accuracy=(tf.cast(correct_prediction,"float"))
#accuracy=correct_prediction
#Print the results
#print sess.run(accuracy, feed_dict={x: mnist.test.images})

# Draw Encode/Decode Result
N_COL = 10
N_ROW = 2
plt.figure(figsize=(N_COL, N_ROW*2.5))
batch_xs, _ = mnist.train.next_batch(N_COL*N_ROW)
for row in range(N_ROW):
    for col in range(N_COL):
        i = row*N_COL + col
        data = batch_xs[i:i+1]

        # Draw Input Data(x)
        plt.subplot(2*N_ROW, N_COL, 2*row*N_COL+col+1)
        plt.title('IN:%02d' % i)
        #plt.imshow(data.reshape((28, 28)), cmap=magma, clim=(0, 1.0), origin='upper')
        plt.imshow(data.reshape((28, 28)), clim=(0, 1.0), origin='upper')
        #plt.imshow(data.reshape((28, 28)))
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")

        # Draw Output Data(y)
        plt.subplot(2*N_ROW, N_COL, 2*row*N_COL + N_COL+col+1)
        plt.title('OUT:%02d' % i)
        y_value = y.eval(session=sess, feed_dict={x: data})
        #plt.imshow(y_value.reshape((28, 28)), cmap=magma, clim=(0, 1.0), origin='upper')
        plt.imshow(y_value.reshape((28, 28)), clim=(0, 1.0), origin='upper')
        #plt.imshow(y_value.reshape((28, 28)))
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")

plt.savefig("result.png")
plt.show()
"""