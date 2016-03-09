#Program by Ajay Krishna Teja Kavuri
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#Importing the tensorflow library
import tensorflow as tf
sess=tf.InteractiveSession()



#-----------------Interactive Model----------------------------------
#x is just an variable that represents the input
#None means any dimensionality
x=tf.placeholder(tf.float32,[None,784])
#Let's create the "modifiable" weights and bias as variables
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
#Initiate the session
sess.run(tf.initialize_all_variables())
#Generte the output 
out=tf.nn.softmax(tf.matmul(x,w)+b)
#Generate the delta for weight corrections using cross entropy
cOut=tf.placeholder(tf.float32,[None,10])
cross_entropy=-tf.reduce_sum(cOut*tf.log(out))
#Automatically does the backpropogation
#We try to minimize the cross_entropy using 0.01 learning rate
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#Train the process
for i in range(1000):
	batch_xs, batch_ys=mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={x:batch_xs,cOut:batch_ys})
#Compare the output 
correct_prediction=tf.equal(tf.argmax(out,1),tf.argmax(cOut,1))
#Derive the accuracy
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
#Print the results
print accuracy.eval(feed_dict={x: mnist.test.images, cOut: mnist.test.labels})

#-----------------Function----------------------------------
#Initialize the variables
#init=tf.initialize_all_variables()
#Launch the session
#sess=tf.Session()
#sess.run(init)
#Loop the training
#for i in range(1000):
	#batch_xs, batch_ys=mnist.train.next_batch(100)
	#sess.run(train_step,feed_dict={x:batch_xs,cOut:batch_ys})
#Compare the output 
#correct_prediction=tf.equal(tf.argmax(out,1),tf.argmax(cOut,1))
#Derive the accuracy
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
#Print the results
#print sess.run(accuracy, feed_dict={x: mnist.test.images, cOut: mnist.test.labels})