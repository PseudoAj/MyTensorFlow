import tensorflow as tf
tf_session = tf.Session()
x = tf.constant(1)
y = tf.constant(1)
tf_session.run(x + y)