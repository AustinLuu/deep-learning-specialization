import numpy as np
import tensorflow as tf

w = tf.Variable(0, dtype=tf.float32) # creating a variable w
cost = tf.add(tf.add(w**2, tf.multiply(-10.0, w)), 25.0) # can be written as this - cost = w**2 - 10*
train = tf.optimizers.SGD(learning_rate=0.1).minimize(cost, w,tape=tf.GradientTape())

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
session.run(w) # Runs the definition of w, if you print this it will print zero
session.run(train)

print("W after one iteration:", session.run(w))

for i in range(1000):
    session.run(train)

print("W after 1000 iterations:", session.run(w))
