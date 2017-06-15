import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.15 + 0.35 + 0.03*np.random.rand(100).astype(np.float32)

Weights = tf.Variable(tf.random_uniform([1],-0.1,0.1))
bias = tf.Variable(tf.zeros([1]))

y = Weights * x_data + bias
loss = tf.reduce_mean(tf.square(y - y_data))
opt = tf.train.AdamOptimizer(0.3)
train = opt.minimize(loss)
init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	eps = 1e-8
	for t in range(500):
		w_old=np.array([sess.run(Weights),sess.run(bias)])
		sess.run(train)
		if t%20==0:
			print("Step ",t,sess.run(Weights),sess.run(bias))
		w_new = np.array([sess.run(Weights),sess.run(bias)])
		if np.max(np.abs(w_old - w_new))<eps:
			print("Step ",t,sess.run(Weights),sess.run(bias))
			break

