# implement of Restricted Boltzmann Machine for Collaborative Filtering
# Paper can be found at: http://www.machinelearning.org/proceedings/icml2007/papers/407.pdf

import tensorflow as tf
import numpy as np
from sklearn.utils import gen_batches

class RBM_CF(object):

	def __init__(self, visible_dim, hidden_dim = 256, rating_dim = 5, 
		learning_rate = 0.01, train_epoch = 250, k_gibbs = 1, batch_size = 128):

		self.visible_dim = visible_dim
		self.hidden_dim = hidden_dim
		self.rating_dim = rating_dim
		self.learning_rate = learning_rate
		self.train_epoch = train_epoch
		self.k_gibbs = k_gibbs
		self.batch_size = batch_size
		self.update = None
		self.loss = None
		self.batch = None	# [N, R, V]

		abs_val = abs(-4 * np.sqrt(6. / (self.visible_dim + self.hidden_dim)))
		self.W = tf.get_variable('W', [self.rating_dim, self.visible_dim, self.hidden_dim],		# [R, V, H]
			tf.float32, initializer = tf.random_uniform_initializer(minval = -abs_val, maxval = abs_val))	
		self.W_tile = tf.tile(tf.reshape(self.W, [1, self.rating_dim, self.visible_dim, self.hidden_dim]),
			[self.batch_size, 1, 1, 1])		# [N, R, V, H]

		self.v_bias = tf.get_variable('v_bias', [self.rating_dim, self.visible_dim], tf.float32,
			initializer = tf.constant_initializer(0.0))
		self.h_bias = tf.get_variable('h_bias', [1, self.hidden_dim], tf.float32,
			initializer = tf.constant_initializer(0.0))


	def train(self, train_set, val_set):
		# train_set: [T, V]
		# val_set: [S, V]
		init = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(self.train_epoch):
				batches = gen_batches(train_set.shape[0], self.batch_size)
				total_error = 0.0

				for b in batches:
					self.batch = self._tran(train_set[b])
					self._update_params()
					result = sess.run([self.update, self.loss])
					total_error += result[1]

				print("Cost at epoch %s of training: %s" % (epoch, total_error / self.batch_size))

	def _tran(self, data):
		# data: [N, V]
		return tf.transpose(tf.one_hot(data - 1, self.rating_dim), [0, 2, 1])	# [N, R, V]

	def _sample(self, prob):
		return tf.nn.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))

	def _sample_h_given_v(self, v_0):
		# v_0: [N, R, V]
		# return [N, 1, H]
		v_0_reshape = tf.expand_dims(v_0, axis = 2)		# [N, R, 1, H]
		h_prob_temp = tf.matmul(v_0_reshape, self.W_tile)	# [N, R, 1, H]
		h_prob = tf.sigmoid(tf.reduce_sum(h_prob_temp, axis = 1) + self.h_bias)	# [N, 1, H]
		h_sample = self._sample(h_prob)
		return h_prob, h_sample

	def _sample_v_given_h(self, h_sample):
		# h_sample: [N, 1, H]
		# return [N, R, V]
		h_sample_tile = tf.tile(h_sample, [1, self.rating_dim, 1])	# [N, R, H]
		h_sample_tile = tf.expand_dims(h_sample_tile, axis = 2)		# [N, R, 1, H]
		v_prob_temp = tf.reduce_sum(tf.matmul(h_sample_tile, tf.transpose(self.W_tile, 
			perm = [0, 1, 3, 2])), axis = 2) + self.v_bias		# [N, R, V]
		v_prob_temp *= tf.sign(self.batch)
		v_prob = tf.transpose(tf.nn.softmax(tf.transpose(v_prob_temp, [0, 2, 1])), [0, 2, 1])	# [N, R, V]
		v_sample = self._sample(v_prob)
		return v_prob, v_sample

	def _gibbs_step(self, v_0):
		# v_0: [N, R, V]
		v_init = v_0
		v_prob = None
		v_sample = None
		h_prob_0 = None
		h_prob_1 = None
		positive = None
		for step in range(self.k_gibbs):
			# positive phase
			h_prob, h_sample = self._sample_h_given_v(v_init)	# [N, 1, H]
			if step == 0:
				h_prob_0 = tf.reduce_mean(h_prob, axis = 0)		# [1, H]
				positive = self._calculate(v_init, h_prob)
				# positive = self._calculate(v_init, h_sample)
			# negative phase
			v_prob, v_sample = self._sample_v_given_h(h_sample)		# [N, R, V]
			v_init = v_prob
		h_prob_1, h_sample_1 = self._sample_h_given_v(v_prob)	# [N, 1, H]
		negative = self._calculate(v_prob, h_prob_1)
		# positive, negative: [R, V, H]
		# h_prob_0, h_prob_1: [N, 1, H]
		# v_prob, v_sample: [N, R, V]
		return positive, negative, h_prob_0, h_prob_1, v_prob, v_sample

	def _calculate(self, v, h):
		# v: [N, R, V]
		# h: [N, 1, H]
		# return [R, V, H]
		v_tile = tf.expand_dims(v, axis = 3)	# [N, R, V, 1]
		h_tile = tf.tile(tf.expand_dims(h, axis = 1), [1, self.rating_dim, 1, 1])	# [N, R, 1, H]
		return tf.reduce_mean(tf.matmul(v_tile, h_tile), axis = 0)		# [R, V, H]

	def _update_params(self):
		positive, negative, h_prob_0, h_prob_1, v_prob, v_sample = self._gibbs_step(self.batch)
		W_update = self.W.assign_add(self.learning_rate * (positive - negative))
		h_bias_update = self.h_bias.assign_add(self.learning_rate * 
			tf.reduce_mean(h_prob_0 - h_prob_1, axis = 0))
		v_bias_update = self.v_bias.assign_add(self.learning_rate * 
			tf.reduce_mean(self.batch - v_prob, axis = 0))
		v_bias_update = self.v_bias.assign_add(self.learning_rate * 
			tf.reduce_mean(self.batch - v_sample, axis = 0))
		self.update = [W_update, h_bias_update, v_bias_update]
		self.loss = tf.reduce_mean(tf.abs(tf.subtract(self.batch, v_sample)))
		# self.loss = tf.reduce_mean(tf.square(self.batch - v_sample))


















