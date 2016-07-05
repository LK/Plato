import logging
import tensorflow as tf

# Weighting of entropy term in policy loss
BETA = 0.01

# TODO: The packet protocol uses float16, but the neural net uses float32
# throughout. Need to check if everything we're using in the net supports
# float16 operations (https://github.com/tensorflow/tensorflow/issues/1300)

class Network(object):

	def __init__(self, state_dims=4, layers=[1024,1024,1024,1024], actions=4):
		self._layers = layers
		self._state_dims = state_dims
		self._actions = actions

	# Create the base network that value and policy networks build off of
	# Expected inputs from feed_dict: state (matrix of size [batch, state_dims])
	def base_network(self):
		if self._base_network != None:
			return self._base_network

		# Initialize the base network
		prev = tf.Placeholder(tf.float32, shape=(None, self._state_dims),
													name='state')

		for i in range(len(self._layers)):
			with tf.variable_scope('fc%d' % i):
				# Create the ReLU layer and append it to _base_network. Note that the
				# input layer is never added to the network because it is passed in
				# feed_dict.
				prev = relu(prev, self._layers[i])

				if i == len(self._layers)-1:
					self._base_network = prev

		logging.debug('Initialized base network')

		return self._base_network

	# Expected inputs from feed_dict: state (matrix of size [batch, state_dims])
	def value_network(self):
		if self._value_network != None:
			return self._value_network

		self._value_network = linear(self.base_network(), 1)

		logging.debug('Initialized value network')

		return self._value_network

	# Expected inputs from feed_dict: state (matrix of size [batch, state_dims])
	def policy_network(self):
		if self._policy_network != None:
			return self._policy_network

		self._policy_network = softmax(self.base_network(), self._actions)

		logging.debug('Initialized policy network')

		return self._policy_network

	# Expected inputs from feed_dict:
	# - state (matrix of size [batch, state_dims])
	# - action (matrix of size [batch, actions], each row is one-hot encoded)
	# - reward (vector of length batch)
	#
	# NOTE: The action is encoded as a scalar over the network, but it must be
	# passed to the loss as a one-hot vector
	def loss(self):
		if self._loss != None:
			return self._loss

		s = tf.Placeholder(tf.float32, shape=(None, self._state_dims), name='state')
		a = tf.Placeholder(tf.float32, shape=(None, self._actions), name='action')
		r = tf.Placeholder(tf.float32, shape=(None, 1), name='reward')

		# TODO: Try halving LR on value net?
		value_error = tf.nn.l2_loss(r - self._value_network)

		# Computes the entropy of the policy
		entropy = -tf.reduce_sum(self._policy_network *
				tf.log(self._policy_network), reduction_indices=1)

		# Computes the log of the selected action
		policy_error = -tf.reduce_sum(tf.log(self._policy_network * a),
				reduction_indices=1) * (r - self._value_network) + BETA * entropy

		self._loss = value_error + policy_error
		return self._loss

	# Create and return a linear layer
	@staticmethod
	def linear(input_tensor, size):
		# Initialize weights using Xavier initialization
		W = tf.get_variable('weights', [input_tensor.get_shape()[1], size],
												initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable('biases', [size],
												initializer=tf.constant_initializer(0.0))

		return tf.add(tf.matmul(input_tensor, W), b)

	# Create and return a ReLU layer
	@staticmethod
	def relu(input_tensor, size):
		# Initialize weights using Xavier initialization and a slight positive bias
		W = tf.get_variable('weights', [input_tensor.get_shape()[1], size],
												initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable('biases', [size],
												initializer=tf.constant_initializer(0.1))

		return tf.nn.relu(tf.add(tf.matmul(input_tensor, W), b))

	# Create and return a softmax layer
	@staticmethod
	def softmax(input_tensor, size):
		# Initialize weights using Xavier initialization
		W = tf.get_variable('weights', [input_tensor.get_shape()[1], size],
												initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable('biases', [size],
												initializer=tf.constant_initializer(0.0))

		return tf.nn.softmax(tf.add(tf.matmul(input_tensor, W), b))
