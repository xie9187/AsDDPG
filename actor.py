# ===========================
#   Actor DNN
# ===========================
import tensorflow as tf

# Network Parameters - Hidden layers
n_hidden_1 = 256
n_hidden_2 = 256
n_hidden_3 = 256

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 2D input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.03, shape=shape)
    return tf.Variable(initial)

def conv1d(x, W, stride):
    return tf.nn.conv1d(x, W, stride = stride, padding = "SAME")

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op by applying gradient, variable pairs
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tf.placeholder(tf.float32, [None, self.s_dim])

        # FC1 
        w_fc1 = weight_variable([self.s_dim, n_hidden_1])
        b_fc1 = bias_variable([n_hidden_1])
        # FC2
        w_fc2 = weight_variable([n_hidden_1, n_hidden_2])
        b_fc2 = bias_variable([n_hidden_2])
        # FC3
        w_fc3 = weight_variable([n_hidden_2, n_hidden_3])
        b_fc3 = bias_variable([n_hidden_3])
        # Out 0
        w_out_0 = weight_variable([n_hidden_3, self.a_dim/2])
        b_out_0 = bias_variable([self.a_dim/2])
        # Out 1
        w_out_1 = weight_variable([n_hidden_3, self.a_dim/2])
        b_out_1 = bias_variable([self.a_dim/2])

        h_fc1 = tf.nn.relu(tf.matmul(inputs, w_fc1) + b_fc1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3)

        # Run sigmoid on output 0 to get 0 to 1
        out_0 = tf.nn.sigmoid(tf.matmul(h_fc3, w_out_0) + b_out_0)
        # Run tanh on output 1 to get -1 to 1
        out_1 = tf.nn.tanh(tf.matmul(h_fc3, w_out_1) + b_out_1)
        out = tf.concat([out_0, out_1], axis=1)
        scaled_out = tf.multiply(out, self.action_bound)  # Scale output to -action_bound to action_bound
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient,
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs,
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs,
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
