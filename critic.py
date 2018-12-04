# ===========================
#   Critic DNN
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
        x:           Tensor, 4D BHWD input maps
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

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, switch_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.switch_dim = switch_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.switch_a, self.out, self.switch_q, self.value = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_switch_a, self.target_out, self.target_switch_q, self.target_value = \
            self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        self.predicted_switch_q = tf.placeholder(tf.float32, [None])

        # Define switch loss
        self.readout_action = tf.reduce_sum(tf.multiply(self.switch_q, self.switch_a), axis=1)
        self.td_error = tf.square(self.predicted_switch_q - self.readout_action)
        self.switch_loss = tf.reduce_mean(self.td_error)
        # Define loss and optimization Op
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predicted_q_value, self.out))))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss + self.switch_loss)

        # Get the gradient of the net w.r.t. the action
        
        # using advantage!
        # indicies = tf.constant(0)
        # self.advantage = tf.subtract(self.out, tf.gather(tf.transpose(self.readout_action), indicies))
        self.advantage = tf.subtract(self.out, self.value)
        self.action_grads = tf.gradients(self.advantage, self.action)

        # using Q value
        # self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        action = tf.placeholder(tf.float32, [None, self.a_dim])
        switch_a = tf.placeholder(tf.float32, [None, self.switch_dim])

        # FC1 10x64
        w_fc1 = weight_variable([self.s_dim, n_hidden_1])
        b_fc1 = bias_variable([n_hidden_1])
        # FC2
        w_fc2 = weight_variable([n_hidden_1 + self.a_dim, n_hidden_2])
        b_fc2 = bias_variable([n_hidden_2])
        # FC2_switch
        w_fc2_switch = weight_variable([n_hidden_1, n_hidden_2])
        b_fc2_switch = bias_variable([n_hidden_2])
        # FC3
        w_fc3 = weight_variable([n_hidden_2, n_hidden_3])
        b_fc3 = bias_variable([n_hidden_3])
        # FC3 adv
        w_fc3_adv = weight_variable([n_hidden_2, n_hidden_3/2])
        b_fc3_adv = bias_variable([n_hidden_3/2])
        # FC3 critic
        w_fc3_value = weight_variable([n_hidden_2, n_hidden_3/2])
        b_fc3_value = bias_variable([n_hidden_3/2])
        # Out
        w_out = weight_variable([n_hidden_3, 1])
        b_out = bias_variable([1])
        # Out
        w_out_adv = weight_variable([n_hidden_3/2, self.switch_dim])
        b_out_adv = bias_variable([self.switch_dim])
        # Out
        w_out_value = weight_variable([n_hidden_3/2, 1])
        b_out_value = bias_variable([1])

        h_fc1 = tf.nn.relu(tf.matmul(inputs, w_fc1) + b_fc1)
        # critic
        h_fc1_a = tf.concat([h_fc1, action], axis=1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_a, w_fc2) + b_fc2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3)
        # D3QN
        h_fc2_switch = tf.nn.relu(tf.matmul(h_fc1, w_fc2_switch) + b_fc2_switch)
        h_fc3_adv = tf.nn.relu(tf.matmul(h_fc2_switch, w_fc3_adv) + b_fc3_adv)
        h_fc3_value = tf.nn.relu(tf.matmul(h_fc2_switch, w_fc3_value) + b_fc3_value)

        # Critic
        out = tf.matmul(h_fc3, w_out) + b_out
        # D3QN
        out_adv = tf.matmul(h_fc3_adv, w_out_adv) + b_out_adv
        out_value = tf.matmul(h_fc3_value, w_out_value) + b_out_value

        advAvg = tf.expand_dims(tf.reduce_mean(out_adv, axis=1), axis=1)
        advIdentifiable = tf.subtract(out_adv, advAvg)
        q_out = tf.add(out_value, advIdentifiable)

        return inputs, action, switch_a, out, q_out, out_value

    def train(self, inputs, action, switch_a, predicted_q_value, predicted_switch_q):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.switch_a: switch_a,
            self.predicted_q_value: predicted_q_value,
            self.predicted_switch_q: predicted_switch_q
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action,
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
        })

    def action_gradients(self, inputs, actions, switch_a):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.switch_a: switch_a
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def predict_switch(self, inputs):
        return self.sess.run(self.switch_q, feed_dict={
            self.inputs: inputs
        })

    def predict_target_switch(self, inputs):
        return self.sess.run(self.target_switch_q, feed_dict={
            self.target_inputs: inputs
        })    

