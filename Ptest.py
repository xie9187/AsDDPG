# ================================================
# Modified from the work of Patrick Emami:
#       Implementation of DDPG - Deep Deterministic Policy Gradient
#       Algorithm and hyperparameter details can be found here:
#           http://arxiv.org/pdf/1509.02971v2.pdf
#
# Removed TFLearn dependency
# Added Ornstein Uhlenbeck noise function
# Added reward discounting
# Works with discrete actions spaces (Cartpole)
# Tested on CartPole-v0 & -v1 & Pendulum-v0
# Author: Liam Pettigrew
# =================================================
import tensorflow as tf
import numpy as np
import rospy
import random
import time
import copy

from StageWorld import StageWorld
from ReplayBuffer import ReplayBuffer
from noise import Noise
from reward import Reward
from actor import ActorNetwork
from critic import CriticNetwork

import matplotlib.pyplot as plt
import pylab as plt
import matplotlib.colors as colors

# ==========================
#   Training Parameters
# ==========================
# Maximum episodes run
MAX_EPISODES = 1000
# Max episode length
MAX_EP_STEPS = 500
# Episodes with noise
NOISE_MAX_EP = 1000
# Noise parameters - Ornstein Uhlenbeck
DELTA = 0.5 # The rate of change (time)
SIGMA = 0.5 # Volatility of the stochastic processes
OU_A = 3. # The rate of mean reversion
OU_MU = 0. # The long run average interest rate
# E-gready
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.001 # starting value of epsilon

# Reward parameters
REWARD_FACTOR = 0.1 # Total episode reward factor
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

LASER_BEAM = 40
LASER_HIST = 3
ACTION = 2
TARGET = 2
SPEED = 2
SWITCH = 2

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'CartPole-v0' # Discrete: Reward factor = 0.1
#ENV_NAME = 'CartPole-v1' # Discrete: Reward factor = 0.1
#ENV_NAME = 'Pendulum-v0' # Continuous: Reward factor = 0.01
# Directory for storing gym results
MONITOR_DIR = './results/' + ENV_NAME
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 20000
MINIBATCH_SIZE = 32

GAME = 'StageWorld'

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic, noise, reward, discrete, action_bound):
    # Set up summary writer
    summary_writer = tf.summary.FileWriter("ddpg_summary")

    # plot settings
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(env.map, aspect='auto', cmap='hot', vmin=0., vmax=1.5)
    plt.show(block=False)

    rate = rospy.Rate(5)
    loop_time = time.time()
    last_loop_time = loop_time
    i = 0

    T = 0
    for i in range(MAX_EPISODES):
        env.ResetWorld()
        env.GenerateTargetPoint()
        print 'Target: (%.4f, %.4f)' % (env.target_point[0], env.target_point[1])
        target_distance = copy.deepcopy(env.distance)
        ep_reward = 0.
        ep_ave_max_q = 0.
        loop_time_buf = []
        terminal = False

        j = 0
        ep_reward = 0
        ep_ave_max_q = 0
        ep_PID_count = 0.
        while not terminal and not rospy.is_shutdown():
            target1 = env.GetLocalTarget()
            [x, y, theta] =  env.GetSelfStateGT()
            map_img = env.RenderMap([[0, 0], env.target_point])
            
            r, terminal, result = env.GetRewardAndTerminate(j)
            ep_reward += r
            j += 1

            a = env.PIDController(action_bound)
            ep_PID_count += 1.

            action = a[0]
            if  action[0] <= 0.05:
                 action[0] = 0.05
            env.Control(action)

            # plot
            if j == 1:
                im.set_array(map_img)
                fig.canvas.draw()

            last_loop_time = loop_time
            loop_time = time.time()
            loop_time_buf.append(loop_time - last_loop_time)
            T += 1
            rate.sleep()

        summary = tf.Summary()
        summary.value.add(tag='Distance/Time', simple_value=float(target_distance/float(j)/0.2))
        summary.value.add(tag='Distance', simple_value=float(target_distance))
        summary.value.add(tag='Steps', simple_value=float(j))


        summary_writer.add_summary(summary, T)

        summary_writer.flush()


        print 'Episode:',i ,'| Reward: %.4f' % float(target_distance/float(j)/0.2)

            


def main(_):
    with tf.Session() as sess:
        env = StageWorld(LASER_BEAM)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        state_dim = LASER_BEAM * LASER_HIST + SPEED + TARGET

        action_dim = ACTION
        action_bound = [0.5, np.pi/3]
        switch_dim = SWITCH

        discrete = False
        print('Continuous Action Space')

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
        						ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim, switch_dim,
        						CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
        reward = Reward(REWARD_FACTOR, GAMMA)

        try:
            train(sess, env, actor, critic, noise, reward, discrete, action_bound)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    tf.app.run()