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
import matplotlib.colors as colors

# ==========================
#   Training Parameters
# ==========================
# Maximum episodes run
MAX_EPISODES = 50000
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

SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
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

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    print('checkpoint:', checkpoint)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # plot settings
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(env.map, aspect='auto', cmap='hot', vmin=0., vmax=1.5)
    plt.show(block=False)

    # Initialize noise
    ou_level = 0.
    epsilon = INITIAL_EPSILON

    rate = rospy.Rate(5)
    loop_time = time.time()
    last_loop_time = loop_time
    i = 0

    s1 = env.GetLaserObservation()
    s_1 = np.stack((s1, s1, s1), axis=1)

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
            s1 = env.GetLaserObservation()
            s_1 = np.append(np.reshape(s1, (LASER_BEAM, 1)), s_1[:, :(LASER_HIST - 1)], axis=1)
            s__1 = np.reshape(s_1, (LASER_BEAM * LASER_HIST))
            target1 = env.GetLocalTarget()
            speed1 = env.GetSelfSpeed()
            state1 = np.concatenate([s__1, speed1, target1], axis=0)
            [x, y, theta] =  env.GetSelfStateGT()
            map_img = env.RenderMap([[0, 0], env.target_point])
            
            r, terminal, result = env.GetRewardAndTerminate(j)
            ep_reward += r
            if j > 0 :
                buff.add(state, a[0], r, state1, terminal, switch_a_t)      #Add replay buffer
            j += 1
            state = state1

            a = actor.predict(np.reshape(state, (1, actor.s_dim)))
            switch_a = critic.predict_switch(np.reshape(state, (1, actor.s_dim)))
            switch_a_t = np.zeros([SWITCH])
            # Add exploration noise
            if i < NOISE_MAX_EP:
                ou_level = noise.ornstein_uhlenbeck_level(ou_level)
                a = a + ou_level


            if random.random() <= epsilon:
                print("----------Random Switch Action----------")
                switch_index = random.randrange(SWITCH)
                switch_a_t[switch_index] = 1
            else:
                switch_index = np.argmax(switch_a[0])
                switch_a_t[switch_index] = 1

            if switch_index == 1:
	            a = env.PIDController(action_bound)
	            ep_PID_count += 1.
	            print("-----------PID Controller---------------")

            # Set action for discrete and continuous action spaces
            action = a[0]
            if  action[0] <= 0.05:
                 action[0] = 0.05
            env.Control(action)

            # plot
            if j == 1:
                im.set_array(map_img)
                fig.canvas.draw()

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if buff.count() > MINIBATCH_SIZE:
                batch = buff.getBatch(MINIBATCH_SIZE)
                s_batch = np.asarray([e[0] for e in batch])
                a_batch = np.asarray([e[1] for e in batch])
                r_batch = np.asarray([e[2] for e in batch])
                s2_batch = np.asarray([e[3] for e in batch])
                t_batch = np.asarray([e[4] for e in batch])
                switch_a_batch = np.asarray([e[5] for e in batch])
                # y_i = np.asarray([e[1] for e in batch])

                # Calculate targets
                # critic
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))
                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # D3QN
                Q1 = critic.predict_switch(s2_batch)
                Q2 = critic.predict_target_switch(s2_batch)
                switch_y = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        switch_y.append(r_batch[k])
                    else:
                        switch_y.append(r_batch[k] + GAMMA * Q2[k, np.argmax(Q1[k])])
                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, switch_a_batch,\
                                                    np.reshape(y_i, (MINIBATCH_SIZE, 1)),\
                                                    switch_y)

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs, switch_a_batch)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            last_loop_time = loop_time
            loop_time = time.time()
            loop_time_buf.append(loop_time - last_loop_time)
            T += 1
            rate.sleep()

        # scale down epsilon
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / NOISE_MAX_EP

        summary = tf.Summary()
        summary.value.add(tag='Reward', simple_value=float(ep_reward))
        summary.value.add(tag='Qmax', simple_value=float(ep_ave_max_q / float(j)))
        summary.value.add(tag='PIDrate', simple_value=float(ep_PID_count / float(j)))
        summary.value.add(tag='Distance', simple_value=float(target_distance))
        summary.value.add(tag='Result', simple_value=float(result))
        summary.value.add(tag='Steps', simple_value=float(j))

        summary_writer.add_summary(summary, T)

        summary_writer.flush()

        if i > 0 and i % 1000 == 0 :
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = i) 

        print '| Reward: %.2f' % ep_reward, " | Episode:", i, \
        '| Qmax: %.4f' % (ep_ave_max_q / float(j)), \
        " | LoopTime: %.4f" % (np.mean(loop_time_buf)), " | Step:", j-1, '\n'
            


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