
import os
import glob
import os.path as osp
import random
from collections import deque
from time import time, sleep
import gym

import numpy as np
import tensorflow as tf
from keras import backend as K

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, LeakyReLU
from keras.models import Sequential



class FullyConnectedMLP(object):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, h_size=64):
        input_dim = np.prod(obs_shape) + np.prod(act_shape)

        self.model = Sequential()
        self.model.add(Dense(h_size, input_dim=input_dim))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(h_size))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

    def run(self, obs, act):
        flat_obs = tf.contrib.layers.flatten(obs)
        x = tf.concat([flat_obs, act], axis=1)
        return self.model(x)


class RewardPredictor():
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self, args,session):
        
        self.args=args
        self.sess =session
        env = gym.make(args.envname)
        self.sess = session
        self.obs_shape = env.observation_space.shape
        self.discrete_action_space = not hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape
        self.graph = self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def _predict_rewards(self, obs_segments, act_segments, network):
        """
        :param obs_segments: tensor with shape = (batch_size, segment_length) + obs_shape
        :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
        :param network: neural net with .run() that maps obs and act tensors into a (scalar) value tensor
        :return: tensor with shape = (batch_size, segment_length)
        """
        batchsize = tf.shape(obs_segments)[0]
        segment_length = tf.shape(obs_segments)[1]

        # Temporarily chop up segments into individual observations and actions
        obs = tf.reshape(obs_segments, (-1,) + self.obs_shape)
        acts = tf.reshape(act_segments, (-1,) + self.act_shape)

        # Run them through our neural network
        rewards = network.run(obs, acts)

        # Group the rewards back into their segments
        return tf.reshape(rewards, (batchsize, segment_length))

    def _build_model(self):
        """
        Our model takes in path segments with states and actions, and generates Q values.
        These Q values serve as predictions of the true reward.
        We can compare two segments and sum the Q values to get a prediction of a label
        of which segment is better. We then learn the weights for our model by comparing
        these labels with an authority (either a human or synthetic labeler).
        """
        # Set up observation placeholders
        self.segment_obs_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.obs_shape, name="obs_placeholder")
        self.segment_alt_obs_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.obs_shape, name="alt_obs_placeholder")

        self.segment_act_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.act_shape, name="act_placeholder")
        self.segment_alt_act_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.act_shape, name="alt_act_placeholder")


        # A vanilla multi-layer perceptron maps a (state, action) pair to a reward (Q-value)
        mlp = FullyConnectedMLP(self.obs_shape, self.act_shape)

        self.q_value = self._predict_rewards(self.segment_obs_placeholder, self.segment_act_placeholder, mlp)
        alt_q_value = self._predict_rewards(self.segment_alt_obs_placeholder, self.segment_alt_act_placeholder, mlp)

        # We use trajectory segments rather than individual (state, action) pairs because
        # video clips of segments are easier for humans to evaluate
        segment_reward_pred_left = tf.reduce_sum(self.q_value, axis=1)
        segment_reward_pred_right = tf.reduce_sum(alt_q_value, axis=1)
        reward_logits = tf.stack([segment_reward_pred_left, segment_reward_pred_right], axis=1)  # (batch_size, 2)

        self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="comparison_labels")

        # delta = 1e-5
        # clipped_comparison_labels = tf.clip_by_value(self.comparison_labels, delta, 1.0-delta)

        data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reward_logits, labels=self.labels)

        self.loss_op = tf.reduce_mean(data_loss)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op, global_step=global_step)

        return tf.get_default_graph()

    def predict_rewards_path(self, path):
        """Predict the reward for each step in a given path"""
        with self.graph.as_default():
            q_value = self.sess.run(self.q_value, feed_dict={
                self.segment_obs_placeholder: np.asarray([path["obs"]]),
                self.segment_act_placeholder: np.asarray([path["actions"]]),
                K.learning_phase(): False
            })
        return q_value[0]



    def predict_reward(self, ob,ac):
        
        ob=ob.reshape(1,1, self.obs_shape[-1])
        ac=ac.reshape(1,1, self.act_shape[-1])
        """Predict the reward for each step in a given path"""
        with self.graph.as_default():
            q_value = self.sess.run(self.q_value, feed_dict={
                self.segment_obs_placeholder: np.asarray(ob),
                self.segment_act_placeholder: np.asarray(ac),
                K.learning_phase(): False
            })
        return q_value[0]


   
    def train_RF(self,labeled_comparisons):
        left_obs = np.asarray([comp['left']['obs'] for comp in labeled_comparisons])
        left_acts = np.asarray([comp['left']['actions'] for comp in labeled_comparisons])
        right_obs = np.asarray([comp['right']['obs'] for comp in labeled_comparisons])
        right_acts = np.asarray([comp['right']['actions'] for comp in labeled_comparisons])
        labels = np.asarray([comp['label'] for comp in labeled_comparisons])
        for name in glob.glob('Human_Preference/static/Collections/run[0-9]+'):
            print('REMOVING ########',name)
            os.remove(name)
 
        for i in range(self.args.reward_iter):
            with self.graph.as_default():
                _, loss = self.sess.run([self.train_op, self.loss_op], feed_dict={
                    self.segment_obs_placeholder: left_obs,
                    self.segment_act_placeholder: left_acts,
                    self.segment_alt_obs_placeholder: right_obs,
                    self.segment_alt_act_placeholder: right_acts,
                    self.labels: labels,
                    K.learning_phase(): True


                })
                print("Reward training loss:..",i,loss)
        print(" Reward Function trained with one dataset.")



   