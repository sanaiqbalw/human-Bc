import argparse
import gym
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


from run_expert import get_expert_data



def get_minibatch(expert_data, mb_size=128):
    observations = expert_data["observations"]
    actions = expert_data["actions"]
    indices = np.arange(observations.shape[0])
    np.random.shuffle(indices)
    mb_observations = observations[indices[:mb_size], :]
    mb_actions = actions[indices[:mb_size], :].squeeze()
    return mb_observations, mb_actions


class PolicyPredictor():

    def __init__(self, args,session):
        self.args=args
        env = gym.make(args.envname)
        self.sess = tf.InteractiveSession()
        self.sess=session
        self.obs_shape =env.observation_space.shape
        self.discrete_action_space = not hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape
        self.graph = self.build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())



    def build_mlp(self,inputplaceholder):
        with tf.variable_scope("policyPredictor"):
            self.layers = [tf.layers.dense(inputs= inputplaceholder, units=64, activation=tf.tanh, use_bias=True,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))]

            for ix in range(1, 3):
                new_layer = tf.layers.dense(inputs=self.layers[-1], units=64, activation=tf.tanh, use_bias=True,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

                self.layers.append(new_layer)
            # import pdb;pdb.set_trace()

            out= tf.layers.dense(inputs=self.layers[-1], units=self.act_shape[-1], activation=None, use_bias=True,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))


            return out

    def build_graph(self):
        """Creates graph of the neural net"""
        self.x= tf.placeholder(tf.float32, shape=[None, self.obs_shape[-1]])
        self.y= tf.placeholder(tf.float32, shape=[None, self.act_shape[-1]])
      
            # import pdb;pdb.set_trace()

        self.nn_policy_a = self.build_mlp(self.x)
        # Construct the loss function and training information.
        self.loss_op = tf.reduce_mean(tf.reduce_sum((self.nn_policy_a - self.y) * (self.nn_policy_a - self.y), axis=[1]))
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_step = tf.train.AdamOptimizer(self.args.bc_learning_rate).minimize(self.loss_op)


        return tf.get_default_graph()


    def train_bc_policy(self):
        env = gym.make(self.args.envname)
        expert_data, _ = get_expert_data(self.args.expert_policy_file, self.args.envname, self.args.max_timesteps, self.args.num_rollouts)
        print("Got rollouts from expert.")

        for i in range(self.args.bc_training_epochs):
            with self.graph.as_default():
                mb_obs, mb_acts = get_minibatch(expert_data, self.args.bc_minibatch_size)
                _, training_loss = self.sess.run([self.train_step, self.loss_op], feed_dict={self.x: mb_obs, self.y: mb_acts})
            print('BC LOSS:',i,training_loss)
        print("Expert policy cloned.")
        print("="*30)
        return self.nn_policy_a,self.x

    def train_policy_grad(self,args,session,reward):
        pass



    def predict_action(self, state):
        """Predict the action for each state"""
        with self.graph.as_default():
            action = self.sess.run(self.nn_policy, feed_dict={self.x:state})
        return action[0]
