import numpy as np
import tensorflow as tf
import gym

import matplotlib.pyplot as plt




class RewardPredictor():

    def __init__(self, args,session):
        # self.sess = tf.InteractiveSession(config=config)
        self.sess =session
        env = gym.make(args.envname)
        self.obs_shape = env.observation_space.shape
        self.discrete_action_space = not hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape
        # np.prod :product
        self.input_dim = np.prod(self.obs_shape) + np.prod(self.act_shape)
        self.graph = self.build_graph()
        self.sess.run(tf.global_variables_initializer())


    def build_mlp(self,input_placeholder,
                  scope,  output_size=1,n_layers=2,size=64,
                  activation=tf.tanh,output_activation=None):

        with tf.variable_scope(scope):
            layers = [tf.layers.dense(inputs=input_placeholder, units=size, activation=activation, use_bias=True,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))]

            for ix in range(1, n_layers):
                new_layer = tf.layers.dense(inputs=layers[-1], units=size, activation=activation, use_bias=True,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

                layers.append(new_layer)

            # output units =1  for each state-action one reward
            out = tf.layers.dense(inputs=layers[-1], units=output_size, activation=output_activation, use_bias=True,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

            return out



    def build_graph(self):
        self.input_ph=tf.placeholder(tf.float32, shape=[None, self.input_dim])
        self.human_labels= tf.placeholder(dtype=tf.int32, shape=[None,1], name="comparison_labels")


        # create  mlp for reward prediction and get  predicted rewards:
        self.reward_logits=self.build_mlp(self.input_ph, scope="rewardPredictor")

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.reward_logits, labels=self.human_labels)

        self.loss_op = tf.reduce_mean(loss)

        self.train_op = tf.train.AdamOptimizer(.001).minimize(tf.reduce_mean(self.loss_op))

        return tf.get_default_graph()


    def train_RF(self,labeled_comparisons,reward_iter):

        # minibatch_size = min(64, len(labeled_comparisons))
        # labeled_comparisons = np.random.sample(labeled_comparisons, minibatch_size)

        # These are trajectory segments rather than individual (state, action) pairs

        left_obs = np.asarray([comp['left']['obs'] for comp in labeled_comparisons])
        left_acts = np.asarray([comp['left']['actions'] for comp in labeled_comparisons])
        right_obs = np.asarray([comp['right']['obs'] for comp in labeled_comparisons])
        right_acts = np.asarray([comp['right']['actions'] for comp in labeled_comparisons])
        print(len(labeled_comparisons))
        labels = np.asarray([comp['label'] for comp in labeled_comparisons])
        print('shape of labels',labels.shape)
        print(labels)


        # todo:stack left and right observations and action:
        obs=tf.stack([ left_obs,right_obs], axis=1)
        acts=tf.stack([ left_acts,right_acts], axis=1)
        print(obs.shape,acts.shape)

        #todo: convert segments into individual action and rewards
        # 1. get segment length
        batchsize = tf.shape(left_obs)[0]
        segment_length = tf.shape(left_obs)[1]


        # 2. chop up segments into individual observations and actions
        obs = tf.reshape(obs, (-1,) + self.obs_shape)
        acts = tf.reshape(acts, (-1,) + self.act_shape)




        labels=tf.reshape(labels, (-1,) + (1,1))
        print(obs.shape,acts.shape)

        # flatten observatiion array and append actions to amke input for reward function
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/flatten

        acts  = tf.cast(acts ,tf.float64)
        flat_obs = tf.contrib.layers.flatten(obs)
        input_ = tf.concat((flat_obs, acts), axis=1)
        print(input_.shape)
        input_=input_.eval()


        # using action values of segments get 
        reward_logit=self.predict_reward(input_)

        labels=labels.eval()
      


        # self.input_ph=tf.placeholder(tf.float32, shape=[None, self.input_dim])
        # self.output_ph= tf.placeholder(dtype=tf.int32, shape=(None,), name="comparison_labels")


        # # create  mlp for reward prediction and get  predicted rewards:
        # reward_logits=self.build_mlp(self.input_ph, scope="rewardPredictor")

        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reward_logits, labels=self.output_ph)

        # self.loss_op = tf.reduce_mean(loss)

        # self.train_op = tf.train.AdamOptimizer(.001).minimize(loss_op)


        for i in range(self.args.reward_iter):
            with self.graph.as_default():
                 _, loss = self.sess.run([self.train_op, self.loss_op],feed_dict={self.output_ph:labels,self.input_ph: input_})
        print(" Reward Function trained with one dataset.")

        return 


    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
        with self.graph.as_default():
            reward = self.sess.run(self.reward_logits, feed_dict={self.input_ph:path})
        return reward[0]





