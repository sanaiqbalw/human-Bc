import argparse
import gym
import numpy as np
import tensorflow as tf

# import matplotlib.pyplot as plt


from run_expert import get_expert_data
from rollouts import do_rollout
from envs import get_timesteps_per_episode



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
        self.env = gym.make(args.envname)
        env=self.env
        self.sess = tf.InteractiveSession()
        self.sess=session
        self.obs_shape =env.observation_space.shape
        self.discrete_action_space = not hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape
        self.graph = self.build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    
    def policy_rollout(self,reward_fn):
        paths = []
        env=self.env

        for itr in range(0,self.args.num_rollouts):
            

            # Run policy rollouts to get sample
            ob = env.reset()
            obs,acs, rewards = [], [], []
            horizon=int(self.args.num_timesteps)

            for hor in range(0,horizon):

                obs.append(ob)
                ac=self.get_pg_action(ob[None])
                try:
                    ac=np.squeeze(ac,axis=(0))
                except:
                    pass
                rew0=reward_fn.predict_reward(ob,ac)

                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew0)
                if done or  hor>horizon:
                    break

            path = {"observation" : np.array(obs),
                    "reward" : np.array(rewards),
                    "action" : np.array(acs)}
            paths.append(path)

        print('GENERATED POLICY ROLLOUT')
        return paths


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
        env=self.env
        """Creates graph of the neural net"""
        self.x= tf.placeholder(tf.float32, shape=[None, self.obs_shape[-1]])
        self.y= tf.placeholder(tf.float32, shape=[None, self.act_shape[-1]])
        self.reward = tf.placeholder(tf.float32, [None])

            # import pdb;pdb.set_trace()

        self.nn_policy_a = self.build_mlp(self.x)
        # bc-clone
        # Construct the loss function and training information.
        self.loss_op = tf.reduce_mean(tf.reduce_sum((self.nn_policy_a - self.y) * (self.nn_policy_a - self.y), axis=[1]))
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_step = tf.train.AdamOptimizer(self.args.bc_learning_rate).minimize(self.loss_op,global_step=tf.contrib.framework.get_global_step())
        

        # _____policy gradient==discrete___:

        # action_mask = tf.one_hot(self.nn_policy_a, int(self.act_shape[-1]), 1,0)
        # self.action_prob = tf.nn.softmax(self.nn_policy_a)
        # self.action_value_pred = tf.reduce_sum(self.action_prob * action_mask, 1)

        # self.loss_pg = tf.reduce_mean(-tf.log(self.action_value_pred) * self.target)
        # # self.optimizer = tf.train.AdamOptimizer(learning_rate=.001)
        # # # self.train_op_pg = self.optimizer.minimize(self.pg_loss, global_step=tf.contrib.framework.get_global_step())

        # self.train_op_pg = tf.train.AdamOptimizer(self.args.bc_learning_rate).minimize(self.loss_pg)


        # _____policy gradient==continuous___:
        # we  assume normal distribution on actions find 
        # sigma and mean of distribution

        self.mu = tf.squeeze(self.nn_policy_a)
        self.sigma = tf.exp(tf.get_variable(name='variance',shape=[self.act_shape[-1]], initializer=tf.zeros_initializer()))
        # logstd should just be a trainable variable, not a network output.
        self.sigma = tf.nn.softplus(self.sigma) + 1e-5
        self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.action = self.normal_dist._sample_n(1)
    
        self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])
      
        # Loss and train op

        self.loss_pg = -tf.reduce_mean(self.normal_dist.log_prob(self.action) * self.reward)
        self.train_op_pg = tf.train.AdamOptimizer(self.args.bc_learning_rate).minimize(self.loss_pg)

        return tf.get_default_graph()


    def train_bc_policy(self):
        env = self.env
        expert_data, _ = get_expert_data(self.args.expert_policy_file, self.args.envname, self.args.max_timesteps, self.args.num_rollouts)
        print("Got rollouts from expert.")

        for i in range(self.args.bc_training_epochs):
            with self.graph.as_default():
                mb_obs, mb_acts = get_minibatch(expert_data, self.args.bc_minibatch_size)
                _, training_loss = self.sess.run([self.train_step, self.loss_op], feed_dict={self.x: mb_obs, self.y: mb_acts})
            if i%100==0:
                print('BC LOSS:',i,training_loss)
        print("Expert policy cloned.")
        print("="*30)

        return self.nn_policy_a,self.x

    def train_policy_grad(self,reward_fn):
        env = gym.make(self.args.envname)
        paths=self.policy_rollout(reward_fn)
        mb_obs = np.concatenate([path["observation"] for path in paths])
        mb_acts = np.concatenate([path["action"] for path in paths])
        reward_obtained = [path["reward"].sum() for path in paths]
        print(mb_obs.shape,mb_acts.shape)


        total_timesteps = 0
        epoch_loss_reward_all=[]
        for i in range(self.args.policy_iter):
            timesteps_this_batch = 0
            epoch_loss_reward=[]
            cost=0
            for path in range(len(paths)):
                with self.graph.as_default():
                    _, training_loss = self.sess.run([self.train_op_pg , self.loss_pg], feed_dict={self.x: mb_obs, self.y: mb_acts,self.reward:reward_obtained})
                cost=np.mean(training_loss)
            print('policy gradient Mean Loss:',i,cost)
        print("finished policy gradient.")
        print("="*30)
        return self.nn_policy_a,self.x
        


    def predict_action_bc(self, state):
        """Predict the action for each state"""
        with self.graph.as_default():
            ac = self.sess.run(self.nn_policy_a, feed_dict={self.x:state})
        return ac

    def get_pg_action(self, state):
        with self.graph.as_default():
            action = self.sess.run(self.action,feed_dict={self.x:state})
        return action
         


