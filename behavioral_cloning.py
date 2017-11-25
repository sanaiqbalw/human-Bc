'''
parse
get session
mlp
run cloning
get reward

'''
import argparse

import gym
import numpy as np
import tensorflow as tf

from run_expert import get_expert_data




def get_tf_session():
    """ Returning a session. """
    tf.reset_default_graph()
    session = tf.Session()
    return session


def get_minibatch(expert_data, mb_size=128):
    observations = expert_data["observations"]
    actions = expert_data["actions"]
    indices = np.arange(observations.shape[0])
    np.random.shuffle(indices)
    mb_observations = observations[indices[:mb_size], :]
    mb_actions = actions[indices[:mb_size], :].squeeze()
    return mb_observations, mb_actions


def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        output_activation=None):
    with tf.variable_scope(scope):
        layers = [tf.layers.dense(inputs=input_placeholder, units=size, activation=activation, use_bias=True,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))]

        for ix in range(1, n_layers):
            new_layer = tf.layers.dense(inputs=layers[-1], units=size, activation=activation, use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

            layers.append(new_layer)

        out = tf.layers.dense(inputs=layers[-1], units=output_size, activation=output_activation, use_bias=True,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        return out


def run_cloning(args, session):
    env = gym.make(args.envname)
    expert_data, _ = get_expert_data(args.expert_policy_file, args.envname, args.max_timesteps, args.num_rollouts)
    print("Got rollouts from expert.")
    expert_obs = expert_data["observations"]
    expert_acts = expert_data["actions"]
    x = tf.placeholder(tf.float32, shape=[None, expert_obs.shape[-1]])
    y = tf.placeholder(tf.float32, shape=[None, expert_acts.shape[-1]])
    nn_policy = build_mlp(x, expert_acts.shape[-1], scope="bc")

    # # Save weights as a single vector to make saving/loading easy.
    # weights_bc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='clone_net')
    # weight_vector = tf.concat([tf.reshape(w, [-1]) for w in weights_bc], axis=0)

    # Construct the loss function and training information.
    l2_loss = tf.reduce_mean(
        tf.reduce_sum((nn_policy - y) * (nn_policy - y), axis=[1])
    )
    train_step = tf.train.AdamOptimizer(args.bc_learning_rate).minimize(l2_loss)

    loss_over_epochs = []
    epochs = []
    returns_over_epochs = []
    session.run(tf.global_variables_initializer())

    for i in range(args.bc_training_epochs):
        mb_obs, mb_acts = get_minibatch(expert_data, args.bc_minibatch_size)
        _, training_loss = session.run([train_step, l2_loss], feed_dict={x: mb_obs, y: mb_acts})

        # act = session.run(nn_policy, feed_dict={x: mb_obs})

        # if i % args.bc_check_every == 0:
        #     returns = get_rewards(args, session, nn_policy, x, env)
        #     print("Iteration :", i, "Training Loss", training_loss)
        #     print("mean(returns): {}\nstd(returns): {}\n".format(np.mean(returns), np.std(returns)))
        #     epochs.append(i)
        #     loss_over_epochs.append(training_loss)
        #     returns_over_epochs.append(returns)
    # mean_return_over_epochs = list(map(lambda x: np.mean(x), returns_over_epochs))
    print("Expert policy cloned.")
    print("="*30)
    return nn_policy, x


def get_expert_labels(args, session, nn_policy, policy_fn, x, env):
    expert_actions = []
    observations = []
    returns = []
    max_steps = env.spec.timestep_limit

    for _ in range(args.num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0
        steps = 0
        while not done:
            # Take steps by expanding observation (to get shapes to match).
            exp_obs = np.expand_dims(obs, axis=0)
            action = np.squeeze(session.run(nn_policy, feed_dict={x: exp_obs}))
            obs, r, done, _ = env.step(action)
            totalr += r
            observations.append(obs)
            steps += 1
            if args.render: env.render()
            if steps >= max_steps: break
        returns.append(totalr)
    with tf.Session():
        for obs in observations:
            expert_actions.append(policy_fn(obs[None, :]))

    return np.array(observations), np.array(expert_actions)


def get_rewards(args, session, nn_policy, x, env):
    actions = []
    observations = []
    returns = []
    max_steps = env.spec.timestep_limit

    for _ in range(args.num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0
        steps = 0
        while not done:
            # Take steps by expanding observation (to get shapes to match).
            exp_obs = np.expand_dims(obs, axis=0)
            action = np.squeeze(session.run(nn_policy, feed_dict={x: exp_obs}))
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render: env.render()
            if steps >= max_steps: break
        returns.append(totalr)

    return returns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str, default="experts/Hopper-v1.pkl")
    parser.add_argument('--envname', type=str, default="Hopper-v1")
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=25,
                        help='Number of expert roll outs')
    parser.add_argument('--bc_training_epochs', type=int, default=2001,
                        help='Number of training epochs')
    parser.add_argument('--bc_minibatch_size', type=int, default=10000,
                        help='Batch size')
    parser.add_argument('--bc_learning_rate', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--bc_check_every', type=int, default=500,
                        help='Check Performance in this many epochs')

    parser.add_argument('--dagger_steps', type=int, default=10,
                        help='Number of dagger steps')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    session = get_tf_session()
    # Seed the generator
    np.random.seed(7)
    tf.set_random_seed(11)

    run_cloning(args, session)


if __name__ == "__main__":
    main()
