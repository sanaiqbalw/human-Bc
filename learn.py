import argparse
import gym
import numpy as np
import tensorflow as tf
from behavioral_cloning import run_cloning
from envs import make_with_torque_removed, get_timesteps_per_episode
from rollouts import segments_from_rollout
from comparisons import ComparisonsCollector
from rewards_predictor import ComparisonRewardPredictor

CLIP_LENGTH = 1.5


def get_tf_session():
    """ Returning a session. """
    tf.reset_default_graph()
    session = tf.Session()
    return session


def get_bc_policy(args, session):
    # Returns a trained BC policy
    bc_policy = run_cloning(args, session)
    return bc_policy


def teach(session, args):
    pretrain_labels = args.pretrain_labels
    num_timesteps = int(args.num_timesteps)

    # Step 1 : Get the Behavior cloned policy
    bc_policy, sy_ob = get_bc_policy(args, session)

    current_policy = bc_policy
    for iteration in range(args.num_iters):
        # Step 2 : Generate rollouts with this policy and Sample segments from these rollouts
        pretrain_segments = segments_from_rollout(args.envname, make_with_torque_removed,
                                                  bc_policy, sy_ob, session,
                                                  n_desired_segments=pretrain_labels * 2,
                                                  clip_length_in_seconds=CLIP_LENGTH)

        # Step 3 : Instantiate comparisons
        collector = ComparisonsCollector(pretrain_segments,pretrain_labels, iteration,
                                         make_with_torque_removed(args.envname))
        collector.process_comparisons()

        # Step 4 : Serialize the sampled segments and wait for user input
        # Run Flask server on the side and get user inputs
        # Once all inputs are received serialize the segments with labels
        # On user input deserialize segments with labels

        is_labelling_done = input("Enter Y when done with labelling, else don't do anything.")
        if is_labelling_done.upper() == "Y":
            labeled_comparisons = collector.collect_comparison_labels()
            # print(labeled_comparisons)

        # Step 5 : Train rewards predictor with data collected in Step 4
            nn_reward=ComparisonRewardPredictorr(env=args.envname)
            nn_reward.train_RF(labeled_comparisons,args.reward_iter)
            new_policy=Pg(args.env,nn_reward)

            # Step 6 : Use the reward predictor learned in Step 5 and update the policy
            # Let's try policy gradients for this step
        print("Done with the loop")
    print("Done with the training")


def parse_args():
# python learn.py --expert_policy_file experts/Hopper-v1.pkl --envname Hopper-v1 --num_rollouts 3 --bc_training_epochs 10 --pretrain_labels 2 --num_iters 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str, default="experts/Hopper-v1.pkl")
    parser.add_argument('--envname', type=str, default="Hopper-v1")
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=25,
                        help='Number of expert roll outs')
    parser.add_argument('--bc_training_epochs', type=int, default=2001,
                        help='Number of training epochs for behavioral cloning')
    parser.add_argument('--bc_minibatch_size', type=int, default=10000,
                        help='Batch size for behavioral cloning')
    parser.add_argument('--bc_learning_rate', type=float, default=0.001,
                        help='Learning Rate for behavioral cloning')
    parser.add_argument('--bc_check_every', type=int, default=500,
                        help='Check Performance in this many epochs for behavioral cloning')
    parser.add_argument('--pretrain_labels', default=200, type=int,
                        help='Number of labels to ask from human')
    parser.add_argument('--num_iters', default=5, type=int,
                        help='Number of iterations')
    parser.add_argument('--num_timesteps', default=5e6, type=int)
    parser.add_argument('--reward_iter', type=int, default=10,
                    help='number of iterations for training reward under one labelling set')
    parser.add_argument('--policy_iter', type=int, default=10,
                    help='number of iterations for training policy under one reward function')



    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    session = get_tf_session()
    # Seed the generator
    np.random.seed(7)
    tf.set_random_seed(11)
    teach(session, args)


if __name__ == '__main__':
    main()
