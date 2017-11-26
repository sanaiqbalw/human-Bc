import numpy as np
import tensorflow as tf
from envs import get_timesteps_per_episode


def _slice_path(path, segment_length, start_pos=0):
    return {
        k: np.asarray(v[start_pos:(start_pos + segment_length)])
        for k, v in path.items()
        if k in ['obs', "actions", 'original_rewards', 'human_obs']}


def create_segment_q_states(segment):
    obs_Ds = segment["obs"]
    act_Ds =segment["actions"]
    # print(obs_Ds.shape,act_Ds.shape)
    return np.concatenate([obs_Ds, act_Ds], axis=1)


def sample_segment_from_path(path, segment_length):
    """Returns a segment sampled from a random place in a path. Returns None if the path is too short"""
    path_length = len(path["obs"])
    if path_length < segment_length:
        return None

    # Modification : Moving start_pos to 10 to weed out start dead beats
    start_pos = np.random.randint(10, path_length - segment_length + 1)

    # Build segment
    segment = _slice_path(path, segment_length, start_pos)

    # Add q_states
    segment["q_states"] = create_segment_q_states(segment)
    return segment


def do_rollout(env, policy_fn, sy_ob, session):
    """ Builds a path by running through an environment using a provided function to select actions. """
    obs, rewards, actions, human_obs = [], [], [], []
    max_timesteps_per_episode = get_timesteps_per_episode(env)
    ob = env.reset()
    # Primary environment loop
    for i in range(max_timesteps_per_episode):
        action = session.run(policy_fn, feed_dict={sy_ob: ob[None]})[0]
        try:
            action=np.squeeze(ac,axis=(0))
        except:
            pass
        # action = policy_fn(env, ob)
        obs.append(ob)
        actions.append(action)
        ob, rew, done, info = env.step(action)
        rewards.append(rew)
        human_obs.append(info.get("human_obs"))
        if done:
            break
    # Build path dictionary
    path = {
        "obs": np.array(obs),
        "original_rewards": np.array(rewards),
        "actions": np.array(actions),
        "human_obs": np.array(human_obs)}
    return path


def segments_from_rollout(env_id, make_env, policy_fn, sy_ob, session, n_desired_segments, clip_length_in_seconds):
    """ Generate a list of path segments by doing rollouts. """
    segments = []
    env = make_env(env_id)

    segment_length = int(clip_length_in_seconds * env.fps)
    while len(segments) < n_desired_segments:
        path = do_rollout(env, policy_fn, sy_ob, session)
        # Calculate the number of segments to sample from the path
        # Such that the probability of sampling the same part twice is fairly low.
        segments_for_this_path = max(1, int(0.25 * len(path["obs"]) / segment_length))
        for _ in range(segments_for_this_path):
            segment = sample_segment_from_path(path, segment_length)
            if segment:
                segments.append(segment)

            if len(segments) % 50 == 0 and len(segments) > 0:
                print("Collected %s/%s segments" % (len(segments), n_desired_segments))

    print("Successfully collected %s segments" % (len(segments)))
    return segments
