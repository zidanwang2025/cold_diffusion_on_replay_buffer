import os
import collections
import numpy as np
import gymnasium as gym
import pdb
import h5py

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# with suppress_output():
#     ## d4rl prints out a variety of warnings
#     import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps

    # TEMP
    env.spec.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env, validation=False):
    # dataset = env.get_dataset(normalize=True)
    dataset = env.get_dataset(normalize=True, single_goal=False, fixed_start=False, validation=validation)
    return dataset

def sequence_dataset(env, preprocess_fn, validation=False, data_path=None, observation_dim=4, action_dim=2, patch_dataset=True, dataset_size='1.6k'):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    if data_path is None:
        # dataset = get_dataset(env, validation=validation)
        dataset = env.get_dataset(normalize=True, single_goal=False, fixed_start=False, validation=validation, dataset_size=dataset_size)
        dataset = preprocess_fn(dataset)

        N = dataset['rewards'].shape[0]
        data_ = collections.defaultdict(list)

        # The newer version of the dataset adds an explicit
        # timeouts field. Keep old method for backwards compatability.
        use_timeouts = 'timeouts' in dataset

        episode_step = 0

        for i in range(N):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == env.spec.max_episode_steps - 1)

            for k in dataset:
                if 'metadata' in k: continue
                data_[k].append(dataset[k][i])

            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])

                if patch_dataset and len(episode_data['observations']) > 1:
                    # Due to a bug, the last state is being random, so overwrite it with the one before.
                    episode_data['observations'][-1] = episode_data['observations'][-2]

                yield episode_data
                data_ = collections.defaultdict(list)

            episode_step += 1
    else:
        with h5py.File(data_path, "r") as f:
            all_obs = f['obs'][()]
            all_actions = f['actions'][()]
            for i in range(len(all_obs)):
                episode_data = {}
                episode_data["observations"] = all_obs[i][:, :observation_dim]
                episode_data["actions"] = all_actions[i][:, :action_dim]
                episode_data["terminals"] = np.zeros(len(all_obs[i]), dtype=bool)
                episode_data["terminals"][-1] = True
                episode_data['rewards'] = np.random.rand(len(all_obs[i]))

                yield episode_data


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
