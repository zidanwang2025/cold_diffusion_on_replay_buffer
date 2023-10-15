from collections import namedtuple
import numpy as np
import torch

from utils1.helpers import get_preprocess_fn
from utils1.d4rl_utils import load_environment, sequence_dataset
from utils1.data_normalization_utils import DatasetNormalizer
from utils1.buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')


# class Batch:
#     def __init__(self,trajectories=None,conditions=None):
#         self.trajectories=trajectories
#         self.conditions=conditions
#
# class ValueBatch:
#     def __init__(self,trajectories=None,conditions=None,values=None):
#         self.trajectories=trajectories
#         self.conditions=conditions
#         self.values=values

# batch = batch._replace(trajectories=new_traj.to(Args.device))

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='maze2d-large-v1', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=1000, termination_penalty=0, use_padding=True, validation=False):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn, validation=validation)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        # print("initial fields")
        # print(fields)

        # if Args.include_goal_in_state:
        #     new_obs = np.zeros(
        #         (fields.observations.shape[0], fields.observations.shape[1], fields.observations.shape[2] + 2))
        #     # trajectory length
        #     # traj_len = 384
        #     traj_len = Args.horizon
        #     # len for pos is 2, if you want to repeat pos and vol, change this to 4
        #     # repeat_len = 2
        #     repeat_len = Args.repeat_len
        #     for index in range(fields.observations.shape[0]):
        #         # original_observations contains 4w step/trajectory, so you need to cut to get the reasonable trajectory
        #         # and fill others with zero
        #         goal_pos = np.array([fields.observations[index, traj_len - 1, :repeat_len] for _ in range(traj_len)])
        #         # find lines of zeros we need
        #         zeros = np.zeros((fields.observations.shape[1] - traj_len, 2))
        #         goal_pos = np.vstack((goal_pos, zeros))
        #         new_obs[index] = np.hstack((fields.observations[index], goal_pos))
        #     fields.observations=new_obs
        #     print("changed shape")
        #     print(new_obs.shape)
        #     print("copied shape")
        #     print(fields.observations.shape)
        #     Args.observation_dim=fields.observations.shape[-1]
        #     print("What's in fields")
        #     print(fields)
        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)


        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)

            # normed = self.normalizer(array, key)

            normed = array

            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)

        return batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
