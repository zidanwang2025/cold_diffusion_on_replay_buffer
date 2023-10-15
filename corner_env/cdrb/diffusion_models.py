import einops
import copy

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from experiment_config import Args
import utils1 as utils
from utils1.helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)
from utils1.progress import Silent
import pickle
import sys
import random
import time
import copy

sys.path.append("..")
from replay_buffer import ReplayBuffer
from gymnasium_registration import initialize_env

# Shortcuts
from torch import nonzero as torch_nonzero
from random import randint as random_randint

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, action_weight=1.0, loss_discount=1.0, loss_weights=None,
        trim_buffer_mode="kmeans", data_path=None, max_dist=None, min_dist = 0.,
        dist_scheduler='linear', replay_dataset="gymnasium-corner-env-standard"
    ):
        super().__init__()
        self.action_weight = None
        initialize_env()
        if Args.include_goal_in_state:
            observation_dim=observation_dim+Args.repeat_len
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        self.n_timesteps = int(n_timesteps)

        # NOTE: We shouldn't have `Args` in the default argument list, since it may be evaluated before Args._update
        if max_dist is None:
            max_dist = 2*np.sqrt(Args.observation_dim+Args.action_dim)

        if dist_scheduler == 'linear':
            line_distance_schedule = torch.linspace(0, max_dist - min_dist, n_timesteps)
            line_distance_schedule[1:] += min_dist
            self.distance_schedule = line_distance_schedule
        elif dist_scheduler == 'sigmoid':
            start, end = min_dist, max_dist
            xs = torch.linspace(-6, 6, n_timesteps)
            self.distance_schedule = torch.sigmoid(xs) * (end - start) + start
        elif dist_scheduler == "sqrt":
            line_distance_schedule = torch.linspace(0, max_dist, n_timesteps)
            self.distance_schedule = torch.sqrt(line_distance_schedule / max_dist) * (max_dist - min_dist) + min_dist
        elif dist_scheduler == "log":
            self.distance_schedule = torch.log(torch.linspace(1, n_timesteps+1, n_timesteps)) * (max_dist - min_dist) / np.log(n_timesteps+1) + min_dist
        else:
            raise ValueError(f'Unknown schedule: {dist_scheduler}')

        self.noise_schedule = torch.linspace(0, Args.forward_sample_noise, n_timesteps)

        self.clip_denoised = clip_denoised

        # get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        env = gym.make(replay_dataset)
        expert_trajectories = env.get_dataset(normalize=True, dataset_size=Args.dataset_size)
        # I should just save the trimmed replay buffer and save it in the env, call it here
        self.replay_buffer = ReplayBuffer(state_dim=self.observation_dim, action_dim=self.action_dim,
                                          dataset_name=replay_dataset, trim_buffer_mode=trim_buffer_mode)

        self.replay_buffer.convert_dict(expert_trajectories, file_path=data_path, k=Args.k_cluster, joint_as=Args.join_action_state)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        # manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights
    #------------------------------------------ sampling ------------------------------------------#


    @torch.no_grad()
    def backward_sample(self, x, cond, t):
        x_recon = self.model(x, cond, t)

        # How do we clip without hardcoding?
        # if self.clip_denoised:
        #     x_recon.clamp_(-1., 1.)
        # else:

        # not sure if we can get rid of the sigma term, but what would it even mean in our scenario
        return self.forward_sample(x_recon, t-1, noise=None)


    @torch.no_grad()
    def backward_sample_cold(self, x, cond, t):
        x_recon = self.model(x, cond, t)
        x_cold = x - self.forward_sample(x_recon, t, noise=None) + self.forward_sample(x_recon, t-1, noise=None)
        return x_cold


    @torch.no_grad()
    def backward_sample_loop(self, shape, cond, verbose=False, cold=False):

        batch_size = shape[0]
        if Args.join_action_state:
            x = self.replay_buffer.sample_action_state_joint(batch_size, shape[1])
        else:
            action, state = self.replay_buffer.sample_action_state_separate(batch_size, shape[1])
            x = torch.cat((action, state), dim=2)
        x = apply_conditioning(x, cond, self.action_dim, cond_vels=Args.cond_vels)

        recon_list = []
        progress = utils.Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(1, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, dtype=torch.long, device=Args.device)
            if cold:
                x = self.backward_sample_cold(x, cond, timesteps)
            else:
                x = self.backward_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim, cond_vels=Args.cond_vels)
            recon_list.append(x)

            progress.update({'t': i})


        progress.close()

        return recon_list

    # @torch.no_grad()
    # def conditional_sample(self, cond, *args, horizon=None, **kwargs):
    #     '''
    #         conditions : [ (time, state), ... ]
    #     '''
    #     device = self.betas.device
    #     batch_size = len(cond[0])
    #     horizon = horizon or self.horizon
    #     shape = (batch_size, horizon, self.transition_dim)
    #
    #     return self.p_sample_loop(shape, cond, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def forward_sample(self, x_start, t, noise=None):
        if Args.trim_buffer_mode == 'kmeans':
            if Args.join_action_state:
                sampled_values = self.replay_buffer.action_state_kmeans
            else:
                sampled_actions = self.replay_buffer.action_kmeans
                sampled_states = self.replay_buffer.state_kmeans
        elif Args.trim_buffer_mode == 'euclidean':
            if Args.join_action_state:
                sampled_values = self.replay_buffer.action_state_trimmed
            else:
                sampled_actions = self.replay_buffer.action_trimmed
                sampled_states = self.replay_buffer.state_trimmed
        elif Args.trim_buffer_mode == 'nsample':
            if Args.join_action_state:
                sampled_values = self.replay_buffer.sample_state_action_joint(1, 5000)[0]
            else:
                sampled_actions, sampled_states = self.replay_buffer.sample_state_action_separate(1, 5000)
                sampled_actions = sampled_actions[0]
                sampled_states = sampled_states[0]
        else:
            print("Trim buffer mode incorrect. Your input: ", Args.trim_buffer_mode)
            exit()

        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)

        noise_schedule = self.noise_schedule.to(noise.device)
        noise_t = extract(noise_schedule, t, x_start.shape) * noise

        x_hat_dup = copy.deepcopy(x_start)
        
        # started = time.perf_counter()
        self.distance_schedule = self.distance_schedule.to(x_hat_dup.device)
        if Args.join_action_state:
            sampled_values = sampled_values.to(x_hat_dup.device)
            batch_size, traj_len, dim = x_hat_dup.shape

            # b: batch_size, s: traj_len, d: dim
            x_hat_flat = einops.rearrange(x_hat_dup, 'b s d -> (b s) d')

            # sampled_values:   (replay_buffer_size, dim)
            # x_hat_flat:       (batch * traj_len, dim)
            # => flat_distances:     (batch * traj_len, replay_buffer_size)
            flat_distances = torch.linalg.norm(sampled_values[None, :, :] - x_hat_flat[:, None, :], axis=2)

            distances = einops.rearrange(flat_distances, '(b s) r -> b s r', b=batch_size, s=traj_len)

            # t: (batch, )
            # distance_schedule[t]: (batch, )
            # distances: (batch, traj_len, buffer_size)
            # mask: (batch, traj_len, buffer_size)
            distance_threshold = self.distance_schedule[t][:, None, None]
            mask = distances < distance_threshold

            # mask: (batch, traj_len, buffer_size)
            flat_mask = einops.rearrange(mask, 'b s r -> (b s) r')
            # mask = einops.rearrange(flat_mask, '(b s) r -> b s r', b=x_hat.shape[0], s=x_hat.shape[1])
            
            # 1. Augment the mask with an additional entry that shows whether there is any point within the ball
            # additional_mask: (batch, traj_len) => (batch * traj_len, 1)
            additional_mask = torch.logical_not(mask.sum(-1).type(torch.bool))
            additional_mask = einops.rearrange(additional_mask, 'b s -> (b s) 1')
            # mask = torch.cat((mask, additional_mask[None, None, :].repeat(batch_size, traj_len, 1)), dim=1)

            # additional_mask: (batch * traj_len, 1)
            # flat_mask: (batch * traj_len, buffer_size) => (batch * traj_len, buffer_size + 1)
            flat_mask = torch.cat((flat_mask, additional_mask), dim=1)

            # 2. See the mask as probability distribution (per row) and sample from it
            # flat_mask: (batch * traj_len, buffer_size + 1)
            # sample_inds: (batch * traj_len)
            sample_inds = torch.multinomial(flat_mask.type(torch.float), 1)
            sample_inds = sample_inds[:, 0]
            
            # 3. Prepare an augmented tensor that stores the points in the buffer and current x_hat
            # Concat the buffer and x_hat
            # sampled_values: (buffer_size, dim)
            # x_hat_flat: (batch * traj_len, dim)
            # augmented: (batch * traj_len, buffer_size + 1, dim)

            # sampled_values: (buffer_size, dim) => (batch * traj_len, buffer_size, dim)
            _sampled_values = einops.repeat(sampled_values, 'r d -> (b s) r d', b=batch_size, s=traj_len)
            augmented = torch.cat((_sampled_values, x_hat_flat[:, None, :]), dim=1)
            
            # 4. Sample from the augmented tensor using sample_inds
            # sample_inds: (batch * traj_len, 1)
            # augmented: (batch * traj_len, buffer_size + 1, dim)
            x_hat_dup = augmented[torch.arange(batch_size * traj_len), sample_inds, :]

            # reshape
            x_hat_dup = einops.rearrange(x_hat_dup, '(b s) d -> b s d', b=batch_size, s=traj_len)

            # torch.cuda.synchronize(device=x_hat_dup.device)
            # elapsed = time.perf_counter() - started
            # print(f'elapsed (tensor ops): {elapsed * 1000:.1f} ms')

            x_hat = x_hat_dup

        else:
            raise NotImplementedError()

        if Args.cdrb_add_noise:
            x_hat += noise_t

        return x_hat

    def backward_losses(self, x_start, cond, t):
        # noise = torch.randn_like(x_start)/10
        # x_noisy = self.forward_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = self.forward_sample(x_start, t)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim, cond_vels=Args.cond_vels)
        x_recon = self.model(x_noisy, cond, t)
        if Args.apply_condition_for_loss:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim, cond_vels=Args.cond_vels)

        # assert noise.shape == x_recon.shape
        if Args.weight_loss_by_current_radius != 0:
            online_weight = torch.reciprocal(self.distance_schedule[t])
            if Args.weight_loss_by_current_radius > 1:
                online_weight = torch.pow(online_weight, Args.weight_loss_by_current_radius)
            online_weight = einops.repeat(online_weight, "B -> B T D", T=x_noisy.shape[1], D=x_noisy.shape[2])
        else:
            online_weight = 1
        loss, info = self.loss_fn(x_recon, x_start, online_weight)

        return loss, info

    def loss(self, x, cond):
        batch_size = len(x)

        t = torch.randint(1, self.n_timesteps, (batch_size,), device=x.device).long()
        # t = torch.randn(batch_size)
        # t = torch.full((32,), 127, dtype=torch.int64)
        # t = torch.range(1, 128, 4, dtype=torch.int64)

        return self.backward_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
