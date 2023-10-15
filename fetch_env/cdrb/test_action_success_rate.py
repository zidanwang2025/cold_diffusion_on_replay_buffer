import json
import os
import pathlib
import pickle
import sys
import gymnasium as gym
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import h5py
from tqdm import tqdm
import wandb


from diffusion_models import GaussianDiffusion
from experiment_config import Args
from temporal_unet import TemporalUnet
from utils1.helpers import apply_conditioning
from utils1.d4rl_utils import sequence_dataset

OBS_MIN = None
OBS_MAX = None

def normalize(state=None, data_path=None):
    if "ogdata" in Args.nickname:
        return state
    global OBS_MIN, OBS_MAX
    if OBS_MAX is None:
        with h5py.File(data_path, 'r') as f:
            OBS_MIN = f['obs'].attrs['min']
            OBS_MAX = f['obs'].attrs['max']

    if state is not None:
        state_len = state.shape[-1]
        state = (state - OBS_MIN[:state_len]) / (OBS_MAX[:state_len] - OBS_MIN[:state_len])
        state = state * 2 - 1

    return state

def denormalize(state=None, data_path=None):
    if "ogdata" in Args.nickname:
        return state
    global OBS_MIN, OBS_MAX
    if OBS_MAX is None:
        with h5py.File(data_path, 'r') as f:
            OBS_MIN = f['obs'].attrs['min']
            OBS_MAX = f['obs'].attrs['max']
           
    if state is not None:
        state_len = state.shape[-1]
        state = (state + 1) / 2
        state = state * (OBS_MAX[:state_len] - OBS_MIN[:state_len]) + OBS_MIN[:state_len]

    return state

def update_args(**kwargs):
    Args._update(kwargs)
    Args.control = (Args.control == 'True')
    Args.test_validation = (Args.test_validation == 'True')
    Args.save_traj = int(Args.save_traj)
    try:
        Args.cond_start = int(Args.cond_start)
        Args.cond_end = int(Args.cond_end)
    except ValueError:
        pass


def is_success(cur_state, desired_goal):
    '''check if the current state is close enough to the desired goal'''
    if np.linalg.norm(cur_state[3:6] - desired_goal[:3]) < 0.05:
        return True
    else:
        return False

def setup():
    '''initialize env and set up diffusion model, renderer, param_string, episode_data'''

    model = TemporalUnet(
            horizon=Args.short_horizon,
            transition_dim=Args.observation_dim + Args.action_dim + Args.repeat_len if Args.include_goal_in_state else Args.observation_dim + Args.action_dim,
            cond_dim=Args.observation_dim + Args.repeat_len if Args.include_goal_in_state else Args.observation_dim,
            dim_mults=Args.dim_mults,
        )
    model.to(Args.device)


    diffusion = GaussianDiffusion(
        model,
        horizon=Args.short_horizon,
        observation_dim=Args.observation_dim,
        action_dim=Args.action_dim,
        n_timesteps=Args.n_diffusion_steps,
        loss_type=Args.loss_type,
        clip_denoised=Args.clip_denoised,
        ## loss weighting
        action_weight=Args.action_weight,
        loss_weights=Args.loss_weights,
        loss_discount=Args.loss_discount,
        trim_buffer_mode=Args.trim_buffer_mode,
        data_path=Args.data_path,
    )
    diffusion.to(Args.device)


    param_string = "join_" + str(Args.join_action_state) \
        + "_trim_" + Args.trim_buffer_mode \
        + "_noise_" + str(Args.forward_sample_noise) \
        + "_weight_" + str(Args.action_weight) \
        + "_horizon_" + str(Args.horizon) \
        + "_epoch_" + str(int(Args.n_train_steps/Args.n_steps_per_epoch)) \
        + "_obsdim_" + str(Args.observation_dim) \
        + "_actdim_" + str(Args.action_dim) \
        + "_dataset_" + str(Args.dataset)
    if Args.nickname is not None:
        param_string += "_nickname_" + Args.nickname

    if Args.trim_buffer_mode == "kmeans":
        param_string += "_k_" + str(Args.k_cluster)
    elif Args.trim_buffer_mode == "euclidean":
        param_string += "_d_" + str(Args.d_distance)

    file_path = Args.snapshot_root + "/" + param_string + ".pt"

    diffusion.load_state_dict(torch.load(file_path, map_location=torch.device(Args.device))['model'])
    diffusion.to(Args.device)

    # get min and max of observation space
    normalize(data_path=Args.data_path)

    data_path = Args.data_path
    if Args.test_validation:
        '''use none normalization validation set for testing'''
        data_path = data_path.replace("normalized", "no-images")

    itr = sequence_dataset(data_path=data_path, observation_dim=Args.observation_dim, action_dim=Args.action_dim, validation=Args.test_validation, mode=Args.validation_mode)
    episode_data = []
    for data in itr:
        episode_data.append(data)

    return diffusion, episode_data


def trim_traj(episode_data, expert_trajectory=None):
    if expert_trajectory is None:
        random_index = np.random.randint(0, len(episode_data))
        expert_trajectory = episode_data[random_index]

    expert_trajectory = expert_trajectory['observations'][-Args.horizon:]

    return expert_trajectory

def generate_recon(diffusion,
                     Args,
                     cold_diffusion=False,   
                     x_start=None,           
                     cond={}                
                     ):
    
    if x_start is not None:

        x_start = torch.from_numpy(x_start).to(Args.device)

        x_start_repeat = x_start.repeat(Args.n_diffusion_steps, 1, 1)
        noise = torch.randn_like(x_start_repeat).to(Args.device)

        t = torch.range(0, Args.n_diffusion_steps - 1, 1, dtype=torch.int64).to(Args.device)
        
        x_noisy = diffusion.forward_sample(x_start_repeat, t, noise)
        x_noisy = apply_conditioning(x_noisy, cond, Args.action_dim)
        x_noisy = x_noisy.to(dtype=torch.float)

        x_recon = x_noisy[-1].repeat(1, 1, 1)
        for i in reversed(range(1, Args.n_diffusion_steps)):
            timesteps = torch.full((1,), i, dtype=torch.long)
            if cold_diffusion:
                x_recon = diffusion.backward_sample_cold(x_recon, cond, timesteps)
            else:
                x_recon = diffusion.backward_sample(x_recon, cond, timesteps)

            x_recon = apply_conditioning(x_recon, cond, Args.action_dim)
                

    else:
        shape = (1, Args.horizon, Args.observation_dim+Args.action_dim)
        x_recon = diffusion.backward_sample_loop(shape, cond, verbose=False, cold=cold_diffusion)
        x_recon = x_recon[-1]

    return x_recon

def sample_states(generated_states, num_states):
    length = len(generated_states)
    interval = (length - 1) / (num_states - 1)
    rounded_interval = int(round(interval))
    indices = np.arange(0, length, rounded_interval, dtype=int)
    generated_states = generated_states[indices]
    return generated_states

def visualize(diffusion, episode_data):
    generated_to_save = []

    if Args.test_validation:
        validation_data = episode_data
        env = gym.make(Args.dataset)
        env.set_fixed_block_pos(np.array([0, 0]))
        obs, info = env.reset()
        constants = obs['achieved_goal'][:2]

    s_list = [1]
    for s in s_list:
        success = 0
        env = gym.make(Args.dataset)
        for each_run in range(Args.n_test):
            if Args.render_mode == 'expert':
                x_start = trim_traj(episode_data)
            elif Args.render_mode == 'random_rb':
                x_start = None
            
            obs, info = env.reset()
            cond = {}
            if Args.render_mode == 'expert':
                pass
            else:
                if Args.test_validation:
                    val_index = min(each_run, len(validation_data)-1)
                    data = validation_data[val_index]
                    block_pos = data['achieved_goals'][0, :2]
                    goal = data['desired_goals'][0]
                    env.set_fixed_goal(goal)
                    if "Reach" not in Args.dataset:
                        env.set_fixed_block_pos(block_pos-constants)
                    obs, info = env.reset()
                if "Fetch" in Args.dataset and "Reach" not in Args.dataset:
                    cond['dataset'] = Args.dataset
                    cond['cond_start'] = Args.cond_start
                    cond['cond_end'] = Args.cond_end
                    cond['goal_index'] = Args.horizon - 1
                    for cond_index in range(1, Args.cond_count_front):
                        cond[cond_index] = torch.FloatTensor(normalize(obs['observation'][:Args.observation_dim]))
                    start = normalize(obs['observation'][:Args.observation_dim])
                    goal = np.concatenate((obs['desired_goal'], obs['desired_goal']))
                    goal = normalize(goal)
                        
                    cond[0] = torch.FloatTensor(start)
                    cond[Args.horizon-1] = torch.FloatTensor(goal)
                    cond = {key: val.to(Args.device) if isinstance(key, int) else val for key, val in cond.items()}

                elif "Reach" in Args.dataset:
                    cond['dataset'] = Args.dataset
                    cond['cond_start'] = Args.cond_start
                    cond['cond_end'] = Args.cond_end
                    cond['goal_index'] = Args.horizon - 1
                    for cond_index in range(1, Args.cond_count_front):
                        cond[cond_index] = torch.FloatTensor(normalize(obs['observation'][:Args.observation_dim]))
                    start = normalize(obs['observation'][:Args.observation_dim])
                    goal = normalize(obs['desired_goal'])

                    cond[0] = torch.FloatTensor(start)
                    cond[Args.horizon-1] = torch.FloatTensor(goal)
                    cond = {key: val.to(Args.device) if isinstance(key, int) else val for key, val in cond.items()}

            x_recon = generate_recon(diffusion,
                                Args,
                                cold_diffusion=False,
                                x_start=x_start,
                                cond=cond
            )

            generated_actions = x_recon[0, :, :Args.action_dim]

            action_based_traj = [start[:3]]
            for action in generated_actions:
                action = action.cpu().numpy()    
                obs_dict, _, _, _, info = env.step(action)
                action_based_traj.append(obs_dict["observation"][:3])
                if info["is_success"]:
                    success += 1
                    break

        print("Success rate:", success/Args.n_test*100)
 
        wandb.log({'generated action success rate (%)': success/Args.n_test*100,
                    'validation mode': Args.validation_mode})

def main():
    update_args()
    diffusion, episode_data = setup()
    wandb.login()

    wandb.init(
        # Set the project where this run will be logged
        project=Args.dataset,
        name=Args.model_type + str(Args.observation_dim),
        config=vars(Args),
    )
    visualize(diffusion, episode_data)
    wandb.finish()

if __name__ == "__main__":
    main()
