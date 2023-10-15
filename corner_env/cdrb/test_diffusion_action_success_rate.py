import sys
import gymnasium as gym
import numpy as np
import torch

from diffusion_models import GaussianDiffusion
from experiment_config import Args
from temporal_unet import TemporalUnet
from temporal_transformer import TemporalTransformer
from utils1.helpers import apply_conditioning

sys.path.append("..")
from gymnasium_corner_env_renderer import CornerEnvRenderer
from replay_buffer import ReplayBuffer
from gymnasium_registration import initialize_env

def update_args(**kwargs):
    Args._update(kwargs)
    Args.join_action_state = (Args.join_action_state == "True")

def setup():
    '''initialize env and set up diffusion model, renderer, param_string, episode_data'''

    initialize_env()
    env = gym.make("gymnasium-corner-env-archive")
    env.reset()

    renderer = CornerEnvRenderer(env, multi=True)

    expert_trajectories = env.get_dataset(normalize=True)
    replay_buffer = ReplayBuffer(state_dim=Args.observation_dim, action_dim=Args.action_dim)
    replay_buffer.convert_dict(expert_trajectories, k=Args.k_cluster, joint_as=Args.join_action_state)

    if Args.model == "models.TemporalUnet":
        model = TemporalUnet(
            horizon=Args.short_horizon,
            transition_dim=Args.observation_dim + Args.action_dim + Args.repeat_len if Args.include_goal_in_state else Args.observation_dim + Args.action_dim,
            cond_dim=Args.observation_dim + Args.repeat_len if Args.include_goal_in_state else Args.observation_dim,
            dim_mults=Args.dim_mults,
        )
    elif Args.model == "models.TemporalTransformer":
        model = TemporalTransformer(Args.observation_dim + Args.action_dim + Args.repeat_len if Args.include_goal_in_state else Args.observation_dim + Args.action_dim)
    else:
        raise NotImplementedError("Please choose TemporalUnet or TemporalTransformer.")
    model.to(Args.device)

    diffusion = GaussianDiffusion(
        model,
        horizon=Args.short_horizon,
        observation_dim=Args.observation_dim,
        action_dim=Args.action_dim,
        n_timesteps=Args.n_diffusion_steps,
        loss_type=Args.loss_type,
        clip_denoised=Args.clip_denoised,
        action_weight=Args.action_weight,
        loss_weights=Args.loss_weights,
        loss_discount=Args.loss_discount,
    )


    param_string = "join_" + str(Args.join_action_state) \
        + "_trim_" + Args.trim_buffer_mode \
        + "_noise_" + str(Args.forward_sample_noise) \
        + "_weight_" + str(Args.action_weight) \
        + "_horizon_" + str(Args.horizon) \
        + "_epoch_" + str(int(Args.n_train_steps/Args.n_steps_per_epoch)) \
        + "_obsdim_" + str(Args.observation_dim) \
        + "_actdim_" + str(Args.action_dim)

    if Args.trim_buffer_mode == "kmeans":
        param_string += "_k_" + str(Args.k_cluster)
    elif Args.trim_buffer_mode == "euclidean":
        param_string += "_d_" + str(Args.d_distance)

    file_path = Args.snapshot_root + "/" + param_string + ".pt"

    diffusion.load_state_dict(torch.load(file_path)['model'])

    dataset = env.get_dataset(normalize=True, single_goal=False, fixed_start=False)
    N = dataset['rewards'].shape[0]
    episode_data = {}
    current_episode = []
    episode_step = 0
    state_counter = 0
    for i in range(N):
        state_counter += 1
        done_bool = bool(dataset['terminals'][i])
        current_episode.append(np.hstack((dataset['actions'][i], dataset['observations'][i])))
        if done_bool:
            while state_counter % 4 != 0:
                current_episode.append(np.hstack((dataset['actions'][i], dataset['observations'][i])))
                state_counter += 1
            if len(current_episode) >= Args.horizon:
                episode_data[episode_step] = np.array(current_episode)
                episode_step += 1
            current_episode = []
            state_counter = 0

    return diffusion, renderer, param_string, episode_data


def trim_traj(episode_data, expert_trajectory=None):
    if expert_trajectory is None:
        random_index = np.random.randint(0, len(episode_data))
        expert_trajectory = episode_data[random_index]

    expert_trajectory = expert_trajectory[-Args.horizon:]

    return expert_trajectory

def generate_recon(diffusion,
                     Args,
                     cold_diffusion=False,    # True if use cold diffusion backward process
                     x_start=None,            # an expert trajectory of states and actions
                     cond={}                  # conditioning applied to x
                     ):
    
    if x_start is not None:

        x_start = torch.from_numpy(x_start)

        x_start_repeat = x_start.repeat(Args.n_diffusion_steps, 1, 1)
        noise = torch.randn_like(x_start_repeat)

        t = torch.range(0, Args.n_diffusion_steps - 1, 1, dtype=torch.int64)
        
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

def test_action(diffusion, renderer, param_string, episode_data):
    success = 0
    for each_test in range(Args.num_tests):
        if Args.render_mode == 'expert':
            x_start = trim_traj(episode_data)
        elif Args.render_mode == 'random_rb':
            x_start = None

        env = gym.make("gymnasium-corner-env-archive")
        env.reset()
        start = env.reset_pos
        goal = env.goal
        old_min = -6.5
        old_max = 5.5
        new_min = -1
        new_max = 1
        start = (((start - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
        goal = (((goal - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

        start = np.append(start, np.array([0., 0.])).astype(np.float32)
        goal = np.append(goal, np.array([0., 0.])).astype(np.float32)
        # range (-1, 1)
        cond = {0: torch.FloatTensor(start), Args.horizon-1: torch.FloatTensor(goal)}

        x_recon = generate_recon(diffusion,
                    Args,
                    cold_diffusion=False,
                    x_start=x_start,
                    cond=cond
                    )
        generated_actions = x_recon[0, :, :2]

        old_min = -1
        old_max = 1
        new_min = -6.5
        new_max = 5.5

        start = cond[0][:2]
        goal = cond[Args.horizon-1][:2]

        start = (((start - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
        start = start.detach().numpy()
        goal = (((goal - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
        goal = goal.detach().numpy()

        # env.reset()
        env.set_start_goal(start, goal)

        action_based_traj = [start]
        for action in generated_actions:
            action = action.detach().numpy()
            obs_dict, reward, terminated, truncated, info = env.step(action)
            action_based_traj.append(obs_dict["observation"][:2])
            if info["success"]:
                break
        success += info["success"]

    print(f"Out of {Args.num_tests} tests, action reached goal for {success} times.")
    print(f"Success rate: {float(success/Args.num_tests*100)}%.")

def main():
    update_args()
    diffusion, renderer, param_string, episode_data = setup()
    test_action(diffusion, renderer, param_string, episode_data)

if __name__ == "__main__":
    main()