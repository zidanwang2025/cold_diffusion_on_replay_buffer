import sys
import torch
import wandb
import cv2

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from pathlib import Path

import datasets
import main_trainer
from diffusion_models import GaussianDiffusion
from experiment_config import Args
from temporal_unet import TemporalUnet
from utils1.helpers import apply_conditioning

sys.path.append("..")
from gymnasium_corner_env_renderer import CornerEnvRenderer
from gymnasium_registration import initialize_env

def setup_env():
    '''initialize env and set up diffusion model, renderer, param_string, episode_data'''

    ### set environment
    initialize_env()
    env = gym.make("gymnasium-corner-env")
    env.reset()

    renderer = CornerEnvRenderer(env, multi=True)
    return env, renderer

def update_args(**kwargs):
    Args._update(kwargs)

    Args.short_horizon = Args.horizon * Args.shorten_ratio

    Args.horizon = int(Args.horizon)
    Args.short_horizon = int(Args.short_horizon)

    print("include goal:", Args.include_goal_in_state)
    print("pin goal:", Args.pin_goal)
    print("train horizon:", Args.short_horizon)

def prepare_cond_for_random_rb(number_of_data):
    env = gym.make("gymnasium-corner-env")

    def normalize(start, goal):
        old_min = -6.5 # -6.5 Normalized so that goal coordinates (-5.5,-5.5), (5.5, 5.5) become (-1.,-1.), (1., 1.)
        old_max = 5.5
        new_min = -1
        new_max = 1
        start = (((start - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
        goal = (((goal - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

        start = np.append(start, np.array([0., 0.])).astype(np.float32)
        goal = np.append(goal, np.array([0., 0.])).astype(np.float32)
        return start, goal

    start_list, goal_list = [], []
    for i in range(number_of_data):
        env.reset()
        start = env.reset_pos
        goal = env.goal
        start, goal = normalize(start, goal)
        start_list.append(start)
        goal_list.append(goal)
    starts = np.stack(start_list, axis=0)
    goals = np.stack(goal_list, axis=0)

    # range (-1, 1)
    cond = {0: torch.FloatTensor(starts), Args.horizon-1: torch.FloatTensor(goals)}
    cond = {key: val.to(Args.device) for key, val in cond.items()}

    return cond

def prepare_val_traj(env, Args, trim="tail"):
    """
    trim: choise "head", "tail"
    """
    dataset = env.get_dataset(normalize=True, single_goal=False, fixed_start=False, validation=True)
    N = dataset['rewards'].shape[0]
    episode_data = []
    current_episode = []
    state_counter = 0
    for i in range(N):
        state_counter += 1
        done_bool = bool(dataset['terminals'][i])
        current_episode.append(np.hstack((dataset['actions'][i], dataset['observations'][i])))
        if done_bool:
            while state_counter % 4 != 0:
                current_episode.append(np.hstack((dataset['actions'][i], dataset['observations'][i])))
                state_counter += 1
            if len(current_episode) >= Args.horizon: # If the length of episode is shorter than Args.horizon, skip this episode
                if trim == "head":
                    episode_data.append(current_episode[:Args.horizon])
                elif trim == "tail":
                    episode_data.append(current_episode[-Args.horizon:])
                else:
                    NotImplementedError("trim is head or tail.")
            current_episode = []
            state_counter = 0
    
    if len(episode_data) == 0:
        raise ValueError("Please increase Args.horizon. There is no validation data the length of which is shorter than Args.horizon.")
    return np.array(episode_data)

def get_condition(traj, Args):
    return {0: traj[:,0,Args.action_dim:], Args.horizon-1: traj[:,-1,Args.action_dim:]}

def pred_traj_from_expert(val_traj, cond, diffusion, cold=False):
    batch_size = val_traj.shape[0]

    noise = torch.randn_like(val_traj).to(Args.device)
    timesteps = torch.full((batch_size,), (Args.n_diffusion_steps - 1), dtype=torch.long).to(Args.device)

    x_noisy = diffusion.forward_sample(val_traj, timesteps, noise)
    x_noisy = apply_conditioning(x_noisy, cond, Args.action_dim)
    x_noisy = x_noisy.to(dtype=torch.float)

    pred_traj = [x_noisy]
    for i in reversed(range(1, Args.n_diffusion_steps)):
        print('backward step:', i)
        timesteps = torch.full((batch_size,), i, dtype=torch.long).to(Args.device)
        if cold:
            x_noisy = diffusion.backward_sample_cold(x_noisy, cond, timesteps)
        else:
            x_noisy = diffusion.backward_sample(x_noisy, cond, timesteps)
        x_noisy = apply_conditioning(x_noisy, cond, Args.action_dim)
        pred_traj.append(x_noisy)

    return pred_traj

class WaypointController(object):
    def __init__(self, p_gain=1.0, d_gain=-0.1, solve_thresh=0.01):
        # Initialize controller with Proportional and Derivative gains, and solution threshold
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.solve_thresh = solve_thresh
        self.vel_thresh = 0.1
        self.prev_prop = None

    def get_action(self, location, velocity, target):
        # Compute the action using PD control and clip to (-1, 1)
        # Compute control
        prop = target - location
        action = self.p_gain * prop + self.d_gain * velocity

        # Clip the action within the range of -1.0 to 1.0
        action = np.clip(action, -1.0, 1.0)
        return action

    def is_path_solved(self, location, target):
        dist = np.linalg.norm(location - target)
        return dist < self.solve_thresh

def evaluate(renderer, episode_data, predicted_data, save_dir, num_vis=10, save_wandb=True, run_controller=True):

    def shift_range(state, old_min, old_max, new_min, new_max):
        old_range = old_max - old_min
        new_range = new_max - new_min
        new_state = (((state - old_min) * new_range) / old_range) + new_min
        return new_state

    def render_and_tonumpy(trajectory):
        fig, ax = plt.subplots(figsize=(8, 8))
        renderer.render_trajectory(trajectory, ax, reset=True, pin=True, gif=False)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        numpy_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
        plt.close()
        return numpy_array

    count = 0
    frame_list = []
    controller_step_list = []
    for repeat in range(num_vis):
        print(f"index: {repeat+1}/{num_vis}")

        controller = WaypointController()

        gt_trajectory = episode_data[repeat,:, Args.action_dim:].cpu()
        pred_trajectory = predicted_data[repeat,:, Args.action_dim:].cpu()
        
        # get goal for visualize
        start = pred_trajectory[0][:2].numpy()
        start = shift_range(start, -1.0, 1.0, -6.5, 5.5) #-6.5 and 5.5 might need to be changed
        val_goal = pred_trajectory[-1][:2].numpy()
        val_goal = shift_range(val_goal, -1.0, 1.0, -6.5, 5.5)

        if len(renderer.square_arr) != len(renderer.square_arr[0]):
            raise NotImplementedError("TODO")
        
        wall_length = 1 # perimeter wall
        length_from_wall_to_wall = len(renderer.square_arr) - (2 * wall_length)
        possible_goals = renderer.possible_goals - 0.5 - (1 - wall_length)
        possible_goals = [shift_range(goal, 0, length_from_wall_to_wall, -6.5, 5.5) for goal in possible_goals]

        errors = np.sum((np.array(possible_goals) - val_goal) ** 2, axis=1)
        min_error_index = np.argmin(errors)
        goal = possible_goals[min_error_index]
        
        denorm_goal = torch.tensor([shift_range(goal, -6.5, 5.5, -1.0, 1.0)])
        gt_trajectory = torch.cat([gt_trajectory[:,:2], denorm_goal], 0)
        gt_image_array = render_and_tonumpy(gt_trajectory)
        pred_trajectory = torch.cat([pred_trajectory[:,:2], denorm_goal], 0)
        pred_image_array = render_and_tonumpy(pred_trajectory[:, :2])
        pred_trajectory = pred_trajectory.numpy()

        if run_controller:
            initialize_env()
            env = gym.make("gymnasium-corner-env")
            env.reset()
            env.set_start_goal(start, goal)
            
            controller_generated_traj = []
            current_location = start
            velocity = np.zeros(2)
            stop = False
            for i, state in enumerate(pred_trajectory[:-1]):
                target = np.array(pred_trajectory[i+1, :2])
                target = shift_range(target, -1.0, 1.0, -6.5, 5.5)
                num_steps = 0
                    
                while not controller.is_path_solved(current_location, target) and num_steps < Args.num_action_per_step: #not sure how many actions we should be allowing the agent to take between each pair of observations, maybe just 1?
                    action = controller.get_action(current_location, velocity, target)
                    
                    obs, reward, terminated, truncated, info = env.step(action)

                    velocity = obs['observation'][2:]
                    current_location = obs['observation'][:2]
                    controller_generated_traj.append(current_location)

                    num_steps += 1
                    if info['success']:
                        stop = True
                        break
                if stop:
                    print("Success!")
                    count += 1
                    break

            for i, state in enumerate(controller_generated_traj):
                controller_generated_traj[i] = shift_range(state, -6.5, 5.5, -1.0, 1.0)
            
            controller_step_list.append(len(controller_generated_traj))
            controller_image_array = render_and_tonumpy(torch.FloatTensor(np.array(controller_generated_traj)))

        if run_controller:
            frame = np.concatenate((gt_image_array, pred_image_array, controller_image_array), axis=1)
            frame = cv2.putText(frame, f'val index: {repeat}, success: {info["success"]}, controller_len: {len(controller_generated_traj)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            frame = np.concatenate((gt_image_array, pred_image_array), axis=1)
            frame = cv2.putText(frame, f'val index: {repeat}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        frame_list.append(frame)

    video = np.asarray(frame_list).transpose(0,3,1,2)

    if save_wandb:
        wandb.log({"video":wandb.Video(video, fps=5, format="mp4")})
        if run_controller:
            wandb.log({"success rate": count/num_vis,
                       "controller_steps": sum(controller_step_list) / len(controller_step_list), 
                       "Num episodes": num_vis})
    else:
        output_path = f"{save_dir}/video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 5
        width = video.shape[2]
        height = video.shape[1]
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i in range(video):
            frame = video[i]
            video_writer.write(frame)

        video_writer.release()
        print(f"success rate: {count / num_vis}")
    
def main(**kwargs):
    # import diffuser.utils as utils
    # from maze_exp.config.default_args import Args

    # config
    cold_diffusion=False # inference method
    n_data = Args.num_tests

    update_args(**kwargs)
    # set env
    env, renderer = setup_env()

    # wandb.init(project=f"Maze2D2",config=vars(Args),name=f"in-{Args.include_goal_in_state}-pin-{Args.pin_goal}-hor-{int(Args.short_horizon)}",reinit=True)

    def get_dataset(Args):
        if Args.pin_goal:
            dataset = datasets.GoalDataset(
                env=Args.dataset,
                horizon=Args.short_horizon,
                normalizer=Args.normalizer,
                preprocess_fns=Args.preprocess_fns,
                use_padding=Args.use_padding,
                max_n_episodes=Args.max_n_episodes,
                max_path_length=Args.max_path_length,
            )
        else:
            dataset = datasets.SequenceDataset(
                env=Args.dataset,
                horizon=Args.short_horizon,
                normalizer=Args.normalizer,
                preprocess_fns=Args.preprocess_fns,
                use_padding=Args.use_padding,
                max_n_episodes=Args.max_n_episodes,
                max_path_length=Args.max_path_length,
            )

        val_dataset = datasets.GoalDataset(
            env=Args.dataset,
            horizon=Args.horizon,
            normalizer=Args.normalizer,
            preprocess_fns=Args.preprocess_fns,
            use_padding=Args.use_padding,
            max_n_episodes=Args.max_n_episodes,
            max_path_length=Args.max_path_length,
            validation=True
        )

        return dataset, val_dataset

    def get_diffuser(Args, dataset, val_dataset):

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
        )
        diffusion.to(Args.device)
        # diffusion.load_state_dict(torch.load("changing_goal_start/state_100.pt")['model'])


        trainer = main_trainer.Trainer(
            diffusion,
            dataset,
            val_dataset,
            renderer=None,
            train_batch_size=Args.batch_size,
            train_lr=Args.learning_rate,
            gradient_accumulate_every=Args.gradient_accumulate_every,
            ema_decay=Args.ema_decay,
            sample_freq=Args.sample_freq,
            save_freq=Args.save_freq,
            log_freq=Args.log_freq,
            label_freq=int(Args.n_train_steps // Args.n_saves),
            save_parallel=Args.save_parallel,
            results_folder=Args.snapshot_root,
            bucket=Args.bucket,
            n_reference=Args.n_reference,
            n_samples=Args.n_samples,
        )

        return trainer

    dataset, val_dataset = get_dataset(Args)

    trainer = get_diffuser(Args, dataset, val_dataset)

    n_epochs = int(Args.n_train_steps // Args.n_steps_per_epoch)

    param_string = "noise_" + str(Args.forward_sample_noise) \
        + "_weight_" + str(Args.action_weight) \
        + "_horizon_" + str(Args.horizon) \
        + "_epoch_" + str(int(Args.n_train_steps/Args.n_steps_per_epoch))

    loss_log = {"train": [], "validation":[]}
    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} | {Args.snapshot_root}')
        train_loss, val_loss = trainer.train(n_train_steps=Args.n_steps_per_epoch)
        loss_log["train"].append(train_loss)
        loss_log["validation"].append(val_loss)

        model_path = Path(Args.snapshot_root) / wandb.run.entity / wandb.run.project / wandb.run.group / wandb.run.id / f"{param_string}.pt"
        model_path.parent.mkdir(0o775, parents=True, exist_ok=True)

        # with open('loss_log/'+param_string+'.pickle', 'wb') as f:
        #     pickle.dump(loss_log, f)
        # model_path = Args.snapshot_root + "/" + param_string + ".pt"
        # with open("log.txt", "a") as f: # Open the file for writing
        #     f.write(f"{param_string}: {i}\n")

        print(f'Saving model to {model_path}')
        trainer.save(model_path)
    
    diffusion = trainer.model

    print("#############################################")
    print("Start evaluation")

    # set up infernece data
    if Args.condition == 'from env':
        val_traj = None
        batch = n_data
        cond = prepare_cond_for_random_rb(n_data)
        raise NotImplementedError("Visualization for this option is not implemented now.")
    elif Args.condition == 'from_val_tail':
        # get expert
        val_traj = prepare_val_traj(env, Args, trim="tail")
        val_traj = torch.from_numpy(val_traj).to(Args.device)
        batch = val_traj.shape[0]

        # get condition
        cond = get_condition(val_traj, Args)
    elif Args.condition == 'from_val_head':
        # get expert
        val_traj = prepare_val_traj(env, Args, trim="head")
        val_traj = torch.from_numpy(val_traj).to(Args.device)
        batch = val_traj.shape[0]

        # get condition
        cond = get_condition(val_traj, Args)
    else:
        raise NotImplementedError("TODO")

    # inference
    if Args.prediction == 'random_rb':
        shape = (batch, Args.horizon, Args.observation_dim+Args.action_dim)
        pred_traj = diffusion.backward_sample_loop(shape, cond, verbose=False, cold=cold_diffusion)
    elif Args.prediction == 'reconstruct':
        pred_traj = pred_traj_from_expert(val_traj, cond, diffusion, cold=cold_diffusion)
    else:
        raise NotImplemented("TODO")

    # rendering
    outdir=Path(os.getenv("LMN_OUTPUT_DIR", ".")) / 'controller' / Args.condition / Args.prediction
    evaluate(renderer, val_traj, pred_traj[-1], outdir, num_vis=Args.num_tests, run_controller=Args.controller)

if __name__ == '__main__':
    import os
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        cvd = 2
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cvd)

    kwargs = {}
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="cdrb",
        config=vars(Args),
    )
    main(**kwargs)
    wandb.finish()