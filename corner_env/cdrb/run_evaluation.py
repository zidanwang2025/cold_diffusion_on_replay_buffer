#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
import json
import torch
import cv2
import numpy as np
import wandb
import matplotlib.pyplot as plt
import gymnasium as gym
from experiment_config import Args
from helpers import get_dataset, get_diffuser, shift_range, get_maze_range
from utils1.helpers import apply_conditioning, get_condition, WaypointController

# HACK: append the parent dir to sys.path to deal with python's stupid import system
from pathlib import Path
import sys
curr_dir = Path(__file__).resolve().parent
sys.path.append(str(curr_dir.parent))
from gymnasium_corner_env_renderer import CornerEnvRenderer
from gymnasium_registration import initialize_env


def render_and_tonumpy(renderer, trajectory):
    if Args.dataset == "gymnasium-corner-env-archive":
        ratio = 7/6
    else:
        ratio = 15/13

    fig, ax = plt.subplots(figsize=(8, 8))
    renderer.render_trajectory(trajectory, ax, ratio, reset=True, pin=True, gif=False)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    numpy_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
    plt.close()
    return numpy_array


def setup_env():
    done = False
    while not done:
        try:
            env = gym.make(Args.dataset)
            done = True
        except ValueError:
            print("try again")
        except KeyboardInterrupt:
            sys.exit()
    env.reset()

    renderer = CornerEnvRenderer(env, multi=True)
    return env, renderer


def prepare_cond_for_random_rb(env, number_of_data):

    start_list, goal_list = [], []
    normalize_constants = get_maze_range(Args)
    for i in range(number_of_data):
        env.reset()
        start = env.reset_pos
        goal = env.goal
        start = shift_range(start, normalize_constants[0], normalize_constants[1], -1.0, 1.0)
        goal = shift_range(goal, normalize_constants[0], normalize_constants[1], -1.0, 1.0)
        start_list.append(start)
        goal_list.append(goal)
    starts = np.stack(start_list, axis=0)
    goals = np.stack(goal_list, axis=0)

    # range (-1, 1)
    cond = {0: torch.FloatTensor(starts), Args.horizon-1: torch.FloatTensor(goals)}
    cond = {key: val.to(Args.device) for key, val in cond.items()}

    return cond

def prepare_val_trajs(env, Args, trim="tail", patch_dataset=True):
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
            if patch_dataset and len(current_episode) > 1:
                # Due to a bug, the last state is being random, so overwrite it with the one before.
                current_episode[-1][2:] = current_episode[-2][2:]
                dataset['observations'][i] = dataset['observations'][i-1]

            # TODO TY: Why do we need this?
            # Duplicate the last action and observation to achieve len(current_episode) % 4 == 0
            while state_counter % 4 != 0:
                current_episode.append(np.hstack((dataset['actions'][i], dataset['observations'][i])))
                state_counter += 1

            # If the length of episode is shorter than Args.horizon, skip this episode
            if len(current_episode) >= Args.horizon:
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

def pred_traj_from_expert(val_traj, cond, diffusion, cold=False, cond_vels=True):
    batch_size = val_traj.shape[0]

    noise = torch.randn_like(val_traj).to(Args.device)
    timesteps = torch.full((batch_size,), (Args.n_diffusion_steps - 1), dtype=torch.long).to(Args.device)

    x_noisy = diffusion.forward_sample(val_traj, timesteps, noise)
    x_noisy = apply_conditioning(x_noisy, cond, Args.action_dim, cond_vels=cond_vels)
    x_noisy = x_noisy.to(dtype=torch.float)

    pred_traj = [x_noisy]
    for i in reversed(range(1, Args.n_diffusion_steps)):
        print('backward step:', i)
        timesteps = torch.full((batch_size,), i, dtype=torch.long).to(Args.device)
        if cold:
            x_noisy = diffusion.backward_sample_cold(x_noisy, cond, timesteps)
        else:
            x_noisy = diffusion.backward_sample(x_noisy, cond, timesteps)
        x_noisy = apply_conditioning(x_noisy, cond, Args.action_dim, cond_vels=cond_vels)
        pred_traj.append(x_noisy)

    return pred_traj


def run_actions(env, action_list):
    """Simply execute the given list of actions in the env."""
    exec_traj = []

    for i, action in enumerate(action_list):
        num_steps = 0
        obs, reward, terminated, truncated, info = env.step(action)

        current_location = obs['observation'][:2]
        exec_traj.append(current_location)

        num_steps += 1
        if info['success']:
            print("Success!")
            return exec_traj, True

    return exec_traj, False


def run_controller(env, trajectory):
    """Execute WaypointController in the env."""
    exec_traj = []
    controller = WaypointController(solve_thresh=0.1)
    current_location = env.reset_pos  # Start location
    velocity = np.zeros(2)
    normalize_constants = get_maze_range(Args)

    for i, state in enumerate(trajectory[:-1]):
        target = np.array(trajectory[i+1, :2])
        target = shift_range(target, -1.0, 1.0, normalize_constants[0], normalize_constants[1])
        num_steps = 0

        # Allow large steps only for the last action
        allowed_steps = Args.num_action_per_step if i < len(trajectory) - 2 else 20
        while not controller.is_path_solved(current_location, target) and num_steps < allowed_steps:
            action = controller.get_action(current_location, velocity, target)

            obs, reward, terminated, truncated, info = env.step(action)

            velocity = obs['observation'][2:]
            current_location = obs['observation'][:2]
            exec_traj.append(current_location)

            num_steps += 1
            if info['success']:
                print("Success!")
                return exec_traj, True

    return exec_traj, False


def make_eval_episodes(env, possible_goals, selection_strategy, Args):
    """Make episodes for evaluation.

    Args:
        possible_goals
        selection_strategy (str): 'from_env', 'from_val_tail', 'from_val_head'

    Returns:
        cond (dict): This has start and goal info
        batch_size (int)
        val_traj (np.ndarray): (n_data, horizon, state_dim)
    """
    n_data = Args.num_tests

    # For clarity
    assert selection_strategy in ['from_env', 'from_val_tail', 'from_val_head']

    if selection_strategy == 'from_env':
        val_trajs = None
        batch_size = n_data
        cond = prepare_cond_for_random_rb(env, n_data)
        raise NotImplementedError("Visualization for this option is not implemented now.")
    else:
        assert selection_strategy in ['from_val_tail', 'from_val_head']
        # get expert
        trim = 'tail' if selection_strategy == 'from_val_tail' else 'head'
        normalize_constants = get_maze_range(Args)
        val_trajs = prepare_val_trajs(env, Args, trim=trim)
        val_trajs = torch.from_numpy(val_trajs).to(Args.device)

        # Make it slightly more robust (new dataset has an extra dimension)
        assert val_trajs.ndim in [3, 4]
        if val_trajs.ndim == 4 and val_trajs.shape[2] == 1:
            val_trajs = val_trajs.squeeze(2)

        val_goals = val_trajs[:, -1, Args.action_dim:Args.action_dim + 2].cpu().numpy()  # (batch_size, 2)
        # val_goals = shift_range(val_goals, -1.0, 1.0, normalize_constants[0], normalize_constants[1])
        batch_size = val_trajs.shape[0]

        # Normalize back possible_goals
        possible_goals = np.array(possible_goals)
        possible_goals = shift_range(possible_goals, normalize_constants[0], normalize_constants[1], -1., 1.)

        # Figure out which goal the gt_trajectory is heading to
        # possible_goals: (4, 2)
        errors = np.sum((possible_goals[None, :, :] - val_goals[:, None, :]) ** 2, axis=-1)  # (batch_size, 4)
        min_error_inds = np.argmin(errors, axis=1)  # (batch_size,)
        orig_goals = possible_goals[min_error_inds]  # (batch_size, 2)

        # Replace the end of val_traj with orig_goals
        # (batch_size, horizon, state_dim)
        val_trajs[:, -1, Args.action_dim:Args.action_dim + 2] = torch.as_tensor(orig_goals)

        # get condition
        cond = get_condition(val_trajs, Args)

        # CRITICAL: Set the velocities to zero
        for timestep, state in cond.items():
            state[:, -2:] = 0.0  # Set the velocities to zero

    return cond, batch_size, val_trajs


def run_episode(ep_idx, episode_data, predicted_data, normalize_constants, traj_exec):
    log_dict = {}
    controller_imgs = {}

    _, renderer = setup_env()
    gt_trajectory = episode_data[ep_idx,:, Args.action_dim:]
    pred_trajectory = predicted_data[ep_idx,:, Args.action_dim:]
    gt_act_trajectory = episode_data[ep_idx, :, :Args.action_dim]
    act_trajectory = predicted_data[ep_idx, :, :Args.action_dim]

    # get goal for visualize
    start = pred_trajectory[0][:2]
    start = shift_range(start, -1.0, 1.0, normalize_constants[0], normalize_constants[1])
    val_goal = pred_trajectory[-1][:2]
    val_goal = shift_range(val_goal, -1.0, 1.0, normalize_constants[0], normalize_constants[1])

    if len(renderer.square_arr) != len(renderer.square_arr[0]):
        raise NotImplementedError("TODO")

    # TODO: This shouldn't be necessary once we set the end of pred_trajectory to be the original goal
    wall_length = 1 # perimeter wall
    length_from_wall_to_wall = len(renderer.square_arr) - (2 * wall_length)
    possible_goals = renderer.possible_goals - 0.5 - (1 - wall_length)
    possible_goals = [shift_range(goal, 0, length_from_wall_to_wall, normalize_constants[0], normalize_constants[1]) for goal in possible_goals]

    # Figure out which goal the gt_trajectory is heading to
    errors = np.sum((np.array(possible_goals) - val_goal) ** 2, axis=1)
    min_error_index = np.argmin(errors)
    orig_goal = possible_goals[min_error_index]  # env coordinate (something like [-6, 6])

    gt_trajectory = gt_trajectory[:, :2]
    gt_image_array = render_and_tonumpy(renderer, gt_trajectory)
    pred_trajectory = pred_trajectory[:, :2]
    pred_image_array = render_and_tonumpy(renderer, pred_trajectory)

    # compute cumulative distance of gt and predicted trajectories
    cumulative_distance_gt_traj = np.sum(np.linalg.norm(gt_trajectory[1:] - gt_trajectory[:-1], axis=1))
    cumulative_distance_pred_traj = np.sum(np.linalg.norm(pred_trajectory[1:] - pred_trajectory[:-1], axis=1))
    log_dict["distances/gt_traj"] = cumulative_distance_gt_traj
    log_dict["distances/pred_traj"] = cumulative_distance_pred_traj
    log_dict["distance_ratios/pred_traj"] = cumulative_distance_pred_traj / cumulative_distance_gt_traj
    
    gt_image_array = cv2.putText(gt_image_array, f'dist: {cumulative_distance_gt_traj:.4g}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    pred_image_array = cv2.putText(pred_image_array, f'dist: {cumulative_distance_pred_traj:.4g}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Execute the trajectory
    # initialize_env()
    done = False
    while not done:
        try:
            env = gym.make(Args.dataset)
            done = True
        except ValueError:
            print("try again")
        except KeyboardInterrupt:
            sys.exit()

    for _traj_exec in traj_exec:

        # Reset env
        env.reset()
        env.set_start_goal(start, orig_goal)  # Use the original goal

        # Execute the trajectory with the controller, or just execute the generated action
        if _traj_exec == 'controller':
            executed_traj, _success = run_controller(env, pred_trajectory)
        elif _traj_exec == 'generated_actions':
            executed_traj, _success = run_actions(env, act_trajectory)
        elif _traj_exec == 'gt_actions':
            executed_traj, _success = run_actions(env, gt_act_trajectory)
        else:
            raise ValueError(f'invalid value for traj_exec: {_traj_exec}')

        log_dict[f"success_rate/{_traj_exec}"] = int(_success)

        # Normalize the executed traj
        executed_traj = [
            shift_range(state, normalize_constants[0], normalize_constants[1], -1.0, 1.0)
            for state in executed_traj
        ]

        # compute the cumulative distance of executed traj
        cumulative_distance_exec = np.sum(np.linalg.norm(np.array(executed_traj[1:]) - np.array(executed_traj[:-1]), axis=1))
        log_dict[f"controller_steps/{_traj_exec}"] = len(executed_traj)
        log_dict[f"distances/{_traj_exec}"] = cumulative_distance_exec
        log_dict[f"distance_ratios/{_traj_exec}"] = cumulative_distance_exec / cumulative_distance_gt_traj
        if _success:
            log_dict[f'success_distance_ratios/{_traj_exec}'] = cumulative_distance_exec / cumulative_distance_gt_traj

        img = render_and_tonumpy(renderer, torch.FloatTensor(np.array(executed_traj)))
        img = cv2.putText(img, _traj_exec, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        img = cv2.putText(img, f'success: {_success}, controller_len: {len(executed_traj)} dist: {cumulative_distance_exec:.4g}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        controller_imgs[_traj_exec] = img

    frame = np.concatenate(
        (gt_image_array, pred_image_array, *[img for name, img in sorted(controller_imgs.items())]),
        axis=1
    )
    frame = cv2.putText(frame, f'val index: {ep_idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return frame, log_dict, ep_idx

def evaluate_traj(episode_data, predicted_data, num_vis=10, traj_exec=['controller'], epoch=0, multi=16):
    """
    Args:
      - episode_data (Tensor): (batch_size, episode_len, state_action_dim)
      - predicted_data (Tensor): same shape as above
      - traj_exec (List | None): list of controller types (['controller', 'generated_actions', None])
    """
    assert isinstance(traj_exec, list)

    frame_list = []
    log_dict = defaultdict(list)
    normalize_constants = get_maze_range(Args)
    if num_vis > len(predicted_data):
        print(f"num_vis is changed from {num_vis} to {len(predicted_data)}")
        num_vis = len(predicted_data)

    num_workers = multi
    initialize_env()
    # for ep_idx in range(num_vis):
    #     run_episode(ep_idx, episode_data, predicted_data, normalize_constants, traj_exec)
    executor = ProcessPoolExecutor(max_workers=num_workers)
    all_tasks = [
        executor.submit(
            run_episode,
            *(ep_idx, episode_data, predicted_data, normalize_constants, traj_exec)
        ) for ep_idx in range(num_vis)
    ]

    frame_list, index_list = [], []
    for ep_idx, task in enumerate(as_completed(all_tasks)):
        frame, ep_log_dict, index = task.result()
        for key in ep_log_dict.keys():
            log_dict[key].append(ep_log_dict[key])
        frame_list.append(frame)
        index_list.append(index)
    sorted_list = sorted(index_list)
    index_list = [index_list.index(x) for x in sorted_list]
    sorted_frame_list = [frame_list[i] for i in index_list]

    video = np.asarray(sorted_frame_list).transpose(0, 3, 1, 2)
    wandb_log = {"num_episodes": num_vis, "epoch":epoch, "video": wandb.Video(video, fps=5, format="mp4")}
    for key in log_dict.keys():
        wandb_log[key] = sum(log_dict[key]) / len(log_dict[key])
    wandb.log(wandb_log)

def prepare_model(model_path):
    dataset, val_dataset = get_dataset(Args)
    trainer, param_string = get_diffuser(Args, dataset, val_dataset)
    diffusion = trainer.model
    print(f'Loading the model from {model_path}...')
    diffusion.load_state_dict(torch.load(model_path)['model'])
    print(f'Loading the model from {model_path}...done')
    return diffusion


def evaluate(diffusion, selection_strategy, epoch, multi=16):
    # Set up the environment
    env, renderer = setup_env()

    # Set up the evaluation episodes
    normalize_constants = get_maze_range(Args)
    wall_length = 1 # perimeter wall
    length_from_wall_to_wall = len(renderer.square_arr) - (2 * wall_length)
    possible_goals = renderer.possible_goals - 0.5 - (1 - wall_length)
    possible_goals = [shift_range(goal, 0, length_from_wall_to_wall, normalize_constants[0], normalize_constants[1]) for goal in possible_goals]
    cond, batch_size, eval_traj  = make_eval_episodes(env, possible_goals, selection_strategy, Args)

    # Generate a batch of trajectories from `cond`
    cold_diffusion = False
    if Args.prediction == 'random_rb':
        shape = (batch_size, Args.horizon, Args.observation_dim+Args.action_dim)
        pred_traj = diffusion.backward_sample_loop(shape, cond, verbose=False, cold=cold_diffusion)
    elif Args.prediction == 'reconstruct':
        assert selection_strategy != 'from_env'
        pred_traj = pred_traj_from_expert(eval_traj, cond, diffusion, cold=cold_diffusion, cond_vels=Args.cond_vels)
    else:
        raise NotImplemented("TODO")

    # Evaluate and render the generated trajectory
    assert selection_strategy != 'from_env'
    evaluate_traj(eval_traj.cpu().numpy(), pred_traj[-1].cpu().numpy(), num_vis=Args.num_tests, traj_exec=Args.traj_exec, epoch=epoch, multi=multi)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_dir', default=None)
    parser.add_argument('-m', '--multi', type=int, default=16)
    parser.add_argument('-l', '--line-number', type=int, default=None, required=True)
    parser.add_argument('-e', '--epoch', action='store_true')
    args = parser.parse_args()

    # Load args from the sweep file and the metainfo file
    sweep_file = Path(args.sweep_dir) / 'sweep.jsonl'
    from params_proto.hyper import Sweep
    sweep = Sweep(Args).load(str(sweep_file.resolve()))
    kwargs = list(sweep)[args.line_number]

    # Load metainfo from the json file
    metainfo_file = str((Path(args.sweep_dir) / 'meta.json').resolve())
    with open(metainfo_file, 'r') as f:
        metainfo = json.load(f)

    run_entity = metainfo['entity']
    run_project = metainfo['proj']
    group = metainfo['group']
    run_id = metainfo['run_ids'][args.line_number]

    # TEMP:
    # from textwrap import dedent
    # run_entity = 'takuma-yoneda'
    # run_project = 'cdrb-newgoals'
    # group = 'takuma-20230531-longlong-horizon'
    # run_id = '20ucj5lt'
    # json_config = dedent('''
    # {"model": "models.TemporalTransformer", "n_test": 1, "bucket": null, "commit": null, "config": "config.maze2d", "device": "cuda", "loader": "datasets.SequenceDataset", "prefix": "diffusion/", "dataset": "gymnasium-corner-env-tight", "horizon": 180, "logbase": "logs", "n_saves": 50, "exp_name": "diffusion/H384_T256", "log_freq": 10, "max_dist": 1.4142135623730951, "pin_goal": true, "renderer": "utils.Maze2dRenderer", "save_gif": true, "condition": "from_val_tail", "data_path": null, "diffusion": "models.ColdDiffusionRB", "dim_mults": [1, 4, 8], "ema_decay": 0.99, "k_cluster": 2500, "loss_type": "l2", "n_samples": 5, "num_tests": 100, "save_freq": 1000, "traj_exec": ["generated_actions", "controller"], "action_dim": 2, "batch_size": 320, "controller": true, "d_distance": 0.5, "normalizer": "LimitsNormalizer", "prediction": "random_rb", "repeat_len": 2, "n_condition": 1, "n_reference": 50, "render_mode": "expert", "sample_freq": 5000, "use_padding": false, "loss_weights": null, "action_weight": 1, "clip_denoised": true, "learning_rate": 0.0002, "loss_discount": 1, "n_train_steps": 20000, "save_parallel": false, "short_horizon": 180, "shorten_ratio": 1, "snapshot_root": "/cdrb/snapshot_root", "cdrb_add_noise": false, "dist_scheduler": "linear", "max_n_episodes": 1000, "preprocess_fns": [], "shuffle_length": 0, "visualize_mode": "state", "max_path_length": 400, "observation_dim": 4, "predict_epsilon": false, "trim_buffer_mode": "kmeans", "join_action_state": true, "n_diffusion_steps": 500, "n_steps_per_epoch": 100, "num_action_per_step": 10, "termination_penalty": null, "forward_sample_noise": 0, "include_goal_in_state": false, "gradient_accumulate_every": 2}
    # ''')
    # kwargs = json.loads(json_config)

    Args._update(kwargs)

    # Overwrite some args
    Args.snapshot_root = '/cdrb/snapshot_root'
    Args.num_action_per_step = 5  # Number of steps allowed to reach the next waypoint

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="cdrb-test",
        group="temp",
        config={**vars(Args), 'run_id': run_id, 'run_project': run_project},
        # id=run_id,  # Let's use the same run_id
    )

    # HACK: From the model_path (directory), there should be a single `.pt` file. Let's identify and load from it.
    model_dir = Path(Args.snapshot_root) / run_entity / run_project / group / run_id
    files = list(model_dir.iterdir())
    files.sort()
    if args.epoch:
        for i, model_path in enumerate(files[1:]):
            print(f"\n\nepoch: {i}/{len(files)-2}")
            # Prepare the model
            diffusion = prepare_model(model_path)
            evaluate(diffusion, selection_strategy='from_val_tail',epoch=i, multi=args.multi)
    else:
        diffusion = prepare_model(files[0])
        evaluate(diffusion, selection_strategy='from_val_tail',epoch=(len(files)-2), multi=args.multi)