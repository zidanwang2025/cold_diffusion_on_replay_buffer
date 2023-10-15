import json
import os
import gymnasium as gym
import imageio
import numpy as np
import torch
import h5py
from tqdm import tqdm
import sys
import argparse
import wandb
from collections import deque
from check_in_wall import filter_states
from experiment_config import Args

class FetchPushControl:
    def __init__(self):
        self.last_k_actions = deque(maxlen=3)
        self.reset_phase = 0
        self.reset_step_counter = 0
        self.reset_gripper = True
        self.prev_state = None
        self.prev_action = None
        self.stuck = False
        self.stuck_attempt = 0
        self.manual_approach_pos = None
        self.attempt_manual = 0
        
    def get_offset(self, block_pos, attempts):
        table_center = np.array([1.3, 0.744, 0.42])
        difference = block_pos - table_center
        
        if attempts == 0:
            if difference[1] < 0:
                return np.array([0, 0.02, 0])
            else:
                return np.array([0, -0.02, 0])
        else:
            if difference[0] < 0:
                return np.array([0.02, 0, 0])
            else:
                return np.array([-0.02, 0, 0])

    def get_action(self, current_state, next_state):
        gripper_pos = current_state[0:3]
        block_pos = current_state[3:6]
        target_pos = next_state[3:6]

        block_to_target_dir = target_pos - block_pos
        block_to_target_dir_norm = np.linalg.norm(block_to_target_dir)

        if current_state[5] < 0.4:
            self.reset_gripper=False

        if self.prev_state is not None and self.prev_action is not None:
            state_change = np.linalg.norm(current_state - self.prev_state)
            action_magnitude = np.linalg.norm(self.prev_action)
            if state_change < 0.0005 and action_magnitude > 0.01:
                self.stuck = True
                self.manual_approach_pos = block_pos + self.get_offset(block_pos, self.attempt_manual%2)
                self.attempt_manual += 1
                self.reset_phase = 1
                action = [0, 0, 0]
    
        if self.manual_approach_pos is None:
            approach_position = block_pos - 0.11 * (block_to_target_dir / block_to_target_dir_norm)
        else: 
            approach_position = self.manual_approach_pos

        if self.reset_gripper == True:
            if self.reset_phase == 0: 
                action = [0, 0, 1]
                self.reset_step_counter += 1
                if self.reset_step_counter == 4:
                    self.reset_step_counter = 0
                    self.reset_phase += 1
            elif self.reset_phase == 1: 
                if np.linalg.norm((gripper_pos-approach_position)[:2])>0.05:
                    action = (approach_position - gripper_pos)*20
                    action = np.append(action[:2],0)
                else:
                    if current_state[2] > 0.42:
                        action = [0, 0, -1]
                    else:
                        if self.stuck == True:
                            self.stuck == False
                            self.manual_approach_pos = None
                        action = [0, 0, 0]
                        self.reset_phase +=1
            elif self.reset_phase == 2: 
                if np.linalg.norm((block_pos-gripper_pos)[:2])>0.05:
                    action = block_pos-gripper_pos
                    action *= 3
                else:
                    action = [0, 0, 0]
                    self.reset_gripper = False
        else:
            if np.linalg.norm((block_pos-gripper_pos)[:2])<0.1:
                d = np.linalg.norm(current_state[6:9])
                action_multiplier = 8 * np.log(d + 2)
                action = np.append(block_to_target_dir[:2],0)*action_multiplier
            else: 
                action = [0, 0, 0]
                if current_state[5] > 0.4:
                    self.reset_gripper = True
                    self.reset_phase = 0

        grip = 0.0
        action = np.append(action, grip)

        self.prev_state = current_state
        self.prev_action = action

        return action


class FetchReachControl:
    def __init__(self):
        self.last_k_actions = deque(maxlen=7)
    
    def get_action(self, current_state, next_state):
        d = np.linalg.norm(next_state[0:3] - current_state[0:3])
        action_multiplier = 20
        action = next_state[0:3] - current_state[0:3]
        action = action * action_multiplier
        self.last_k_actions.append(np.squeeze(action))
        action = np.mean(np.array(self.last_k_actions), axis=0)
        action = np.append(action, 1.0)
        return action

class FetchPickControl:
    def __init__(self):
        self.last_k_actions = deque(maxlen=3)
        self.gripper_block_threshold = 0.2
        self.gripped = False
        self.initial_move = 0

    def get_action(self, current_state, next_state):
        d = np.linalg.norm(current_state[6:9])
        v = np.linalg.norm(current_state[14:17])
        action_multiplier = 10 * np.log(d + 2)
        if d < self.gripper_block_threshold:
            action = current_state[3:6] - current_state[0:3]
            if d < 0.05:
                grip = -1.0
            else:
                grip = 1.0
        else:
            if self.gripped:
                action =current_state[3:6] - current_state[0:3]
            else:
                action = next_state[0:3] - current_state[0:3]
            grip = 1.0

        if v < 0.005 and d < 0.05:
            self.gripped = True
            action = next_state[0:3] - current_state[0:3]
            grip = -1.0

        action = action * action_multiplier
        self.last_k_actions.append(np.squeeze(action))
        action = np.mean(np.array(self.last_k_actions), axis=0)
        action = np.append(action, grip)
        if self.initial_move < 10:
            action = np.array([0, 0, 1, 0])
            if current_state[4] - current_state[1] > 0:
                action[1] = 1
            else:
                action[1] = -1
            self.initial_move += 1
            return action
        return action

def setup(data_path):
    '''initialize env and set up diffusion model, renderer, param_string, episode_data'''

    with open(data_path, 'r') as f:
        generated_states = json.load(f)
    
    generated_states = [item['states'] for item in generated_states]
    generated_states = np.array(generated_states)
    
    return generated_states

def is_success(cur_state, desired_goal, env_name):
    '''check if the current state is close enough to the desired goal'''
    if "reach" not in env_name.lower():
        if np.linalg.norm(cur_state[3:6] - desired_goal[:3]) < 0.05:
            return True
        else:
            return False
    else:
        if np.linalg.norm(cur_state[:3] - desired_goal[:3]) < 0.05:
            return True
        else:
            return False

def test_controller(generated_states, 
                    env_name, 
                    observation_dim, 
                    goal_dim, 
                    s=3, 
                    model="diffuser",
                    seed=111,
                    weight=1,
                    weighted_generated_states=None):
    

    env = gym.make(env_name)
    env.set_fixed_block_pos(np.array([0, 0]))
    constants = env.reset()[0]['achieved_goal']

    if "reach" in env_name.lower():
        offset = 0
    else:
        offset = 3

    unweighted_success = 0
    for each_test in range(len(generated_states)):
        expert_traj = generated_states[each_test]
        expert_block = expert_traj[0, 3:6]
        expert_goal = expert_traj[-1, offset:offset+3]
        env.set_fixed_goal(expert_goal)
        if "reach" not in env_name.lower():
            env.set_fixed_block_pos(expert_block[:2] - constants[:2])
        obs, info = env.reset()        
        
        expert_goal = np.concatenate((expert_goal, expert_goal))
        cur_state = obs['observation']
        if "pick" in env_name.lower():
            controller = FetchPickControl()
        elif "push" in env_name.lower():
            controller = FetchPushControl()
        elif "reach" in env_name.lower():
            controller = FetchReachControl()
        
        reset_count = 0
        for i, next_state in enumerate(expert_traj):
            check_state = next_state.copy()
            if goal_dim == 6:
                next_state = np.concatenate((next_state[3:6], next_state[:3]))
            elif goal_dim == 3:
                next_state = next_state[3:6]
            count = 0
            while np.linalg.norm(cur_state[offset:offset+3] - check_state[offset:offset+3]) >= 0.01 and count < s:
                if "push" in env_name.lower():
                    if controller.reset_gripper == False:
                        reset_count = 0
                        count += 1
                    else:
                        reset_count += 1
                else:
                    count += 1
                env.update_goal(check_state[offset:offset+3])

                action = controller.get_action(cur_state, next_state)

                obs_dict, _, _, _, info = env.step(action)
                cur_state = obs_dict["observation"][:observation_dim]
                if is_success(cur_state, expert_goal, env_name):
                    break
                if "Push" in env_name and reset_count > 50:
                    break
            if is_success(cur_state, expert_goal, env_name):
                break
        if is_success(cur_state, expert_goal, env_name):
            unweighted_success += 1

    if weighted_generated_states is not None:
        weighted_success = 0

        for each_test in range(len(weighted_generated_states)):
            expert_traj = weighted_generated_states[each_test]
            expert_block = expert_traj[0, 3:6]
            expert_goal = expert_traj[-1, offset:offset+3]
            env.set_fixed_goal(expert_goal)
            if "reach" not in env_name.lower():
                env.set_fixed_block_pos(expert_block[:2] - constants[:2])
            obs, info = env.reset()        
            
            expert_goal = np.concatenate((expert_goal, expert_goal))
            cur_state = obs['observation']
            if "pick" in env_name.lower():
                controller = FetchPickControl()
            elif "push" in env_name.lower():
                controller = FetchPushControl()
            elif "reach" in env_name.lower():
                controller = FetchReachControl()
            
            reset_count = 0
            for i, next_state in enumerate(expert_traj):
                check_state = next_state.copy()
                if goal_dim == 6:
                    next_state = np.concatenate((next_state[3:6], next_state[:3]))
                elif goal_dim == 3:
                    next_state = next_state[3:6]
                count = 0
                while np.linalg.norm(cur_state[offset:offset+3] - check_state[offset:offset+3]) >= 0.01 and count < s:
                    if "push" in env_name.lower():
                        if controller.reset_gripper == False:
                            reset_count = 0
                            count += 1
                        else:
                            reset_count += 1
                    else:
                        count += 1
                    env.update_goal(check_state[offset:offset+3])

                    action = controller.get_action(cur_state, next_state)

                    obs_dict, _, _, _, info = env.step(action)
                    cur_state = obs_dict["observation"][:observation_dim]
                    if is_success(cur_state, expert_goal, env_name):
                        break
                    if "Push" in env_name and reset_count > 50:
                        break
                if is_success(cur_state, expert_goal, env_name):
                    break
            if is_success(cur_state, expert_goal, env_name):
                weighted_success += 1
    else:
        weighted_success = unweighted_success
        weighted_generated_states = generated_states

    dict = {"model": model,
            "env": env_name,
            "unweighted success rate": unweighted_success/len(generated_states)*100,
            "weight": weight,
            "weighted success rate": weighted_success/len(weighted_generated_states)*100*weight,
            "allowed actions": s,
            "seed": seed,
            "nickname": Args.nickname,
            "epoch": int(Args.n_train_steps/100)
            }

    with open("success_rate.jsonl", 'a') as f:
        json.dump(dict, f)
        f.write('\n')
