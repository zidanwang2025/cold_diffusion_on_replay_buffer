import os
import gymnasium as gym
import numpy as np
from time import sleep
import json


def filter_states(env_name, generated_states, model, offset=0.02):
    '''filter out states that are in the wall for diffuser only'''
    range_dict = {
            'FetchPushObstacle-v2': {'x_min': 1.22, 'x_max': 1.38, 'y_min': 0.728, 'y_max': 0.772, 'z_min': 0.1, 'z_max': 0.5},
            'FetchPickObstacle-v2': {'x_min': 1.15, 'x_max': 1.45, 'y_min': 0.728, 'y_max': 0.772, 'z_min': 0.1, 'z_max': 0.5},
            'FetchReachObstacle-v2': {'x_min': 1.15, 'x_max': 1.45, 'y_min': 0.695, 'y_max': 0.705, 'z_min': 0.17, 'z_max': 0.77},
            'FetchReachObstacle-v2_2': {'x_min': 1.15, 'x_max': 1.45, 'y_min': 0.835, 'y_max': 0.845, 'z_min': 0.17, 'z_max': 0.77},
            }


    check_gripper_dict = {
            'FetchPushObstacle-v2': True,
            'FetchPickObstacle-v2': True,
            'FetchReachObstacle-v2': True,
            }
    check_block_dict = {
            'FetchPushObstacle-v2': True,
            'FetchPickObstacle-v2': True,
            'FetchReachObstacle-v2': True,
            }


    in_wall = 0
    index_to_remove = []
    check_gripper = check_gripper_dict[env_name]

    check_block = check_block_dict[env_name]
        
    for i, traj in enumerate(generated_states):
        for j, state in enumerate(traj):
            if check_gripper:
                if (range_dict[env_name]['x_max'] > state[0] > range_dict[env_name]['x_min'] and 
                range_dict[env_name]['y_max'] > state[1] > range_dict[env_name]['y_min'] and 
                range_dict[env_name]['z_max'] > state[2] > range_dict[env_name]['z_min']):              
                    in_wall += 1
                    index_to_remove.append(i)
                    break
                if env_name == 'FetchReachObstacle-v2':
                    if (range_dict[env_name+'_2']['x_max'] > state[0] > range_dict[env_name+'_2']['x_min'] and
                    range_dict[env_name+'_2']['y_max'] > state[1] > range_dict[env_name+'_2']['y_min'] and
                    range_dict[env_name+'_2']['z_max'] > state[2] > range_dict[env_name+'_2']['z_min']):
                        in_wall += 1
                        index_to_remove.append(i)
                        break

            if check_block and env_name != 'FetchReachObstacle-v2':
                if (range_dict[env_name]['x_max']+offset > state[3] > range_dict[env_name]['x_min']-offset and 
                range_dict[env_name]['y_max']+offset > state[4] > range_dict[env_name]['y_min']-offset and 
                range_dict[env_name]['z_max']+offset > state[5] > range_dict[env_name]['z_min']-offset):              
                    in_wall += 1
                    index_to_remove.append(i)
                    break
                
    percentage = 1 - in_wall / len(generated_states)
    generated_states = np.delete(generated_states, index_to_remove, axis=0)

    return percentage, generated_states