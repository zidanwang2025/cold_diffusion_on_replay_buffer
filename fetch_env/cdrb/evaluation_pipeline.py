'''
python evaluation_pipeline.py --model-type CDRB --test-validation True --validation-mode succeeded --cond-start 0 --cond-end 6 --join-action-state True --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1000 --horizon 36 --short-horizon 36 --n-train-steps 20000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPickAndPlace-v2-no-images.hdf5 --dataset FetchPickAndPlace-v2 --nickname ogdata --control-goal-size 25 --set-seed 271969
'''
import os
import sys
import json
import h5py
import numpy as np
from experiment_config import Args
from save_generated_states import update_args, setup, save_traj
from json_controller_test import test_controller
from check_in_wall import filter_states


def read_json():
    data_path = f'../saved_trajectories/{Args.dataset}/{Args.model_type}_seed_{Args.set_seed}_nickname_{Args.nickname}_epoch_{int(Args.n_train_steps/100)}.json'
    if not os.path.exists(data_path):
        print("generating and saving trajectories")
        diffusion, episode_data = setup()
        save_traj(diffusion, episode_data)
    else:
        print("loading saved trajectories")
    with open(data_path, 'r') as f:
        generated_states = json.load(f)
        
    generated_states = [item['states'] for item in generated_states]
    generated_states = np.array(generated_states)

    return generated_states

def main():
    # Generate and save diffusion trajectories
    update_args()

    generated_states = read_json()

    if "Obstacle" in Args.dataset:
        weight, weighted_generated_states = filter_states(Args.dataset, generated_states, Args.model_type, offset=0.01)
    else:
        weight = 1
        weighted_generated_states = None

    if "PickAndPlace" in Args.dataset:
        allowed_action = 6
    elif "PickObstacle" in Args.dataset:
        allowed_action = 2
    elif "PushObstacle" in Args.dataset:
        allowed_action = 1
    elif "Push" in Args.dataset:
        allowed_action = 9
    elif "ReachObstacle" in Args.dataset:
        allowed_action = 13
    elif "Reach" in Args.dataset:
        allowed_action = 1

    test_controller(generated_states, 
                    Args.dataset, 
                    Args.observation_dim, 
                    Args.observation_dim, 
                    s=allowed_action, 
                    model=Args.model_type,
                    seed=Args.set_seed,
                    weight=weight,
                    weighted_generated_states=weighted_generated_states)

if __name__ == "__main__":
    main()