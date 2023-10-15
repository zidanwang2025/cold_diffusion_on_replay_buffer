# Terminal Commands

This repository contains code for training and testing diffusion models in different settings. The contents are designed for the Gymnasium fetch environment (https://github.com/Farama-Foundation/Gymnasium-Robotics/tree/main/gymnasium_robotics/envs/fetch).

For default configurations, refer to the following files:
- CDRB models: `cdrb/experiment_config.py`
- Diffuser: `diffuser/experiment_config.py`


<br>
<br>

## CDRB Model Settings

CDRB models can be trained, tested, and visualized in different settings.

First set current directory to ```cdrb```.

### Individual model training  

```python launcher.py --n-diffusion-steps 16 --join-action-state True --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1000 --horizon 56 --short-horizon 56 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPickObstacle-v2-normalized.hdf5 --dataset FetchPickObstacle-v2 --nickname normed_weight0_16 --weight-loss-by-current-radius 0```

- Set `--horizon` and `--short-horizon` to equal values, which must be multiples of 4.
- Set `--join-action-state False` to use a replay buffer derived from calculating k-clusters or Euclidean distance for states and actions separately. This is in effect in the forward process of `cdrb/diffusion_models.py` and `replay_buffer.py`.
- Trained models are automatically saved to `cdrb/all_model`. See file names for corresponding settings.
- The "epoch" in a model's file name is `n-train-steps/n-steps-per-epoch`. Make sure to use the same configuration when visualizing or testing.
- Refer to this table for corresponding environments and settings

| Fetch Envs | horizon | observation-dim |
|----------|----------|----------|
| FetchReach | 28 | 10 |
| FetchReachObstacle | 28 | 10 |
| FetchPickAndPlace | 56 | 25 |
| FetchPickObstacle | 56 | 25 |
| FetchPush | 56 | 25 |
| FetchPushObstacle | 56 | 25 |


<br>
<br>

### Bulk training and evaluation
```python generate_shell_script.py```

- The program generates a shell script called `train_eval_command.sh` for training the models on six Fetch environments and run evaluation after training.
- It will automatically log the results on Weights and Biases, so make sure that you are logged in before running the script.


<br>
<br>

### Test generated actions for success rate (directly from CDRB model)
```python test_action_success_rate.py --n-test 500 --cond-start 0 --cond-end 6 --test-validation True --validation-mode succeeded --model-type diffuser --n-diffusion-steps 16 --join-action-state True --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1000 --horizon 56 --short-horizon 56 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPickObstacle-v2-normalized.hdf5 --dataset FetchPickObstacle-v2 --nickname normed_weight0_16 --weight-loss-by-current-radius 0```

- The program tests if the goal is reached using the generated actions directly from the CDRB model.

- the start and goal in each test are randomly generated

- set ```--n-test``` to any number of tests desired. 

- the result will be printed in the terminal.

<br>
<br>


## Diffuser Settings

First set current directory to ```diffuser```.

### Individual model training

```python launcher.py --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 56 --short-horizon 56 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPickObstacle-v2-normalized.hdf5 --dataset FetchPickObstacle-v2 --nickname normed_16```

- Set `--horizon` and `--short-horizon` to equal values, which must be multiples of 4.
- Trained models are automatically saved to `diffuser/all_model`. See file names for corresponding settings.
- The "epoch" in a model's file name is `n-train-steps/n-steps-per-epoch`. Make sure to use the same configuration when visualizing or testing.

<br>
<br>

### Bulk training and evaluation
```python generate_shell_script.py```

- The program generates a shell script called `train_eval_command.sh` for training the models on six Fetch environments and run evaluation after training.
- It will automatically log the results on Weights and Biases, so make sure that you are logged in before running the script.

<br>
<br>

### Test generated actions for success rate (directly from diffuser model)

```python test_action_success_rate.py --n-test 500 --cond-start 0 --cond-end 6 --test-validation True --validation-mode succeeded --model-type diffuser --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 56 --short-horizon 56 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPickObstacle-v2-normalized.hdf5 --dataset FetchPickObstacle-v2 --nickname normed_16```

- The program tests if the goal is reached using the generated actions directly from the CDRB model.

- the start and goal in each test are randomly generated

- set ```--n-test``` to any number of tests desired. 

- the result will be printed in the terminal.

