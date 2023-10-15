# Terminal Commands

This repository contains code for training and testing diffusion models in different settings. The contents are designed for the Gymnasium maze environment (https://github.com/Farama-Foundation/Gymnasium-Robotics/tree/main/gymnasium_robotics/envs/maze).

For default configurations, refer to the following files:
- CDRB models: `cdrb/experiment_config.py`
- Diffuser: `diffuser/experiment_config.py`


<br>
<br>

## CDRB Model Settings

CDRB models can be trained, tested, and visualized in different settings.

First set current directory to ```cdrb```.

### Train kmeans model

```python3 launcher.py  --join-action-state True --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1000 --horizon 200 --short-horizon 200 --n-train-steps 20000 --n-steps-per-epoch 100```

- Set `--horizon` and `--short-horizon` to equal values, which must be multiples of 4.
- Use `--trim-buffer-mode euclidean` and `--d-distance 0.5` instead of `--k-cluster` for a Euclidean replay buffer.
- Set `--join-action-state False` to use a replay buffer derived from calculating k-clusters or Euclidean distance for states and actions separately. This is in effect in the forward process of `cdrb/diffusion_models.py` and `replay_buffer.py`.
- Trained models are automatically saved to `cdrb/all_model`. See file names for corresponding settings.
- The "epoch" in a model's file name is `n-train-steps/n-steps-per-epoch`. Make sure to use the same configuration when visualizing or testing.
- The training process is logged to `cdrb/loss_log` and `cdrb/log.txt`.
  - `loss_log` contains pickle files of loss dictionaries `{"train": [1.1, 0.9, ...], "validation": [1.3, 1.0, ...]}`.

<br>
<br>

### Training and evaluation with one code for sweep
```python3 launcher.py Train_and_Eval.py --sweep <path-to-json> -l <line-number>```

- Choice '--dataset' from 'gymnasium-corner-env-standard' or 'gymnasium-corner-env-tight'
- Choice '--diffusion' from 'models.ColdDiffusionRB' or 'models.GaussianDiffusion'
- Choice '--model' from 'models.TemporalUnet' (Convolution based) or 'models.TemporalTransformer' (Transformer based)

- '--condition' determine how to get start and goal points that are conditioned to model during evaluation. If you choose 'from_val_tail' start and goal points are obtained from validation dataset. Otherwise, start and goal points are randomly sampled from the environment. I recommend to choose 'from_val_tail'
- '--prediction' determine how to sample the initial noise. 'random_rb' indicates that the initial sample is obtrained randomly. If --difusion is 'models.ColdDiffusionRB', then it is sampled from the replay buffer. In contrast, if --difusion is 'models.GaussianDiffusion', it is sampled from a gaussian distribution without using the replay buffer.

<br>
<br>

### Test generated actions for success rate (directly from CDRB model)

```python3 test_diffusion_action_success_rate.py  --join-action-state True --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1000 --short-horizon 128 --render-mode random_rb --num-tests 1000 --n-train-steps 20000 --n-steps-per-epoch 100```

- The program tests if the goal is reached using the generated actions directly from the CDRB model.

- the start and goal in each test are randomly generated

- set ```--num-tests``` to any number of tests desired. 

- the result will be printed in the terminal.

<br>
<br>


## Diffuser Settings

First set current directory to ```diffuser```.

### Train diffuser model

```python3 launcher.py --forward-sample-noise 0.0 --action-weight 1 --horizon 128 --short-horizon 128 --n-train-steps 20000 --n-steps-per-epoch 100```

- Set `--horizon` and `--short-horizon` to equal values, which must be multiples of 4.
- Trained models are automatically saved to `diffuser/all_model`. See file names for corresponding settings.
- The "epoch" in a model's file name is `n-train-steps/n-steps-per-epoch`. Make sure to use the same configuration when visualizing or testing.
- The training process is logged to `diffuser/loss_log` and `diffuser/log.txt`.
  - `loss_log` contains pickle files of loss dictionaries `{"train": [1.1, 0.9, ...], "validation": [1.3, 1.0, ...]}`.

<br>
<br>

### Test generated actions for success rate (directly from diffuser model)

```python3 test_diffusion_action_success_rate.py --forward-sample-noise 0.0 --action-weight 1 --short-horizon 128 --render-mode random_rb --num-tests 1000 --n-train-steps 20000 --n-steps-per-epoch 100```

- The program tests if the goal is reached using the generated actions directly from the diffuser model.

- the start and goal in each test are randomly generated

- set ```--num-tests``` to any number of tests desired. 

- the result will be printed in the terminal.

