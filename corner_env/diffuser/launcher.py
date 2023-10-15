import pickle
import sys
import argparse

import numpy as np

import datasets
import main_trainer
from diffusion_models import GaussianDiffusion
from experiment_config import Args
from temporal_unet import TemporalUnet
import torch

sys.path.append("..")
from gymnasium_registration import initialize_env

initialize_env()

def main(**kwargs):
    # import diffuser.utils as utils
    # from maze_exp.config.default_args import Args

    # wandb.login()

    Args._update(kwargs)

    Args.short_horizon = Args.horizon * Args.shorten_ratio

    Args.horizon = int(Args.horizon)
    Args.short_horizon = int(Args.short_horizon)

    print("include goal:", Args.include_goal_in_state)
    print("pin goal:", Args.pin_goal)
    print("train horizon:", Args.short_horizon)

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
        with open('loss_log/'+param_string+'.pickle', 'wb') as f:
            pickle.dump(loss_log, f)
        # wandb.log({"loss":loss}, commit = True)
        model_path = Args.snapshot_root + "/" + param_string + ".pt"
        trainer.save(model_path)
        with open("log.txt", "a") as f: # Open the file for writing
            f.write(f"{param_string}: {i}\n")

main()
