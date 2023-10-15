import pickle
import sys
import argparse
from pathlib import Path

import numpy as np

import datasets
import main_trainer
from diffusion_models import GaussianDiffusion
from experiment_config import Args
from temporal_unet import TemporalUnet
import torch
import wandb

sys.path.append("..")

def main(**kwargs):
    Args._update(kwargs)

    Args.short_horizon = Args.horizon * Args.shorten_ratio

    Args.horizon = int(Args.horizon)
    Args.short_horizon = int(Args.short_horizon)

    print("include goal:", Args.include_goal_in_state)
    print("pin goal:", Args.pin_goal)
    print("train horizon:", Args.short_horizon)

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
                data_path=Args.data_path,
                observation_dim=Args.observation_dim,
                action_dim=Args.action_dim,
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
                data_path=Args.data_path,
                observation_dim=Args.observation_dim,
                action_dim=Args.action_dim,
            )

        val_dataset = datasets.GoalDataset(
            env=Args.dataset,
            horizon=Args.horizon,
            normalizer=Args.normalizer,
            preprocess_fns=Args.preprocess_fns,
            use_padding=Args.use_padding,
            max_n_episodes=Args.max_n_episodes,
            max_path_length=Args.max_path_length,
            validation=True,
            data_path=Args.data_path,
            observation_dim=Args.observation_dim,
            action_dim=Args.action_dim,
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
            trim_buffer_mode=Args.trim_buffer_mode,
            data_path=Args.data_path,
        )
        diffusion.to(Args.device)


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
            device=Args.device,
        )

        return trainer

    dataset, val_dataset = get_dataset(Args)

    trainer = get_diffuser(Args, dataset, val_dataset)

    n_epochs = int(Args.n_train_steps // Args.n_steps_per_epoch)

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

    loss_log = {"train": [], "validation":[]}
    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} | {Args.snapshot_root}')
        train_loss, val_loss = trainer.train(n_train_steps=Args.n_steps_per_epoch)
        loss_log["train"].append(train_loss)
        loss_log["validation"].append(val_loss)

        model_path = Path(Args.snapshot_root) / f"{param_string}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        print(f'Saving model to {model_path}')
        trainer.save(model_path)

if __name__ == '__main__':
    import os
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        cvd = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cvd)

    kwargs = {}
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project=Args.dataset,
        name=Args.model_type,
        config=vars(Args),
    )
    main(**kwargs)
    wandb.finish()