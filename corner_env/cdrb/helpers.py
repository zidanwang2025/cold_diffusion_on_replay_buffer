import datasets
from temporal_unet import TemporalUnet
from temporal_transformer import TemporalTransformer

from diffusion_models import GaussianDiffusion as ColdDiffusionRB
from diffusion_gymnasium_original.diffusion_models import GaussianDiffusion
import main_trainer

# HACK
import sys; sys.path.append("..")
from gymnasium_registration import initialize_env


def get_dataset(Args):
    initialize_env()
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
            dataset_size=Args.dataset_size
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
            dataset_size=Args.dataset_size
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
        dataset_size=Args.dataset_size
    )

    return dataset, val_dataset

    
def get_diffuser(Args, dataset, val_dataset):
    
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

    if Args.diffusion == "models.ColdDiffusionRB":
        diffusion = ColdDiffusionRB(
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
            max_dist=Args.max_dist,
            dist_scheduler=Args.dist_scheduler,
            replay_dataset=Args.dataset,
        )

        param_string = "join_" + str(Args.join_action_state) \
            + "_trim_" + Args.trim_buffer_mode \
            + "_noise_" + str(Args.forward_sample_noise) \
            + "_weight_" + str(Args.action_weight) \
            + "_horizon_" + str(Args.horizon) \
            + "_epoch_" + str(int(Args.n_train_steps/Args.n_steps_per_epoch))

        if Args.trim_buffer_mode == "kmeans":
            param_string += "_k_" + str(Args.k_cluster)
        elif Args.trim_buffer_mode == "euclidean":
            param_string += "_d_" + str(Args.d_distance)

    elif Args.diffusion == "models.GaussianDiffusion":
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

        param_string = "noise_" + str(Args.forward_sample_noise) \
            + "_weight_" + str(Args.action_weight) \
            + "_horizon_" + str(Args.horizon) \
            + "_epoch_" + str(int(Args.n_train_steps/Args.n_steps_per_epoch))
    else:
        raise NotImplementedError("TODO")
    
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

    return trainer, param_string


def shift_range(state, old_min, old_max, new_min, new_max):
    """Apply normalization to a state."""
    old_range = old_max - old_min
    new_range = new_max - new_min
    new_state = (((state - old_min) * new_range) / old_range) + new_min
    return new_state


def get_maze_range(Args):
    if Args.dataset == "gymnasium-corner-env-archive":
        normalize_constants = [-6.0, 6.0]
    else:
        normalize_constants = [-6.5, 6.5]
    return normalize_constants