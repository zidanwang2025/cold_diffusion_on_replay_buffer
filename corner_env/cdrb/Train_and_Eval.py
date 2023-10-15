import wandb
from pathlib import Path
from helpers import get_dataset, get_diffuser
from run_evaluation import evaluate
from experiment_config import Args


def main():
    # Prepare the model
    dataset, val_dataset = get_dataset(Args)
    trainer, param_string = get_diffuser(Args, dataset, val_dataset)

    # Main training loop
    n_epochs = int(Args.n_train_steps // Args.n_steps_per_epoch)
    loss_log = {"train": [], "validation": []}
    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} | {Args.snapshot_root}')
        train_loss, val_loss = trainer.train(n_train_steps=Args.n_steps_per_epoch)
        loss_log["train"].append(train_loss)
        loss_log["validation"].append(val_loss)

        # save model at each epoch
        model_path = Path(Args.snapshot_root) / wandb.run.entity / wandb.run.project / wandb.run.group / wandb.run.id / f"{param_string}_epoch_{str(i).zfill(4)}.pt"
        model_path.parent.mkdir(0o775, parents=True, exist_ok=True)
        print(f'Saving model to {model_path}')
        trainer.save(model_path)

    # save the final model
    model_path = Path(Args.snapshot_root) / wandb.run.entity / wandb.run.project / wandb.run.group / wandb.run.id / f"{param_string}.pt"
    print(f'Saving model to {model_path}')
    trainer.save(model_path)

    # Evaluation
    print("#############################################")
    print("Starting evaluation...")
    diffusion = trainer.model
    evaluate(diffusion, selection_strategy=Args.condition, epoch=i)


if __name__ == '__main__':
    import argparse
    import os
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        cvd = 2
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cvd)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, default=None)
    parser.add_argument('-l', '--line-number', type=int, default=None)
    args = parser.parse_args()

    if args.sweep is not None:
        from params_proto.hyper import Sweep
        sweep = Sweep(Args).load(args.sweep)
        kwargs = list(sweep)[args.line_number]
        group_name = Path(args.sweep).stem

    Args._update(kwargs)
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="cdrb-newgoals",
        group=group_name,
        config=vars(Args),
    )

    main()
    wandb.finish()
