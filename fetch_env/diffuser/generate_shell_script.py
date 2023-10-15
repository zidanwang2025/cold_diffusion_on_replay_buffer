def main():
    # Generate CDRB training script with various settings
    script_file_name = 'train_eval_command.sh'
    seed_list = [111, 222, 333]
    commands = []

    commands.append(f"python launcher.py --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 56 --short-horizon 56 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPickObstacle-v2-normalized.hdf5 --dataset FetchPickObstacle-v2 --nickname normed_16 --control-goal-size 25 &")
    commands.append(f"python launcher.py --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 36 --short-horizon 36 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPickAndPlace-v2-normalized.hdf5 --dataset FetchPickAndPlace-v2 --nickname normed_16 --control-goal-size 25 &")
    commands.append(f"python launcher.py --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 56 --short-horizon 56 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPushObstacle-v2-normalized.hdf5 --dataset FetchPushObstacle-v2 --nickname normed_16 --control-goal-size 25 &")
    commands.append(f"python launcher.py --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 36 --short-horizon 36 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPush-v2-normalized.hdf5 --dataset FetchPush-v2 --nickname normed_16 --control-goal-size 25 &")
    commands.append(f"python launcher.py --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 28 --short-horizon 28 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 10 --action-dim 4 --data-path ../expert_data/FetchReachObstacle-v2-normalized.hdf5 --dataset FetchReachObstacle-v2 --nickname normed_16 --control-goal-size 25 &")
    commands.append(f"python launcher.py --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 28 --short-horizon 28 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 10 --action-dim 4 --data-path ../expert_data/FetchReach-v2-normalized.hdf5 --dataset FetchReach-v2 --nickname normed_16 --control-goal-size 25 &")
    commands.append("wait")
    commands.append("rm success_rate.jsonl")
    for seed in seed_list:
        commands.append(f"python evaluation_pipeline.py --n-test 500 --cond-start 0 --cond-end 6 --test-validation True --validation-mode succeeded --model-type diffuser --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 56 --short-horizon 56 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPickObstacle-v2-normalized.hdf5 --dataset FetchPickObstacle-v2 --nickname normed_16 --control-goal-size 25 --set-seed {seed} &")
        commands.append(f"python evaluation_pipeline.py --n-test 500 --cond-start 0 --cond-end 6 --test-validation True --validation-mode succeeded --model-type diffuser --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 36 --short-horizon 36 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPickAndPlace-v2-normalized.hdf5 --dataset FetchPickAndPlace-v2 --nickname normed_16 --control-goal-size 25 --set-seed {seed} &")
        commands.append(f"python evaluation_pipeline.py --n-test 500 --cond-start 0 --cond-end 6 --test-validation True --validation-mode succeeded --model-type diffuser --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 56 --short-horizon 56 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPushObstacle-v2-normalized.hdf5 --dataset FetchPushObstacle-v2 --nickname normed_16 --control-goal-size 25 --set-seed {seed} &")
        commands.append(f"python evaluation_pipeline.py --n-test 500 --cond-start 0 --cond-end 6 --test-validation True --validation-mode succeeded --model-type diffuser --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 36 --short-horizon 36 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 25 --action-dim 4 --data-path ../expert_data/FetchPush-v2-normalized.hdf5 --dataset FetchPush-v2 --nickname normed_16 --control-goal-size 25 --set-seed {seed} &")
        commands.append(f"python evaluation_pipeline.py --n-test 500 --cond-start 0 --cond-end 3 --test-validation True --validation-mode succeeded --model-type diffuser --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 28 --short-horizon 28 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 10 --action-dim 4 --data-path ../expert_data/FetchReachObstacle-v2-normalized.hdf5 --dataset FetchReachObstacle-v2 --nickname normed_16 --control-goal-size 25 --set-seed {seed} &")
        commands.append(f"python evaluation_pipeline.py --n-test 500 --cond-start 0 --cond-end 3 --test-validation True --validation-mode succeeded --model-type diffuser --n-diffusion-steps 16 --join-action-state False --trim-buffer-mode kmeans --forward-sample-noise 0.0 --action-weight 1 --k-cluster 1 --horizon 28 --short-horizon 28 --n-train-steps 5000 --n-steps-per-epoch 100 --observation-dim 10 --action-dim 4 --data-path ../expert_data/FetchReach-v2-normalized.hdf5 --dataset FetchReach-v2 --nickname normed_16 --control-goal-size 25 --set-seed {seed} &")
    commands.append("wait")
    commands.append("rm sorted_success_rate.jsonl")
    commands.append("python std_from_jsonl.py")

    commands.insert(0, "export CUDA_VISIBLE_DEVICES=0,1,2,3")
    commands.insert(0, "#!/bin/bash")


    with open(script_file_name, 'w') as f:
        for command in commands:
            f.write(command + "\n")

if __name__ == "__main__":
    main()