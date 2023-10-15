#!/usr/bin/env python3
import wandb
import json
from copy import deepcopy
from pathlib import Path

"""Given wandb project info, this scripts generates a sweep file
Input: wandb project name and group name
What happens:
The script fetches all the jobs that belong to the group,
and create pairs of (wandb-id, arguments)
Output: A sweep file that contains wandb project name and id
"""

def main(entity, proj, group):
    api = wandb.Api()
    name = f'{entity}/{proj}'

    query = {
        "$and": [
            {"group": {"$eq": group}},
            {"state": {"$eq": "finished"}}  # Filter out crashed / failed runs
        ]
    }

    runs = api.runs(name, query)

    sweep_instances = []
    run_ids = []
    for i, run in enumerate(runs):
        sweep_instances.append(deepcopy(run.config))
        run_ids.append(run.id)
    print(f'Number of runs: {i + 1}')

    directory = Path('wandb-runs') / entity / proj / group
    directory.mkdir(parents=True, exist_ok=True)

    # Save the sweep_instances as a jsonl file
    with open(directory / 'sweep.jsonl', 'w') as f:
        for d in sweep_instances:
            json.dump(d, f)
            f.write('\n')

    # Save the meta data
    with open(directory / 'meta.json', 'w') as f:
        json.dump({
            'entity': entity,
            'proj': proj,
            'group': group,
            'run_ids': run_ids
        }, f)

    print('Files are saved under ', directory)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', type=str, default='takuma-yoneda')
    parser.add_argument('--proj', type=str, default='cdrb-newgoals')
    parser.add_argument('--group', type=str, default='takuma-20230531-longlong-horizon')
    args = parser.parse_args()

    main(entity=args.entity, proj=args.proj, group=args.group)
