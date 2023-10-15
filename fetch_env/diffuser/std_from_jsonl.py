import json
import statistics
import numpy as np
import wandb

def main():
    with open('success_rate.jsonl', 'r') as file:
        lines = file.readlines()

    sorted_lines = []
    for line in lines:
        sorted_line = json.dumps(json.loads(line), sort_keys=False)
        sorted_lines.append(sorted_line)

    sorted_lines.sort()

    with open('sorted_success_rate.jsonl', 'w') as file:
        for line in sorted_lines:
            file.write(line + '\n')

    dict_list = []
    with open('sorted_success_rate.jsonl', 'r') as file:
        for line in file:
            json_object = json.loads(line)
            dict_list.append(json_object)

    grouped_data = {}
    for d in dict_list:
        key = (d['model'], d['env'], d['allowed actions'], d['nickname'], d['epoch'])
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(d)

    for key, group in grouped_data.items():
        weighted_success = [d['weighted success rate'] for d in group]
        weighted_success_mean = statistics.mean(weighted_success)
        weighted_success_stdev = statistics.stdev(weighted_success)
        unweighted_success = [d['unweighted success rate'] for d in group]
        unweighted_success_mean = statistics.mean(unweighted_success)
        unweighted_success_stdev = statistics.stdev(unweighted_success)
        weight = [d['weight'] for d in group]
        weight_mean = statistics.mean(weight)

        wandb.login()
        wandb.init(
            project="Fetch",
            name=key[0],
            config={
                "enviroment": key[1],
                "allowed actions": key[2],
                "nickname": key[3],
                "epoch": key[4],
            }
        )
        wandb.log({
            "weighted success rate mean": weighted_success_mean,
            "weighted success rate stdev": weighted_success_stdev,
            "unweighted success rate mean": unweighted_success_mean,
            "unweighted success rate stdev": unweighted_success_stdev,
            "weight mean": weight_mean
            })
        wandb.finish()

if __name__ == '__main__':
    main()