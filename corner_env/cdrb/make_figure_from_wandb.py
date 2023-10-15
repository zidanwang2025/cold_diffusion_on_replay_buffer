#!/usr/bin/env python3
import wandb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

"""Given wandb project info, this scripts generates a sweep file
Input: wandb project name and group name
What happens:
The script fetches all the jobs that belong to the group,
and create pairs of (wandb-id, arguments)
Output: A sweep file that contains wandb project name and id
"""

look_up_dict = {"diffusion": "",
                "model": "",
                "k_cluster": r"$k=$",
                "n_diffusion_steps": r"$n=$",
                "dist_scheduler": "Schedule:",
                "models.ColdDiffusionRB":"CDRB",
                "models.GaussianDiffusion": "diffuser",
                'models.TemporalUnet': "UNet",
                'models.TemporalTransformer': "Transformer"}

def get_runs(entity, proj, name_list):
    api = wandb.Api()
    runs = api.runs(f'{entity}/{proj}')

    run_list = ["none"] * len(name_list)
    for run in tqdm(runs):
        if run.name in name_list:
            index = name_list.index(run.name)
            run_list[index] = run
    
    return run_list

def get_data_from_runs(runs, xkey, ykey, num_gen, configs):
    data_dict = OrderedDict()
    for run in runs:
        data_dict[run.name] = {}
        data_dict[run.name]['yvals'] = []
        for n in range(num_gen):
            ykey_ins = f"{ykey}_{n}"
            # rows = run.history(keys=[xkey, ykey_ins])
            # import pdb; pdb.set_trace()
            rows = [row for idx, row in run.history(keys=[xkey, ykey_ins]).iterrows()]
            if n == 0:
                data_dict[run.name]['xvals'] = [row[xkey] for row in rows]
            data_dict[run.name]['yvals'].append([row[ykey_ins] for row in rows])
    
        data_dict[run.name]['ymean'] = np.mean(np.array(data_dict[run.name]['yvals']), axis=0)
        data_dict[run.name]['ystd'] = np.std(np.array(data_dict[run.name]['yvals']), axis=0)

        for i, config in enumerate(configs):
            if (run.config["diffusion"] == "models.GaussianDiffusion") and (config not in ["diffusion", "model", "n_diffusion_steps"]):
                continue

            value = run.config[config]

            if config == "weight_loss_by_current_radius":
                if value == 0:
                    fig_tag = f"{fig_tag} w/o scaling"
                else:
                    fig_tag = f"{fig_tag} w/ scaling"
            else:
                if config in look_up_dict.keys():
                    config = look_up_dict[config]
                
                if value in look_up_dict.keys():
                    value = look_up_dict[value]
                
                if i == 0:
                    fig_tag = f"{config}{value}"
                else:
                    fig_tag = f"{fig_tag} {config}{value}"
        
        data_dict[run.name]['tag'] = fig_tag
        print(fig_tag)

    return data_dict

def set_fig():
    # fig setting
    import yaml
    from yaml.loader import SafeLoader
    # Open the file and load the file
    with open('../../tools/plots/matplotlibrc.yaml') as f:
        plt_std = yaml.load(f, Loader=SafeLoader)
    plt.style.use('seaborn-whitegrid')
    palette = plt.get_cmap('Set1')
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }

    plt.rcParams["figure.figsize"] = [6.00, 6.00]
    plt.rcParams["figure.autolayout"] = True
    for key, value in plt_std.items():
        plt.rcParams[key] = value

def make_fig(data_dict, filename, ykey):
    set_fig()

    # plt.cla()
    for name in data_dict.keys():
        print(data_dict[name]['tag'])
        x_vals = data_dict[name]['xvals']
        x_vals = (np.array(x_vals) + 1) * 1000

        mean = data_dict[name]['ymean']
        lows = data_dict[name]['ymean'] - (data_dict[name]['ystd'] * 2)
        highs = data_dict[name]['ymean'] + (data_dict[name]['ystd'] * 2)
        print(f"mean: {data_dict[name]['ymean'][-1]}, std: {data_dict[name]['ystd'][-1]}")

        # plt.plot(x_vals, mean, color=colors[0], label=name, marker='o', linewidth=3.0)
        # plt.fill_between(x_vals, lows, highs, color=colors[0], alpha=0.17)
        plt.plot(x_vals, mean, label=data_dict[name]['tag'], marker='o', linewidth=1.0)
        plt.fill_between(x_vals, lows, highs, alpha=0.17)

    # plt.legend(loc='upper right')
    plt.legend()
    plt.tight_layout()
    plt.xlabel(r"Iteration")
    if "success" in ykey:
        plt.ylabel(r"Success Rate")
    elif "distance" in ykey:
        plt.ylabel(r"Path Length")
    # plt.xlabel(r"Forward Diffusion Ratio: $\gamma$")
    # plt.xlim((0, 11))
    # plt.ylim((0, 1.1))

    plt.savefig(f"fig/{filename}.pdf", format='pdf')
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', type=str, default='tendon')
    parser.add_argument('--proj', type=str, default='cdrb-test')
    parser.add_argument('--xkey', type=str, default='epoch')
    parser.add_argument('--ykey', type=str, default='success_rate/controller')
    parser.add_argument('-n', '--num_gen', type=int, default=3)
    parser.add_argument('--filename', type=str, default="temp")
    parser.add_argument('--namelist', type=str, nargs='+')
    parser.add_argument('--configs', type=str, nargs='+')
    
    args = parser.parse_args()

    # get runs from wandb database
    runs = get_runs(entity=args.entity, proj=args.proj, name_list=args.namelist)

    # extract data from runs
    print("extract data")
    data_dict = get_data_from_runs(runs, args.xkey, args.ykey, args.num_gen,args.configs)
    
    print("start visualize")
    make_fig(data_dict, args.filename, args.ykey)
