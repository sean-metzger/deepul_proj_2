#! /usr/bin/env python
import argparse
import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt

# Read in a csv file
# For each run, compute the # min X loss
# avg X loss over last N epochs
#
# knobs: N, avg, min, whether to remove the degenerates
parser = argparse.ArgumentParser(description='SelfAugment analysis script for correlations')

#########
# WANDB #
#########
parser.add_argument('--idfile', type=str, default='', help='file containing ids of runs to analyze')
parser.add_argument('--project', type=str, default='autoself', help='wandb project name')
parser.add_argument('--avg_n', type=int, default=10, help='average number of epochs for computing average stats')
parser.add_argument('--entity', type=str, default='cjrd', help='wandb entity name')
parser.add_argument('--output_pkl', type=str, default='randaug_data.pkl', help='filename to save the analysis pkl')
parser.add_argument('--input_pkl', type=str, default=None, help='filename to load the analysis pkl (skips fetching it from wandb)')


# These are the ids of the runs that were done with 1 aug to 100 epochs
# the val-classify and val-rotation need to be translated to 100epochval-*

single_aug_ids_100 = [
    "7JnEsPsmtwahXlWOhXryJbx2fv5Lswdk",
    "mhrIn1PYVdqGS4V09cCFzIjZpBycwwdM",
    "q7qh8TSViO3bkzUexBTwF6PBqbmQ24kC",
    "JNeXFwHDR2CZYPS7Bi80YHH8MV8FQ9Io",
    "dFLZFkcDvJ8EzFU9tUlRiWJ6s16xrAbE",
    "v135FpKUr6SBi04AXDQ7yQ04YOLoL8UV",
    "9lBP8ZiwKuEhtYxzYtJGhnmNxlpdLcgP",
    "q4UR5LSC6tbUQnuTOHRcVPCPenK2HIGj",
    "XOG4lEMlEX1CY0L5shmAah32spSSBfhd",
    "xd4i77ezKFr7tDbTA7MV0F7ocvJPho3e",
    "RP2xqtpbxHoy1pG8qh8WsE1htqedIiXR",
    "FsuDbYJFF3oD4h7W7HAIDNYQL8eUFh7r",
    "4NPyFOqQ1QFZEwBbgokUYlCW5SPzaxNY",
    "i29atShIOUtnAnfovIhiKzRPVuHrPr4P",
    "Edg7rRkyTjSSBl6kEyd9BW6mz9bECLk2",
    "8bpqDNe9KIlg8KZ6L00t1XatuUIWwgpV",
]

def main(args):
    print("Starting main with args {}".format(args))
    # here we need to setup a data frame
    api = wandb.Api()
    summary_data = []
    if not args.input_pkl:
        with open(args.idfile, 'r') as idfile:
            print("Reading ids from idfile")
            for id in idfile:
                id = id.strip()
                is100 = id in single_aug_ids_100
                print(id)
                run = api.run("{}/{}/{}".format(args.entity, args.project, id))
                sup_acc = []
                rot_acc = []
                sup_acc_100 = []
                rot_acc_100 = []
                icl_loss = []
                for row in run.scan_history():
                    if 'val-classify' in row:
                        if is100:
                            sup_acc_100.append(row['val-classify'])
                        else:
                            sup_acc.append(row['val-classify'])
                    if '100epoch-val-classify' in row:
                        sup_acc_100.append(row['100epoch-val-classify'])
                    if '100epoch-val-rotation' in row:
                        rot_acc_100.append(row['100epoch-val-rotation'])
                    if 'val-rotation' in row:
                        if is100:
                            rot_acc_100.append(row['val-rotation'])
                        else:
                            rot_acc.append(row['val-rotation'])
                    if 'Loss' in row:
                        icl_loss.append(row['Loss'])
                sup_acc = np.array(sup_acc)
                rot_acc = np.array(rot_acc)
                sup_acc_100 = np.array(sup_acc_100)
                rot_acc_100 = np.array(rot_acc_100)
                icl_loss = np.array(icl_loss)

                append_data = { "id": id }

                if len(sup_acc):
                    append_data["max_sup_acc"] = sup_acc.max()
                    append_data["mean_sup_acc"] = sup_acc[-args.avg_n:].mean()
                if len(rot_acc):
                    append_data["max_rot_acc"] = rot_acc.max()
                    append_data["mean_rot_acc"] = rot_acc[-args.avg_n:].mean()
                if len(sup_acc_100):
                    append_data["max_sup_acc_100"] = sup_acc_100.max()
                    append_data["mean_sup_acc_100"] = sup_acc_100[-args.avg_n:].mean()
                if len(rot_acc_100):
                    append_data["max_rot_acc_100"] = rot_acc_100.max()
                    append_data["mean_rot_acc_100"] = rot_acc_100[-args.avg_n:].mean()
                if len(icl_loss):
                    append_data["min_icl_loss"] = icl_loss.min()
                    append_data["mean_icl_loss"] = icl_loss[-args.avg_n:].mean()
                
                summary_data.append(append_data)
    
            data = pd.DataFrame(summary_data)
            data.to_pickle(args.output_pkl)
    else:
        print("Reading pkl file")
        data = pd.read_pickle(args.input_pkl)
        
    print(data.corr().to_markdown())

    # see plots here https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    # Using seaborn's style
    plt.style.use('seaborn-darkgrid')

    # With LaTex fonts
    # plt.style.use('tex')    
    width=345
    fig, ax = plt.subplots(1, 1, figsize=set_size(width))
    ax.plot('max_rot_acc', 'max_sup_acc', data=data, linestyle='none', marker='o', alpha=0.5, markersize=4, color='green')
    ax.plot('max_rot_acc_100', 'max_sup_acc_100', data=data, linestyle='none', marker='o', alpha=0.5, markersize=4, color='green')
    ax.set_facecolor((.93, .93, .93))
    
    ax.set_xlim((20, 75))
    ax.set_ylim((5, 95))
    plt.ylabel('C10 supervised classification acc.')
    plt.xlabel('CIFAR 10 rotation acc.')
    fig.savefig('rand_aug_rotnet.pdf', format='pdf', bbox_inches='tight')
    
    
    # plot the icl
    fig, ax = plt.subplots(1, 1, figsize=set_size(width))
    ax.plot('mean_icl_loss', 'max_sup_acc', data=data, linestyle='none', marker='o', alpha=0.5, markersize=4, color='red')
    ax.plot('mean_icl_loss', 'max_sup_acc_100', data=data, linestyle='none', marker='o', alpha=0.5, markersize=4, color='red')
    ax.set_facecolor((.93, .93, .93))
    
    # ax.set_xlim((20, 75))
    ax.set_ylim((5, 95))
    plt.ylabel('C10 supervised classification acc.')
    plt.xlabel('CIFAR 10 Instance Contrastive Loss')
    fig.savefig('rand_aug_icl_correlation.pdf', format='pdf', bbox_inches='tight')
    
    # TODO add a robust linear regression fit

    
    # TODO separate out the various correlation parameters
    

    


def set_size(width, fraction=1):
    """ Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim



if __name__=="__main__":
    print("Main")
    args = parser.parse_args()
    main(args)
