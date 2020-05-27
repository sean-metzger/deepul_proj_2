#! /usr/bin/env python
import argparse
import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

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
parser.add_argument('--entity', type=str, default='cjrd', help='wandb entity name')
parser.add_argument('--prefix', type=str, default='selfaug', help='prefix to saved plots')
parser.add_argument('--shortname', type=str, default='C-10', help='prefix to saved plots')
parser.add_argument('--dataname', type=str, default='CIFAR-10', help='prefix to saved plots')


def main(args):
    print("Starting main with args {}".format(args))
    # here we need to setup a data frame
    api = wandb.Api()
    
    sup_acc = []
    rot_acc = []
    icl_loss = []
    with open(args.idfile, 'r') as idfile:
        print("Reading ids from idfile")
        for id in idfile:
            id = id.strip()
            run = api.run("{}/{}/{}".format(args.entity, args.project, id))
            if run.summary.get("max_sup_acc"):
                sup_acc.append(run.summary["max_sup_acc"])
                rot_acc.append(run.summary["max_rot_acc"])
                icl_loss.append(run.summary["mean_icl_loss"])
            if run.summary.get("max_sup_acc_100"):
                sup_acc.append(run.summary["max_sup_acc_100"])
                rot_acc.append(run.summary["max_rot_acc_100"])
                icl_loss.append(run.summary["mean_icl_loss_100"])
            
    data = pd.DataFrame({
        "sup_acc": sup_acc,
        "rot_acc": rot_acc,
        "mean_icl": icl_loss
    })

    corr = data.corr(method='spearman')
    print(corr.to_markdown())
    
    
    # see plots here https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    # Using seaborn's style
    plt.style.use('seaborn-darkgrid')

    # With LaTex fonts
    # plt.style.use('tex')    
    width=345
    fig, ax = plt.subplots(1, 1, figsize=set_size(width))
    ax.plot('rot_acc', 'sup_acc', data=data, linestyle='none', marker='o', alpha=0.7, markersize=4, color='tab:blue')
    ax.set_facecolor((.93, .93, .93))

    # robust linear line ransac
    ransac = linear_model.RANSACRegressor()
    X = data.rot_acc
    y = data.sup_acc
    ransac.fit(X.to_numpy().reshape(-1,1), y)
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    ax.plot(line_X, line_y_ransac, color='tab:blue', linewidth=1)

    font = {'family': 'serif',
            'color':  'tab:blue',
            'weight': 'normal',
            'size': 14,
    }    
    plt.text(25, 70,
             r'$\rho = {:0.2f}$'.format(corr.sup_acc.rot_acc),
             fontdict=font,
             bbox=dict(facecolor='white', alpha=0.7),
             verticalalignment='center')
    
    ax.set_xlim((data.rot_acc.min() - 5, data.rot_acc.max() + 5))
    ax.set_ylim((15, 100))
    plt.ylabel('{} supervised classification acc.'.format(args.shortname))
    plt.xlabel('{} rotation acc.'.format(args.dataname))
    fig.savefig('figures/{}_rand_aug_rotnet.pdf'.format(args.prefix), format='pdf', bbox_inches='tight')
    
    
    # plot the icl
    fig, ax = plt.subplots(1, 1, figsize=set_size(width))
    ax.plot('mean_icl', 'sup_acc', data=data, linestyle='none', marker='o', alpha=0.7, markersize=4, color='tab:orange')
    ax.set_facecolor((.93, .93, .93))

    font["color"] = "tab:orange"
    plt.text(6.3, 60,
             r'$\rho = {:0.2f}$'.format(corr.sup_acc.mean_icl),
             fontdict=font,
             bbox=dict(facecolor='white', alpha=1.0),
             verticalalignment='center')

    
    X = data.mean_icl
    y = data.sup_acc
    ransac.fit(X.to_numpy().reshape(-1,1), y)
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    ax.plot(line_X, line_y_ransac, color='tab:orange', linewidth=1) 
    
    # ax.set_xlim((20, 75))
    ax.set_ylim((15, 100))
    ax.set_xlim((6, 11.5))
    plt.ylabel('{} supervised classification acc.'.format(args.shortname))
    plt.xlabel('{} Instance Contrastive Loss'.format(args.dataname))
    fig.savefig('figures/{}_rand_aug_icl_correlation.pdf'.format(args.prefix), format='pdf', bbox_inches='tight')
    
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
