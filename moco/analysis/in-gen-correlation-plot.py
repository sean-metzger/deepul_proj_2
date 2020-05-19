#! /usr/bin/env python
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from imagenetresults import iresults
# Read in a csv file
# For each run, compute the # min X loss
# avg X loss over last N epochs
#
# knobs: N, avg, min, whether to remove the degenerates
parser = argparse.ArgumentParser(description='SelfAugment analysis script for correlations')

#########
# WANDB #
#########
parser.add_argument('--prefix', type=str, default='inet', help='prefix to saved plots')
parser.add_argument('--shortname', type=str, default='ImageNet', help='prefix to saved plots')
parser.add_argument('--dataname', type=str, default='ImageNet', help='prefix to saved plots')


def main(args):
    print("Starting main with args {}".format(args))
    # here we need to setup a data frame
    
    sup_acc = []
    rot_acc = []

    for entry in iresults.values():
        if entry["val-classify"] < 0:
            continue
        sup_acc.append(entry["val-classify"])
        rot_acc.append(entry["val-rotate"])
                
    data = pd.DataFrame({
        "sup_acc": sup_acc,
        "rot_acc": rot_acc,
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
    
    
    plt.text(68.2, 64,
             r'$\rho = {:0.2f}$'.format(corr.sup_acc.rot_acc),
             fontdict=font,
             bbox=dict(facecolor='white', alpha=0.7),
             verticalalignment='center')
    
    ax.set_xlim((68, 75))
    ax.set_ylim((55, 70))
    plt.ylabel('{} supervised classification acc.'.format(args.shortname))
    plt.xlabel('{} rotation acc.'.format(args.dataname))
    fig.savefig('figures/{}_rand_aug_rotnet.pdf'.format(args.prefix), format='pdf', bbox_inches='tight')
    
    
    # # plot the icl
    # fig, ax = plt.subplots(1, 1, figsize=set_size(width))
    # ax.plot('mean_icl', 'sup_acc', data=data, linestyle='none', marker='o', alpha=0.7, markersize=4, color='tab:orange')
    # ax.set_facecolor((.93, .93, .93))

    # font["color"] = "tab:orange"
    # plt.text(6.71, 65,
    #          r'$\rho = {:0.2f}$'.format(corr.sup_acc.mean_icl),
    #          fontdict=font,
    #          bbox=dict(facecolor='white', alpha=1.0),
    #          verticalalignment='center')

    
    # X = data.mean_icl
    # y = data.sup_acc
    # ransac.fit(X.to_numpy().reshape(-1,1), y)
    # line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    # line_y_ransac = ransac.predict(line_X)
    # ax.plot(line_X, line_y_ransac, color='tab:orange', linewidth=1) 
    
    # # ax.set_xlim((20, 75))
    # ax.set_ylim((60, 66))
    # ax.set_xlim((6.7,6.8))
    # plt.ylabel('{} supervised classification acc.'.format(args.shortname))
    # plt.xlabel('{} Instance Contrastive Loss'.format(args.dataname))
    # fig.savefig('figures/{}_rand_aug_icl_correlation.pdf'.format(args.prefix), format='pdf', bbox_inches='tight')
    

    


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
