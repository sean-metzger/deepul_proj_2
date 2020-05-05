#! /usr/bin/env python

import argparse
import pandas as pd
import wandb
import numpy as np

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
parser.add_argument('--avg_n', type=int, default=50, help='average number of epochs for computing average stats')
parser.add_argument('--entity', type=str, default='cjrd', help='wandb entity name')

def main(args):
    # here we need to setup a data frame
    api = wandb.Api()
    summary_data = []
    with open(args.idfile, 'r') as idfile:
        for id in idfile:
            id = id.strip()
            print(id)
            run = api.run("{}/{}/{}".format(args.entity, args.project, id))
            sup_acc = []
            rot_acc = []
            icl_loss = []
            acc1 = []
            for row in run.scan_history():
                if 'val-classify' in row:
                    sup_acc.append(row['val-classify'])
                if 'val-rotation' in row:
                    rot_acc.append(row['val-rotation'])
                if 'Loss' in row:
                    icl_loss.append(row['Loss'])
                if 'Acc@1' in row:
                    acc1.append(row['Acc@1'])
            sup_acc = np.array(sup_acc)
            rot_acc = np.array(rot_acc)
            icl_loss = np.array(icl_loss)
            acc1 = np.array(acc1)
            
            summary_data.append({
                "id": id,
                "max_sup": sup_acc.max(),
                "avg_sup": sup_acc[-args.avg_n:].mean(),
                "max_rot": rot_acc.max(),
                "avg_rot": rot_acc[-args.avg_n:].mean(),
                "max_acc1": acc1[-args.avg_n:].max(),
                "avg_acc1": acc1[-args.avg_n:].mean(),
                "min_icl": icl_loss.min(),
                "avg_icl": icl_loss[-args.avg_n:].mean(),
            })
    data = pd.DataFrame(summary_data)
    print(data.corr().to_markdown())
    



if __name__=="__main__":
    args = parser.parse_args()
    main(args)
