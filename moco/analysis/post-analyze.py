#! /usr/bin/env python
import sys
import argparse
import pandas as pd
import wandb
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser(description='SelfAugment post-analysis script for wandb')

########
# ARGS #
########
parser.add_argument('--project', type=str, default='autoself', help='wandb project name')
parser.add_argument('--avg_n', type=int, default=10, help='average number of epochs for computing average stats')
parser.add_argument('--entity', type=str, default='cjrd', help='wandb entity name')
parser.add_argument('--is100', action='store_true', help='run evaluations took place at 100 epochs but did not use the 100epoch prefix.')

########
# MAIN #
########
def main(args):
    print("Starting post-analyze with args {}".format(args))
    api = wandb.Api()
    now = datetime.now()
    for wid in sys.stdin:
        wid = wid.strip()
        print(wid)
        run = api.run("{}/{}/{}".format(args.entity, args.project, wid))
        
        sup_acc = []
        rot_acc = []
        sup_acc_100 = []
        rot_acc_100 = []
        icl_loss = []
        for row in run.scan_history():
            if 'val-classify' in row:
                if args.is100:
                    sup_acc_100.append(row['val-classify'])
                else:
                    sup_acc.append(row['val-classify'])
            if '100epoch-val-classify' in row:
                sup_acc_100.append(row['100epoch-val-classify'])
            if '100epoch-val-rotation' in row:
                rot_acc_100.append(row['100epoch-val-rotation'])
            if 'val-rotation' in row:
                if args.is100:
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

        update_data = {
            "post_analysis": now
        }

        if len(sup_acc):
            update_data["max_sup_acc"] = sup_acc.max()
            update_data["mean_sup_acc"] = sup_acc[-args.avg_n:].mean()
        if len(rot_acc):
            update_data["max_rot_acc"] = rot_acc.max()
            update_data["mean_rot_acc"] = rot_acc[-args.avg_n:].mean()
        if len(sup_acc_100):
            update_data["max_sup_acc_100"] = sup_acc_100.max()
            update_data["mean_sup_acc_100"] = sup_acc_100[-args.avg_n:].mean()
        if len(rot_acc_100):
            update_data["max_rot_acc_100"] = rot_acc_100.max()
            update_data["mean_rot_acc_100"] = rot_acc_100[-args.avg_n:].mean()
        if len(icl_loss):
            update_data["min_icl_loss"] = icl_loss.min()
            update_data["mean_icl_loss"] = icl_loss[-args.avg_n:].mean()
        for key, val in update_data.items():
            run.summary[key] = val
        run.update()
        print("updated")
        print("")
        
        
            
if __name__=="__main__":
    args = parser.parse_args()
    main(args)
