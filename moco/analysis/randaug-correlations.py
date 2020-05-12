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
                acc1 = []
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
                    if 'Acc@1' in row:
                        acc1.append(row['Acc@1'])
                sup_acc = np.array(sup_acc)
                rot_acc = np.array(rot_acc)
                sup_acc_100 = np.array(sup_acc_100)
                rot_acc_100 = np.array(rot_acc_100)
                icl_loss = np.array(icl_loss)
                acc1 = np.array(acc1)

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
                if len(acc1):
                    append_data["max_acc1"] = acc1.max()
                    append_data["mean_acc1"] = acc1[-args.avg_n:].mean()
                if len(icl_loss):
                    append_data["min_icl_loss"] = icl_loss.min()
                    append_data["mean_icl_loss"] = icl_loss[-args.avg_n:].mean()
                
                summary_data.append(append_data)
    
            data = pd.DataFrame(summary_data)
            data.to_pickle(args.output_pkl)
    else:
        print("Reading pkl file")
        data = pd.read_pkl(args.input_pkl)
        import ipdb; ipdb.set_trace()
            
    print(data.corr().to_markdown())
    



if __name__=="__main__":
    print("Main")
    args = parser.parse_args()
    main(args)
