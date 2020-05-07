#!/bin/bash

#SBATCH -J sa-cjrd           # Job name
#SBATCH -o /scratch/07399/cjrd/logs/sa-cjrd-out.%j       # Name of stdout output file
#SBATCH -e /scratch/07399/cjrd/logs/sa-cjrd-err.%j       # Name of stderr error file
#SBATCH -p v100            # Queue (partition) name v100, development
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 40:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=cjrd@berkeley.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

# export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

module load pytorch-py3/1.3.1
module list
pwd
date

set -x
PORT=$(( ( RANDOM % 60000 )  + 1025 ))

# CIFAR
# srun python -u main_moco.py -a resnet50 --lr 0.4 --batch-size 512 --multiprocessing-distributed --dist-url tcp://localhost:10007 --world-size 1 --rank 0 --epochs 100 --checkpoint_fp $SCRATCH/checkpoints/ --data $SCRATCH --notes "test longhorn tacc 4 gpus cifar for 100 epochs" --moco-t 0.2 --cos --mlp --aug-plus --upload_checkpoints

# IMAGENet RandAug
srun python -u main_moco.py -a resnet50 --lr 0.015 --batch-size 128 --multiprocessing-distributed --dist-url tcp://localhost:${PORT} --world-size 1 --rank 0 --epochs 100 --checkpoint_fp $SCRATCH/checkpoints/ --data $SCRATCH/imagenet12 --notes "longhorn imagenet n=2; m=11; 4 gpus; 100 epoch" --moco-t 0.2 --cos --mlp --upload_checkpoints --checkpoint-interval 20 --dataid imagenet --rand_aug --rand_aug_n 2 --rand_aug_m 11

# IMAGENet MocoV2
#srun python -u main_moco.py -a resnet50 --lr 0.015 --batch-size 128 --multiprocessing-distributed --dist-url tcp://localhost:${PORT} --world-size 1 --rank 0 --epochs 100 --checkpoint_fp $SCRATCH/checkpoints/ --data $SCRATCH/imagenet12 --notes "longhorn imagenet n=2; m=13; 4 gpus; 100 epoch" --moco-t 0.2 --cos --mlp --upload_checkpoints --checkpoint-interval 20 --dataid imagenet --aug-plus
