#!/bin/bash

#SBATCH -J sa-cjrd           # Job name
#SBATCH -o /scratch1/07399/cjrd/logs/sa-cjrd-out.%j       # Name of stdout output file
#SBATCH -e /scratch1/07399/cjrd/logs/sa-cjrd-err.%j       # Name of stderr error file
#SBATCH -p rtx            # Queue (partition) name v100, development
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 08:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=cjrd@berkeley.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

# export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)


module list
pwd
date

set -x
PORT=$(( ( RANDOM % 60000 )  + 1025 ))
PORT2=$(( ( RANDOM % 60000 )  + 1025 ))

KVAL=3


MVAL=4
rid1=7JIs6
python3 -u main_moco.py -a resnet50 --lr 0.4 --batch-size 512 --multiprocessing-distributed --dist-url tcp://localhost:${PORT} --world-size 1 --rank 0 --epochs 500 --checkpoint_fp $SCRATCH/checkpoints/ --data $SCRATCH --notes "K=${KVAL} M=${MVAL}" --moco-t 0.2 --cos --mlp --upload_checkpoints --rand_aug --rand_aug_n 2 --rand_aug_m ${MVAL} --rand_aug_top_k ${KVAL} --resume $SCRATCH/checkpoints/${rid1}_*0300.tar &

MVAL=7
rid2=H81IV
python3 -u main_moco.py -a resnet50 --lr 0.4 --batch-size 512 --multiprocessing-distributed --dist-url tcp://localhost:${PORT2} --world-size 1 --rank 0 --epochs 500 --checkpoint_fp $SCRATCH/checkpoints/ --data $SCRATCH --notes "K=${KVAL} M=${MVAL}" --moco-t 0.2 --cos --mlp --upload_checkpoints --rand_aug --rand_aug_n 2 --rand_aug_m ${MVAL} --rand_aug_top_k ${KVAL} --resume $SCRATCH/checkpoints/${rid2}_*0300.tar &

wait
