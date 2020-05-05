#!/bin/bash
N=$1
for M in 7 9 11; # 4 5
do
    echo "STARTING NEW M VALUE"
    echo ${M}
    /rscratch/cjrd/dul-project/deepul_proj_2/moco/main_moco.py -a resnet50 --lr 0.4 --batch-size 512 --multiprocessing-distributed --dist-url tcp://localhost:10009 --world-size 1 --rank 0 --epochs 500 --checkpoint_fp checkpoints/randaug --data /rscratch/data --rand_aug --rand_aug_m $M --rand_aug_n $N --notes "cifar10 randaug M=${M} N=${N}" --moco-t 0.2 --cos --mlp --rand_aug_orig
done
