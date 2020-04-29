#!/bin/bash
M=$1
for N in 2 3 4;
do
    echo "STARTING NEW N VALUE"
    echo ${N}
    /rscratch/cjrd/dul-project/deepul_proj_2/moco/main_moco.py -a resnet50 --lr 0.4 --batch-size 512 --multiprocessing-distributed --dist-url tcp://localhost:10007 --world-size 1 --rank 0 --epochs 500 --checkpoint_fp checkpoints/randaug --data /rscratch/data --rand_aug --rand_aug_m $M --rand_aug_n $N --notes "cifar10 randaug M=${M} N=${N}" --moco-t 0.2 --cos --mlp
done
