#!/bin/bash

PORT=$(( ( RANDOM % 60000 )  + 1025 ))
for arg; do
    echo "Starting run with prefix $arg"

    python main_lincls.py -a resnet50 --lr 15.0 --batch-size 256 --dist-url "tcp://localhost:${PORT}" --world-size 1 --rank 0 --pretrained $PWD/checkpoints/${arg}*0499.tar --data /rscratch/data --epochs 50 --task rotation --schedule 10 20 --dataid svhn

    echo "Doing supervised eval now"
    python main_lincls.py -a resnet50 --lr 15.0 --batch-size 256 --dist-url "tcp://localhost:${PORT}" --world-size 1 --rank 0 --pretrained $PWD/checkpoints/${arg}*0499.tar --data /rscratch/data --epochs 50 --schedule 10 20 --dataid svhn
    
    # Eval at 100
    # python main_lincls.py -a resnet50 --lr 15.0 --batch-size 256 --dist-url "tcp://localhost:${PORT}" --world-size 1 --rank 0 --pretrained $PWD/checkpoints/${arg}*0100.tar --data /rscratch/data --epochs 50 --task rotation --loss-prefix '100epoch'
    # python main_lincls.py -a resnet50 --lr 15.0 --batch-size 256 --dist-url "tcp://localhost:${PORT}" --world-size 1 --rank 0 --pretrained $PWD/checkpoints/${arg}*0100.tar --data /rscratch/data --epochs 50 --loss-prefix '100epoch'

done
