#!/bin/bash

PORT=$(( ( RANDOM % 60000 )  + 1025 ))
for arg; do
    echo "Starting run with prefix $arg"
    python main_lincls.py -a resnet50 --lr 15.0 --batch-size 256 --dist-url "tcp://localhost:${PORT}" --world-size 1 --rank 0 --pretrained $PWD/checkpoints/${arg}*0499.tar --data /rscratch/data --epochs 150 --task rotation
    python main_lincls.py -a resnet50 --lr 15.0 --batch-size 256 --dist-url "tcp://localhost:${PORT}" --world-size 1 --rank 0 --pretrained $PWD/checkpoints/${arg}*0499.tar --data /rscratch/data --epochs 150 --task rotation --mlp
    python main_lincls.py -a resnet50 --lr 15.0 --batch-size 256 --dist-url "tcp://localhost:${PORT}" --world-size 1 --rank 0 --pretrained $PWD/checkpoints/${arg}*0499.tar --data /rscratch/data --epochs 150

done
