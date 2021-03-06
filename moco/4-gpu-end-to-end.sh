#!/bin/bash

set -e

if [ $# -ne 3 ]
then
    echo "usage 4-gpu-end-to-end.sh CHECKPOINT_DIR DATA_DIR NOTES"
    exit 3
fi

CHECKPOINT_DIR=$1
DATA_DIR=$2
NOTES=$3

# moco v2
python main_moco.py -a resnet50 --lr 0.015  --batch-size 128 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --epochs 400 --checkpoint_fp ${CHECKPOINT_DIR} --data ${DATA_DIR} --notes "${NOTES}" --aug-plus --cos --moco-t 0.2
