#!/bin/bash
START=35
END=50
seq $START 1 $END | \
    xargs -I {} echo "snapshot/checkpoint_e{}.pth" | \
    xargs -I {} python -u ../../bin/test.py --snapshot {} --config config.yaml \
    --dataset VOT2018 2>&1 | tee logs/test_dataset.log
