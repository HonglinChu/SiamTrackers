#!/bin/bash
START=26
END=50
seq $START 1 $END | \
    xargs -I {} echo "./models/snapshot/checkpoint_e{}.pth" |  \
    xargs -I {} python  -u ./bin/my_test.py --snapshot {}  --dataset VOT2018 
