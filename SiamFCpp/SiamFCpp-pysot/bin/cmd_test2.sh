#!/bin/bash
START=31
END=49
seq $START 1 $END | \
    xargs -I {} echo "./models/snapshot/checkpoint_e{}.pth" |  \
    xargs -I {} python  -u ./bin/my_test2.py --snapshot {}  --dataset VOT2018 
