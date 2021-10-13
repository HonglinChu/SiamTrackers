#!/bin/bash
START=11
END=39
seq $START 1 $END | \
    xargs -I {} echo "./updatenet/checkpoint/checkpoint{}.pth.tar" | \
    xargs -I {} python -u ./bin/my_test.py --update_path {} 
