#!/bin/bash
START=11
END=16
seq $START 1 $END | \
    xargs -I {} echo "models/snapshots/siamfcpp_alexnet-got/epoch-{}.pkl" | \
    xargs -I {} python -u ./bin/my_test.py    --model_path {}   
