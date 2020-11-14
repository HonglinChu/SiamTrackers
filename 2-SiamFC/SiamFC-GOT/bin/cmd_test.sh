#!/bin/bash
START=35
END=49
seq $START 1 $END | \
    xargs -I {} echo "./models/siamfc_{}.pth" | \
    xargs -I {} python -u ./bin/my_test.py    --model_path {}   
