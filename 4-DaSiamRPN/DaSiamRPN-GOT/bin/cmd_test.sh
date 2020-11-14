#!/bin/bash
START=16
END=28
seq $START 1 $END | \
    xargs -I {} echo "./models/dasiamrpn_{}.pth" | \
    xargs -I {} python -u ./bin/my_test.py    --model_path {}   
