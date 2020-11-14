#!/bin/bash
START=14
END=17
seq  $START 1 $END | \
    xargs -I {} echo "./models/siamrpn_{}.pth" | \
    xargs -I {} python -u ./bin/my_test.py  --model_path  {} 
    