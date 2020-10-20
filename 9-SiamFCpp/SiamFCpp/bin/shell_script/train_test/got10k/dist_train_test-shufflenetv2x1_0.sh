#!/usr/bin/env bash
python3 -W ignore ./main/dist_train.py --config 'experiments/siamfcpp/train/got10k/siamfcpp_shufflenetv2x1_0-dist_trn.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/train/got10k/siamfcpp_shufflenetv2x1_0-dist_trn.yaml'
