#!/usr/bin/env bash
python3 ./main/train.py --config 'experiments/siamfcpp/train/got10k/siamfcpp_shufflenetv2x0_5-trn.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/train/got10k/siamfcpp_shufflenetv2x0_5-trn.yaml'
