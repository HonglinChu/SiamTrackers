#!/usr/bin/env bash
python3 -W ignore ./main/dist_train.py --config 'models/siamfcpp/train/got10k/siamfcpp_shufflenetv2x0_5-dist_trn.yaml'
python3 ./main/test.py --config 'models/siamfcpp/train/got10k/siamfcpp_shufflenetv2x0_5-dist_trn.yaml'
