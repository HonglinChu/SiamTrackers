#!/usr/bin/env bash
python3 ./main/train.py --config 'experiments/siamfcpp/train/fulldata/siamfcpp_googlenet-trn-fulldata.yaml'
python3 ./main/test.py --config 'experiments/siamfcpp/train/fulldata/siamfcpp_googlenet-trn-fulldata.yaml'
