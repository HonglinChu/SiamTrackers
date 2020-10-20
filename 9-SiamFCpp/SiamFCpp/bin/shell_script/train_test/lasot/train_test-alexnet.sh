#!/usr/bin/env bash
python3 ./bin/train.py --config 'experiments/siamfcpp/train/lasot/siamfcpp_alexnet-trn.yaml'
python3 ./bin/test.py --config 'experiments/siamfcpp/train/lasot/siamfcpp_alexnet-trn.yaml'
