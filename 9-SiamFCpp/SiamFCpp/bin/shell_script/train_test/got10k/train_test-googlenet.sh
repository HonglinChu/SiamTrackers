#!/usr/bin/env bash
python3 ./bin/train.py --config 'experiments/siamfcpp/train/got10k/siamfcpp_googlenet-trn.yaml'
python3 ./bin/test.py --config 'experiments/siamfcpp/train/got10k/siamfcpp_googlenet-trn.yaml'
