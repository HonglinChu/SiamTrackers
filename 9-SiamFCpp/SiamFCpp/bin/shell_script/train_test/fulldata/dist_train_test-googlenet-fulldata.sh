#!/usr/bin/env bash
python3 ./bin/dist_train.py --config 'experiments/siamfcpp/train/fulldata/siamfcpp_googlenet-dist_trn-fulldata.yaml'
python3 ./bin/test.py --config 'experiments/siamfcpp/train/fulldata/siamfcpp_googlenet-dist_trn-fulldata.yaml'
