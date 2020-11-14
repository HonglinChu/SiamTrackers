#!/usr/bin/env bash
python3 -W ignore ./bin/dist_train.py --config 'models/siamfcpp/train/got10k/siamfcpp_alexnet-dist_trn.yaml'
python3 ./bin/test.py --config 'models/siamfcpp/train/got10k/siamfcpp_alexnet-dist_trn.yaml'
