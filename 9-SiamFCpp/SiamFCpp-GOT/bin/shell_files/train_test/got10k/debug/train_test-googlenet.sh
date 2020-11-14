#!/usr/bin/env bash
python3 ./main/train.py --config 'models/siamfcpp/train/got10k/debug/siamfcpp_googlenet-trn.yaml'
python3 ./main/test.py --config 'models/siamfcpp/train/got10k/debug/siamfcpp_googlenet-trn.yaml'
