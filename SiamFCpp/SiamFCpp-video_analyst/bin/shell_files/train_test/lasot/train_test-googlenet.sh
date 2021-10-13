#!/usr/bin/env bash
python3 ./bin/train.py --config 'models/siamfcpp/train/lasot/siamfcpp_googlenet-trn.yaml'
python3 ./bin/test.py --config 'models/siamfcpp/train/lasot/siamfcpp_googlenet-trn.yaml'
