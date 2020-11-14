#!/usr/bin/env bash

cd pysot/utils/
python3 setup.py clean
python3 setup.py build_ext --inplace
cd ../../
