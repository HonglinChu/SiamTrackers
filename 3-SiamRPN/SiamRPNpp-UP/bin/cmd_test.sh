
#!/bin/bash
START=20
END=49
seq $START 1 $END | \
    xargs -I {} echo "./models/siamrpnpp_alexnet/snapshot/checkpoint_e{}.pth" | \
    xargs -I {} python -u ./bin/test.py --snapshot {} --config ./models/siamrpnpp_alexnet/config.yaml \
    --dataset VOT2018 2>&1 | tee ./models/siamrpnpp_alexnet/logs/test_dataset.log
