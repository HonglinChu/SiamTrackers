# SiamCAR-V2

We rename SiamCAR-V2 as SiamFC++

[Official Code](https://github.com/MegviiDetection/video_analyst)

[哔哩哔哩](https://www.bilibili.com/video/BV1NT4y1J7R5?from=search&seid=2666686591863260886)

# Train

```
We only use GOT-10k to train siamcar network

sh ./bin/cmd_dist_train.sh 

```

# Test
```
sh ./bin/cmd_test.sh 
```

```
checkpoint_e20.pth VOT2018  lr-0.35  pk-_0.2  win-_0.45
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------
|  siamcar   |  0.570   |   0.318    |    68.0     | 0.314 |
------------------------------------------------------------

------------------------------------------------------------
checkpoint_e41.pth VOT2018  lr-0.35  pk-_0.2  win-_0.45
------------------------------------------------------------
|Tracker Name| Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------
|  siamcar   |  0.566   |   0.328    |    70.0     | 0.315 |
------------------------------------------------------------

hp_search  load checkpoint_e20 
------------------------------------------------------------------------------------------------
|                  Tracker Name                  | Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------------------------------------------
| checkpoint_e20_r255_pk-0.250_wi-0.460_lr-0.340 |  0.568   |   0.300    |    64.0     | 0.353 |
| checkpoint_e20_r255_pk-0.240_wi-0.440_lr-0.360 |  0.568   |   0.300    |    64.0     | 0.346 |
| checkpoint_e20_r255_pk-0.240_wi-0.450_lr-0.360 |  0.570   |   0.290    |    62.0     | 0.343 |
| checkpoint_e20_r255_pk-0.240_wi-0.460_lr-0.330 |  0.564   |   0.300    |    64.0     | 0.343 |
| checkpoint_e20_r255_pk-0.240_wi-0.460_lr-0.340 |  0.566   |   0.309    |    66.0     | 0.341 |
| checkpoint_e20_r255_pk-0.240_wi-0.450_lr-0.330 |  0.564   |   0.314    |    67.0     | 0.341 |
| checkpoint_e20_r255_pk-0.240_wi-0.450_lr-0.340 |  0.565   |   0.304    |    65.0     | 0.339 |
| checkpoint_e20_r255_pk-0.240_wi-0.460_lr-0.360 |  0.570   |   0.304    |    65.0     | 0.338 |
| checkpoint_e20_r255_pk-0.200_wi-0.460_lr-0.330 |  0.568   |   0.300    |    64.0     | 0.337 |
| checkpoint_e20_r255_pk-0.220_wi-0.450_lr-0.330 |  0.568   |   0.304    |    65.0     | 0.335 |
| checkpoint_e20_r255_pk-0.180_wi-0.460_lr-0.360 |  0.567   |   0.309    |    66.0     | 0.335 |
| checkpoint_e20_r255_pk-0.240_wi-0.440_lr-0.340 |  0.565   |   0.314    |    67.0     | 0.334 |
| checkpoint_e20_r255_pk-0.250_wi-0.450_lr-0.340 |  0.568   |   0.318    |    68.0     | 0.333 |
| checkpoint_e20_r255_pk-0.240_wi-0.440_lr-0.330 |  0.568   |   0.332    |    71.0     | 0.333 |
| checkpoint_e20_r255_pk-0.180_wi-0.450_lr-0.340 |  0.567   |   0.295    |    63.0     | 0.331 |
| checkpoint_e20_r255_pk-0.200_wi-0.440_lr-0.350 |  0.569   |   0.300    |    64.0     | 0.331 |
| checkpoint_e20_r255_pk-0.160_wi-0.460_lr-0.330 |  0.566   |   0.300    |    64.0     | 0.330 |
| checkpoint_e20_r255_pk-0.220_wi-0.460_lr-0.330 |  0.567   |   0.314    |    67.0     | 0.330 |
| checkpoint_e20_r255_pk-0.250_wi-0.440_lr-0.330 |  0.568   |   0.328    |    70.0     | 0.328 |
| checkpoint_e20_r255_pk-0.180_wi-0.460_lr-0.340 |  0.568   |   0.300    |    64.0     | 0.328 |
------------------------------------------------------------------------------------------------
```
# Reference
```
https://github.com/ohhhyeahhh/SiamCAR

https://github.com/MegviiDetection/video_analyst

[1] Guo D ,  Wang J ,  Cui Y , et al. SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking[C]//CVPR,2020.

[2] Xu Y, Wang Z, Li Z, et al. SiamFC++: Towards Robust and Accurate Visual Tracking with Target Estimation Guidelines[C]//AAAI,2020.

```
