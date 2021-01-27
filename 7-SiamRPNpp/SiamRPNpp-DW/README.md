# SiamRPNpp

## Description

[Official Code](https://github.com/STVIR/pysot)(Recommend)

## Compile
```
cd  /home/xxx/xxx/SiamTrackers-master/7-SiamRPNpp/SiamRPNpp-DW 
python setup.py build_ext —-inplace
```
## Train
```
cd  /home/xxx/xxx/SiamTrackers-master/7-SiamRPNpp/SiamRPNpp-DW 
python ./bin/my_train.py
```
## Test
```
cd  /home/xxx/xxx/SiamTrackers-master/7-SiamRPNpp/SiamRPNpp-DW 
python ./bin/my_test.py
```
## Eval
```
cd  /home/xxx/xxx/SiamTrackers-master/7-SiamRPNpp/SiamRPNpp-DW 
python ./bin/my_eval.py
```
## Model 

[BaiduYun](https://pan.baidu.com/s/1ZE3UDZwCTH0TyLkCa1PnYw) password:1bbk (siamrpnpp-dw with VID DET COCO YTB)

[BaiduYun](https://pan.baidu.com/s/1OPNJcBYASMAXPKm-DaoWSQ) password: qyep (siamrpnpp-dw with GOT10k)

## Dataset
How to crop GOT-10k dataset
[BaiDuYun](https://pan.baidu.com/s/17daZiLQj5moUm3F0R2p_fQ)  password:vmhc

- **ILSVRC2015 VID** [BaiDuYun](https://pan.baidu.com/s/1CXWgpAG4CYpk-WnaUY5mAQ) password: uqzj 

- **ILSVRC2015 DET** [BaiDuYun](https://pan.baidu.com/s/1t2IgiYGRu-sdfOYwfeemaQ) password: 6fu7 

- **YTB-Crop511** [BaiduYun](https://pan.baidu.com/s/112zLS_02-Z2ouKGbnPlTjw) password: ebq1 （used in siamrpn++ and siammask）

- **COCO** [BaiduYun](https://pan.baidu.com/s/17AMGS2ezLVd8wFI2NbJQ3w) password: ggya 


## Experiment

My results of siamrpn++(siamrpn_alex_dwxcorr_16gpu) without hp search, using 4 training datasets(COCO, DET,VID,YTB)

VOT2016      E-0.393/A-0.618/R-0.238   

```
|  Tracker Name  | Accuracy | Robustness |    Lost     |  EAO  |

----------------------------------------------------------------

| checkpoint_e38 |  0.619   |   0.266    |    57.0     | 0.384 |

| checkpoint_e37 |  0.617   |   0.261    |    56.0     | 0.383 |

| checkpoint_e47 |  0.620   |   0.261    |    56.0     | 0.379 |

| checkpoint_e36 |  0.622   |   0.252    |    54.0     | 0.373 |

| checkpoint_e35 |  0.622   |   0.261    |    56.0     | 0.370 |

| checkpoint_e50 |  0.623   |   0.256    |    55.0     | 0.370 |

| checkpoint_e42 |  0.620   |   0.270    |    58.0     | 0.367 |

| checkpoint_e49 |  0.618   |   0.275    |    59.0     | 0.364 |

| checkpoint_e39 |  0.622   |   0.284    |    61.0     | 0.364 |

| checkpoint_e40 |  0.616   |   0.284    |    61.0     | 0.363 |

| checkpoint_e44 |  0.616   |   0.275    |    59.0     | 0.362 |

| checkpoint_e46 |  0.623   |   0.266    |    57.0     | 0.359 |

| checkpoint_e43 |  0.623   |   0.280    |    60.0     | 0.355 |

| checkpoint_e45 |  0.615   |   0.284    |    61.0     | 0.354 |

| checkpoint_e48 |  0.616   |   0.284    |    61.0     | 0.354 |

| checkpoint_e41 |  0.621   |   0.280    |    60.0     | 0.354 |

```


VOT2018    E-0.352/A-0.576/R-0.290
```

|  Tracker Name  | Accuracy | Robustness |    Lost     |  EAO  |

----------------------------------------------------------------

| checkpoint_e47 |  0.594   |   0.351    |    75.0     | 0.315 |

| checkpoint_e40 |  0.592   |   0.384    |    82.0     | 0.309 |

| checkpoint_e42 |  0.591   |   0.370    |    79.0     | 0.304 |

| checkpoint_e38 |  0.583   |   0.389    |    83.0     | 0.299 |

| checkpoint_e37 |  0.594   |   0.393    |    84.0     | 0.299 |

| checkpoint_e39 |  0.593   |   0.393    |    84.0     | 0.292 |

| checkpoint_e44 |  0.592   |   0.389    |    83.0     | 0.292 |

| checkpoint_e41 |  0.585   |   0.384    |    82.0     | 0.289 |

| checkpoint_e50 |  0.585   |   0.412    |    88.0     | 0.287 |

| checkpoint_e35 |  0.588   |   0.403    |    86.0     | 0.286 |

| checkpoint_e43 |  0.593   |   0.403    |    86.0     | 0.286 |

| checkpoint_e46 |  0.588   |   0.389    |    83.0     | 0.283 |

| checkpoint_e48 |  0.592   |   0.412    |    88.0     | 0.281 |

| checkpoint_e49 |  0.587   |   0.398    |    85.0     | 0.281 |

| checkpoint_e36 |  0.589   |   0.370    |    79.0     | 0.281 |

| checkpoint_e45 |  0.591   |   0.440    |    94.0     | 0.275 |

```

VOT2019  E-A-R  0.260/0.573/0.547

```

|  Tracker Name  | Accuracy | Robustness |     Lost    |   EAO |

----------------------------------------------------------------

| checkpoint_e40 |  0.587   |   0.592    |    118.0    | 0.258 |

| checkpoint_e47 |  0.588   |   0.547    |    109.0    | 0.257 |

| checkpoint_e42 |  0.584   |   0.572    |    114.0    | 0.254 |

| checkpoint_e38 |  0.577   |   0.602    |    120.0    | 0.249 |

| checkpoint_e39 |  0.587   |   0.597    |    119.0    | 0.247 |

| checkpoint_e37 |  0.589   |   0.627    |    125.0    | 0.245 |

| checkpoint_e50 |  0.579   |   0.637    |    127.0    | 0.242 |

| checkpoint_e41 |  0.579   |   0.612    |    122.0    | 0.239 |

| checkpoint_e44 |  0.587   |   0.607    |    121.0    | 0.239 |

| checkpoint_e45 |  0.586   |   0.642    |    128.0    | 0.239 |

| checkpoint_e35 |  0.583   |   0.627    |    125.0    | 0.238 |

| checkpoint_e43 |  0.587   |   0.632    |    126.0    | 0.237 |

| checkpoint_e46 |  0.582   |   0.617    |    123.0    | 0.233 |

| checkpoint_e49 |  0.581   |   0.617    |    123.0    | 0.233 |

| checkpoint_e48 |  0.586   |   0.632    |    126.0    | 0.231 |

| checkpoint_e36 |  0.583   |   0.602    |    120.0    | 0.228 |

```

## Reference
[SenseTime](https://github.com/STVIR/pysot)
```
[1] Li B, Wu W, Wang Q, et al. Siamrpn++: Evolution of siamese visual tracking with very deep networks.Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 4282-4291.
```
