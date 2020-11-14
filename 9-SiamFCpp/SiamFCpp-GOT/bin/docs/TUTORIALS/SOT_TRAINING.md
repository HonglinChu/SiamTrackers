# VideoAnalyst

## Training

By running [tools/train_test/got10k/train_test-alexnet.sh](../tools/train_test/got10k/train_test-alexnet.sh) or [tools/train_test/got10k/train_test-googlent.sh](../tools/train_test/got10k/train_test-googlenet.sh), a training process succeeded by a benchmarking process will be lauched.

Usage of Python script:

e.g. path/to/config.yaml = experiments/siamfcpp/train/siamfcpp_alexnet-trn.yaml

```Bash
python3 ./main/train.py --config 'path/to/config.yaml'
python3 ./main/test.py --config 'path/to/config.yaml'
```

Resuming from epoch number

```Bash
python3 ./main/train.py --config 'path/to/config.yaml' --resume 10
```

Resuming from certain snapshot file

```Bash
python3 ./main/train.py --config 'path/to/config.yaml' --resume 'snapshots/siamfcpp_alexnet/epoch-10.pkl'
```
Resuming from the latest snapshot file

```Bash
python3 ./main/train.py --config 'path/to/config.yaml' --resume latest
```

### Training with full data

Full data list:

* ILSVRC-VID
* TrackingNet
* COCO
* ILSVRC-DET
* LaSOT
* GOT10k

```Bash
python3 ./main/train.py --confg 'experiments/siamfcpp/train/fulldata/siamfcpp_googlenet-trn-fulldata.yaml'
```


### Training with PyTorch Distributed Data Parallel (DDP)

```Bash
python3 -W ignore ./main/dist_train.py --config 'experiments/siamfcpp/train/fulldata/siamfcpp_alexnet-dist_trn.yaml'
```

Nota: _-W ignore_ neglects warning to ensure the program exits normally so that _test.py_ can run after it.

#### Quick fix for DDP

* "RuntimeError: error executing torch_shm_manage"
  * CUDA version mismatch with installed PyTorch, two solution
    * Install CUDA version that matches installed PyTorch, or
    * Compile PyTorch insteatd of prebuilt binaries so that it matches the installed CUDA
* "RuntimeError: cuda runtime error (2) : out of memory at"
  * pin_memory = False (train.data.pin_memory)
* Other comments
  * DDP may require higher version of PyTorch (e.g. torch==1.4.0) as early versions of PyTorch 1.x seem to have bugs in DDP.

#### Performance Influence of DDP

As reported in several issues (e.g. [Training performance degrades with DistributedDataParallel](https://discuss.pytorch.org/t/training-performance-degrades-with-distributeddataparallel/47152) / [DDP on 8 gpu work much worse then on single](https://discuss.pytorch.org/t/ddp-on-8-gpu-work-much-worse-then-on-single/63358) / [Performance degrades with DataParallel](https://discuss.pytorch.org/t/performance-degrades-with-dataparallel/57452)) and based on our observation, using DDP in a plug-in-and-play way may cause performance degradation. Here we report our results with DDP:

1xlr of DP

| Exp | Pipeline | Dataset | AO (DP)(*) | AO (DDP)(+) | Diff. | Hardware |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| alexnet | SiamFCppTracker | GOT-10k-val | 72.2 | 71.9 | -0.3 | 2080ti |
| alexnet | SiamFCppTracker | GOT-10k-test | 53.1 | 52.6 | -0.5 | 2080ti |
| googlenet | SiamFCppTracker | GOT-10k-val | 76.3 | 76.0 | -0.3 | 2080ti |
| googlenet | SiamFCppTracker | GOT-10k-test | 60.0 | 58.1 | -1.9 | 2080ti |
| shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-val | 73.1 | 72.5 | -0.6 | 2080ti |
| shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-test | 53.1 | 53.0 | -0.1 | 2080ti |
| shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-val | 76.1 | 75.2 | -0.9 | 2080ti |
| shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-test | 55.6 | 55.7 | +0.1 | 2080ti |

2xlr of DP

| Exp | Pipeline | Dataset | AO (DP)(*) | AO (DDP)(+) | Diff. | Hardware |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| alexnet | SiamFCppTracker | GOT-10k-val | 72.2 | 73.0 | +0.8 | 2080ti |
| alexnet | SiamFCppTracker | GOT-10k-test | 53.1 | 53.8 | +0.7 | 2080ti |
| googlenet | SiamFCppTracker | GOT-10k-val | 76.3 | 75.9 | -0.4 | 2080ti |
| googlenet | SiamFCppTracker | GOT-10k-test | 60.0 | 59.5 | -0.5 | 2080ti |
| shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-val | 73.1 | 72.3 | -0.8 | 2080ti |
| shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-test | 53.1 | 53.0 | -0.1 | 2080ti |
| shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-val | 76.1 | 76.7 | +0.5 | 2080ti |
| shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-test | 55.6 | 57.1 | +1.5 | 2080ti |

* (*): AO (DP) reported here comes from the average reported in the following _Stability_ section .
* (+): AO (DDP) reported here are performance of a single training of each experiment. Average level need to be determined with more training trials further.

Several hypotheses need to be taken into consideration with regard to the slight performance degradation :

* BN implementation (sync / non-sync)
  * Currently, we still use normal BN in _model_ module
* Gradient reducing method
  * Currently, we do not change learning rate. However, gradient reducing methods may not be the same in DP and DDP, which requires a further adjustment of learning rate.

We plan to continuously track the issues of DDP to align its performance with DP.

### Configuration Files

This project use [yacs](https://github.com/rbgirshick/yacs) for configuration/hyper-parameter management. Configuration .yaml files are givin under [experiments/train/](../experiments/train/).

Before the training starts, the merged configuration file will be backed up at _EXP_SAVE/EXP_NAME/logs_.

### Training details of SOT

Harware configuration:

* #CPU: 32
* #GPU: 4
* Memory: 64 GiB
* Training with PyTorch DataParallel (DP)

Several indexes related to training process have been listed in the table bellow:

|Experiment|Batch size|#Workers|#CPUs|#GPUs|Epoch speed|Iteration speed|
|---|---|---|---|---|---|---|
|siamfcpp-alexnet| 32 | 32| 32 | 4 |45min/epoch for epoch 0-19| 5it/s for /epoch 0-19 |
|siamfcpp-googlenet| 128 | 64 | 32 | 4 |20min/epoch for epoch 0-9; 24min/epoch for epoch 10-19 | 1.01it/s for epoch 0-9; 1.25s/it for epoch 10-19|
|siamfcpp-shufflenetv2x1_0| 32 | 32 | 32 | 4 |40min/epoch for epoch 0-19| 5it/s for epoch 0-19 |

## Issues with PyTorch Version

### TL;DR

Enlarging learning rate by 2 times while moving from PyTorch==1.1.0 & CUDA==10.0 to PyTorch==1.4.0 & CUDA==10.1

### Description

We empirically found that the learning rate need to be multiplied by two (0.04->0.08 for SiamFC++ (SOT)) in order to yield the same trainig results (verified on GOT-10k) while moving from PyTorch==1.1.0 & CUDA==10.0 to PyTorch==1.4.0 & CUDA==10.1.

We encourage to use PyTorch==1.4.0 & CUDA==10.1, although the reproduction of some inference results (mainly the VOT results) still require previous version at this point. We plan to update them to be compatible with the latest PyTorch&CUDA.

The default yaml configuration files have been updated for that.

## Stability

Stability test has been conducted on GOT-10k benchmark for our experiments (alexnet/googlenet/shufflenetv2x0.5/shufflenetv2x1.0). Concretely, for each experiment, we train on four different (virtual) PC and perform benchmarking on _val_ and _test_ subsets.

Results are listed as follows and they shall serve as reference for reproduction of the experiments by users of this code base.

__Environment__: PyTorch==1.1.0 & CUDA==10.0

### alexnet

| Site ID | Exp | Pipeline | Dataset | AO | Hardware |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | alexnet | SiamFCppTracker | GOT-10k-val | 72.1 | 2080ti |
| 2 | alexnet | SiamFCppTracker | GOT-10k-val | 72.4 | 2080ti |
| 3 | alexnet | SiamFCppTracker | GOT-10k-val | 71.9 | 2080ti |
| 4 | alexnet | SiamFCppTracker | GOT-10k-val | 72.2 | 1080ti |
| 1 | alexnet | SiamFCppTracker | GOT-10k-test | 53.8 | 2080ti |
| 2 | alexnet | SiamFCppTracker | GOT-10k-test | 53.6 | 2080ti |
| 3 | alexnet | SiamFCppTracker | GOT-10k-test | 51.2 | 2080ti |
| 4 | alexnet | SiamFCppTracker | GOT-10k-test | 53.7 | 1080ti |

### googlent

| Site ID | Exp | Pipeline | Dataset | AO | Hardware |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | googlenet | SiamFCppTracker | GOT-10k-val | 76.3 | 2080ti |
| 2 | googlenet | SiamFCppTracker | GOT-10k-val | 76.7 | 2080ti |
| 3 | googlenet | SiamFCppTracker | GOT-10k-val | 76.2 | 2080ti |
| 4 | googlenet | SiamFCppTracker | GOT-10k-val | 76.0 | 1080ti |
| 1 | googlenet | SiamFCppTracker | GOT-10k-test | 60.3 | 2080ti |
| 2 | googlenet | SiamFCppTracker | GOT-10k-test | 59.8 | 2080ti |
| 3 | googlenet | SiamFCppTracker | GOT-10k-test | 59.2 | 2080ti |
| 4 | googlenet | SiamFCppTracker | GOT-10k-test | 60.7 | 1080ti |

### shufflenetv2x0_5

| Site ID | Exp | Pipeline | Dataset | AO | Hardware |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-val | 73.1 | 2080ti |
| 2 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-val | 73.8 | 2080ti |
| 3 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-val | 73.0 | 2080ti |
| 4 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-val | 72.6 | 1080ti |
| 1 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-test | 53.5 | 2080ti |
| 2 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-test | 52.7 | 2080ti |
| 3 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-test | 53.2 | 2080ti |
| 4 | shufflenetv2x0_5 | SiamFCppTracker | GOT-10k-test | 53.1 | 1080ti |

### shufflenetv2x1_0

| Site ID | Exp | Pipeline | Dataset | AO | Hardware |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-val | 76.3 | 2080ti |
| 2 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-val | 76.0 | 2080ti |
| 3 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-val | 76.1 | 2080ti |
| 4 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-val | 76.1 | 1080ti |
| 1 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-test | 57.2 | 2080ti |
| 2 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-test | 54.2 | 2080ti |
| 3 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-test | 55.4 | 2080ti |
| 4 | shufflenetv2x1_0 | SiamFCppTracker | GOT-10k-test | 55.4 | 1080ti |
