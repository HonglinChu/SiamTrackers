# MODEL_ZOO

## Download links

Models & Raw results:

* [Google Drive](https://drive.google.com/open?id=1XhWIU1KIt9wvFpzZqEDaX-GrgZ9AVcOC)
* [Tencent Weiyun](https://share.weiyun.com/56C92l4), code: wg47g7

## Models

### VOT2018

VOT test configuration directory: _experiments/siamfcpp/test/vot_


| Backbone | Pipeline | Dataset | A | R | EAO | FPS@GTX2080Ti | FPS@GTX1080Ti | Config. File |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AlexNet | SiamFCppTracker | VOT2018 |0.576 | 0.183 | 0.393| ~200| ~185 | [siamfcpp_alexnet.yaml](../../experiments/siamfcpp/test/vot/siamfcpp_alexnet.yaml)
 | 
| GoogLeNet | SiamFCppTracker | VOT2018 | 0.583 | 0.173 | 0.426 | ~80 | ~65 | [siamfcpp_googlenet.yaml](../../experiments/siamfcpp/test/vot/siamfcpp_googlenet.yaml) |

#### Multi-template

| Backbone | Pipeline | Dataset | A | R | EAO | FPS@GTX2080Ti | FPS@GTX1080Ti | Config. File |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AlexNet | SiamFCppMultiTempTracker| VOT2018 | 0.597 | 0.215 | 0.370 | ~90 | ~75 | [siamfcpp_alexnet-multi_temp.yaml](../../experiments/siamfcpp/test/vot/multi_temp/siamfcpp_alexnet-multi_temp.yaml) | siamfcpp-alexnet-vot-md5_18fd31a2f94b0296c08fff9b0f9ad240.pkl|
| GoogLeNet | SiamFCppMultiTempTracker | VOT2018 | 0.587 | 0.150 |  0.467 | ~50 | ~45 | [siamfcpp_googlenet-multi_temp.yaml](../../experiments/siamfcpp/test/vot/multi_temp/siamfcpp_googlenet-multi_temp.yaml) |

__Nota__:

Points reported here are reproducible with PyTorch<=1.2.0. For PyTorch>=1.3.0, the reproducibility is not guaranteed due to a "breaking change" of PyTorch. See "Breaking Changes" under [release 1.3.0](https://github.com/pytorch/pytorch/releases) for detail.

However, we still recommend using the newest version of PyTorch as earlier versions usually carry numerous historical bugs (e.g. bugs with dataloader, ddp, etc.).

### GOT-10k

GOT-10k test configuration directory_experiments/siamfcpp/test/got10k_

| Backbone | Pipeline | Dataset | AO (val) | SR.50 (val) | SR.75 (val) | AO (test) | SR.50 (test) | SR.75 (test) | Config. File |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AlexNet | SiamFCppTracker | GOT-10k | 72.0 | 85.0 | 63.3 | 52.6 | 62.5 | 34.7 | [siamfcpp_alexnet-got.yaml](../../experiments/siamfcpp/test/got10k/siamfcpp_alexnet-got.yaml) |
| GoogLeNet | SiamFCppTracker | GOT-10k | 76.4 | 90.4 | 71.8 | 60.4 | 73.7 | 46.4 | [siamfcpp_googlenet-got.yaml](../../experiments/siamfcpp/test/got10k/siamfcpp_googlenet-got.yaml) |
| ShuffleNetV2x0.5 | SiamFCppTracker | GOT-10k | 74.2 | 87.0| 67.1 | 52.9 | 61.7 | 38.1 | [siamfcpp_shufflenetv2x0_5-got.yaml](../../siamfcpp_shufflenetv2x0_5-got.yaml) |
| ShuffleNetV2x1.0 | SiamFCppTracker | GOT-10k-val | 76.6 | 88.8 | 71.5 | 57.9 | 68.1 | 43.6 | [siamfcpp_shufflenetv2x1_0-got.yaml](../../experiments/siamfcpp/test/got10k/siamfcpp_shufflenetv2x1_0-got.yaml) |

### LaSOT

| Backbone | Pipeline | Dataset | Success | Precision | Normalized Precision | Config. File |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GoogLeNet | SiamFCppTracker | LaSOT-test | 55.7 | 55.6 | 58.9 | [siamfcpp_googlenet-lasot.yaml](../../experiments/siamfcpp/test/lasot/siamfcpp_googlenet-lasot.yaml) |

### TrackingNet

| Backbone | Pipeline | Training Data | Test Dataset | Success | Precision | Normalized Precision | Config. File |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GoogLeNet | SiamFCppTracker | TrackingNet-TRAIN | TrackingNet-TEST | 74.5 | 68.5 | 79.8 | [siamfcpp_googlenet-trackingnet.yaml](../../experiments/siamfcpp/test/trackingnet/siamfcpp_googlenet-trackingnet.yaml) |
| GoogLeNet | SiamFCppTracker | fulldata | TrackingNet-TEST | 75.3 | 69.5 | 80.9 | [siamfcpp_googlenet-trackingnet-fulldata.yaml](../../experiments/siamfcpp/test/trackingnet/siamfcpp_googlenet-trackingnet-fulldata.yaml) |


P.S. _fulldata_ denotes COCO, VID, TrackingNet-TRAIN, ILSVRC-VID/DET, LaSOT, GOT10k

### OTB-2015

| Backbone | Pipeline | Dataset | Success | Precision | Config. File|
|:---:|:---:|:---:|:---:|:---:|:---:|
| AlexNet | SiamFCppTracker | OTB2015 | 68.0 | 88.4 | [siamfcpp_alexnet-otb.yaml](../../experiments/siamfcpp/test/otb/siamfcpp_alexnet-otb.yaml) |
| GoogLeNet | SiamFCppTracker | OTB2015 | 68.2 | 89.6 | [siamfcpp_googlenet-otb.yaml](../../experiments/siamfcpp/test/otb/siamfcpp_googlenet-otb.yaml) |


## Improvements

### Large Search Region (x_size)

Augmenting the search region may further improve the performance on some benchmarks. Here we report some of them.

#### Large _x_size_ on GOT-10k

| Backbone | Pipeline | Dataset | x_size | score_size | AO (val) | SR.50 (val) | SR.75 (val) | AO (test) | SR.50 (test) | SR.75 (test) | Config. File |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GoogLeNet | SiamFCppTracker | GOT-10k | 303 | 19 | 76.4 | 90.4 | 71.8 | 60.4 | 73.7 | 46.4 | [siamfcpp_googlenet-got.yaml](../../experiments/siamfcpp/test/got10k/siamfcpp_googlenet-got.yaml) |
| GoogLeNet | SiamFCppTracker | GOT-10k | 335 | 23 | 76.6 | 90.6 | 71.9 | 61.0 | 74.2 | 46.7 | [x_size/siamfcpp_googlenet-got.yaml](../../experiments/siamfcpp/test/got10k/x_size/siamfcpp_googlenet-got.yaml) |


#### Large _x_size_ on LaSOT

| Backbone | Pipeline | Dataset | x_size | score_size | Success | Precision | Normalized Precision | Config. File |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GoogLeNet | SiamFCppTracker | LaSOT-test | 303 | 19 | 55.7 | 55.6 | 58.9 | [siamfcpp_googlenet-lasot.yaml](../../experiments/siamfcpp/test/lasot/siamfcpp_googlenet-lasot.yaml) |
| GoogLeNet | SiamFCppTracker | LaSOT-test | 351 | 25 | 56.4 | 56.4 | 59.8 | - |
| GoogLeNet | SiamFCppTracker | LaSOT-test | 367 | 27 | 56.6 | 56.4 | 60.0 | - |
| GoogLeNet | SiamFCppTracker | LaSOT-test | 383 | 29 | 57.1 | 57.2 | 60.5 | - |
| GoogLeNet | SiamFCppTracker | LaSOT-test | 399 | 31 | __57.7__ | 58.2 | 61.3 | [x_size/siamfcpp_googlenet-lasot.yaml](../../experiments/siamfcpp/test/lasot/x_size/siamfcpp_googlenet-lasot.yaml) |
| GoogLeNet | SiamFCppTracker | LaSOT-test | 415 | 33 | 57.4 | 57.7 | 60.9 | - |

P.S. _window_influence_ may require tuning as search region size slightly change the shape of window of score penalization.


## Pipeline

* SiamFCppTracker
  * [videoanalyst/pipeline/tracker/tracker_impl/siamfcpp_track.py](../videoanalyst/pipeline/tracker/tracker_impl/siamfcpp_track.py)
* SiamFCppMultiTempTracker
  * [videoanalyst/pipeline/tracker/tracker_impl/siamfcpp_track_multi_temp.py](../videoanalyst/pipeline/tracker/tracker_impl/siamfcpp_track_multi_temp.py)

### Remarks

* The results reported in our paper were produced by the implement under the internal deep learning framework. Afterwards, we reimplement our tracking method under PyTorch and there could be some differences between the reported results (under internal framework) and the real results (under PyTorch).
* Differences in hardware configuration (e.g. CPU style / GPU style) may influence some indexes (e.g. FPS)
  * Raw results here have been produced on a shared computing node equipped with _Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz_ and _Nvidia GeForce RTX 2080Ti_ .
  * "~" in the colomns for FPS denotes approximate values. FPS may vary due to factors other than code (e.g. hardware configuration / running status of machine).
* For VOT benchmark, models have been trained on ILSVRC-VID/DET, YoutubeBB, COCO, LaSOT, and GOT-10k (as described in our paper).

## Reproducibility

We have already observed several issues that are related to the reproducibility of the results under VOT benchmark. For example, under pytorch==1.1.0/1.2.0, the results of siamfcpp-googlenet are correct while under pytorch==1.3.0/1.4.0 not.

Following issues would influence the reproducibility of the results of existing models on VOT benchmark:

* PyTorch version
  * e.g. Type Promotion between 1.2.0 and 1.3.0, see Type Promotion on [PyTorch release notes](https://github.com/pytorch/pytorch/releases).
* CUDA/CUDNN version
  * 10.0 / 10.1
  * should be matched with the PyTorch (rebuilding may be needed)
* OpenCV version
  * Slight performance drop has been observed with the following change: 3.2.0.6 -> 4.1.0.25

We recommend keeping up-to-date with latest package version, and thus the points reported here counld be slightly away from the real points. Feel free to point them out in Issues if it is the case so that we can correct them.

Nevertheless, reproducibility of training under GOT-10k has been confirmed with repetition. Thus, there are no need to change software version (package/CUDA/CUDNN) unless you are obligated to verify the VOT result.

In addition, we strongly recommend to train and benchmark trackers on datasets like [GOT-10k](http://got-10k.aitestunion.com), not only because of its rigurous split of train/val/test, but also due to its large scale and diversity which make results stable.
