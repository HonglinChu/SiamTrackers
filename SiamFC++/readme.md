# Pytorch implementation of SiamFC++

## Description
Official code：https://github.com/MegviiDetection/video_analyst (Recommend)

My code: https://github.com/HonglinChu/SiamFCpp 

## SiamFC++

The official code：https://github.com/MegviiDetection/video_analyst (Recommend)

If you download our code, you need download 2  other files("models" , "snapshots") from [BaiDuYun](https://pan.baidu.com/s/1UELKI7DNFCjQ-BM9ebL5_w) password: 7qt8 , and put them into the project.

## SOT 
 Official Guide https://github.com/MegviiDetection/video_analyst
* [SOT_SETUP.md](docs/TUTORIALS/SOT_SETUP.md): instructions for setting-up
* [SOT_MODEL_ZOO.md](docs/TUTORIALS/SOT_MODEL_ZOO.md): description of released sot models
* [SOT_TRAINING.md](docs/TUTORIALS/SOT_TRAINING.md): details related to training
* [SOT_TEST.md](docs/TUTORIALS/SOT_TEST.md): details related to test

## Testing
```
python ./bin/my_test.py
```
## Training
```
python ./bin/my_train.py
```

## File Tree
```
project_root/
├── bin
│   ├── hpo.py
│   ├── my_test.py
│   ├── my_train.py
│   ├── paths.py
│   ├── shell_script
│   ├── test.py
│   └── train.py
├── datasets
│   ├── COCO 
│   ├── DTB70
│   ├── GOT-10k
│   ├── ILSVRC2015 
│   ├── LaSOT 
│   ├── OTB
│   ├── UAV123
│   ├── UAVDT 
│   ├── VisDrone 
│   └── VOT
├── demo
│   ├── main
│   ├── resources
│   └── tools
├── docs
│   ├── DEVELOP
│   ├── resources
│   ├── TEMPLATES
│   └── TUTORIALS
├── experiments
│   └── siamfcpp
├── models
│   ├── alexnet
│   └── siamfcpp
├── results
│   ├── GOT-Benchmark
│   ├── VOT2016
│   └── VOT2018
├── siamfcpp
│   ├── config
│   ├── data
│   ├── engine
│   ├── evaluation
│   ├── model
│   ├── optim
│   ├── pipeline
│   └── utils
└── snapshots
    └── siamfcpp_alexnet-got
```
