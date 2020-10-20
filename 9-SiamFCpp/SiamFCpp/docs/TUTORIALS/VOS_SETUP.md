# Setup-VOS

## Install requirements


## Download models
### SiamFCpp tracker
* [Tencent Weiyun](https://share.weiyun.com/56C92l4), code: wg47g7 with md5sum ad240
### SAT segmentor
* [baidu yun](https://pan.baidu.com/s/1uZ26iZyVJm50dJ3GoLCQ9w), code: rcsn
* [Google Drive](https://drive.google.com/open?id=1UXshq4k9WKx4hNkdpOagJLXPR57ZkBkg)

## Compile evaluation toolkit

```Bash
bash compile.sh
```

## Set datasets

### Davis
Download [Davis](https://davischallenge.org/davis2017/code.html)
Set soft link to dataset directory 

```bash
ln -s path_to_datasets datasets/DAVIS
```
### YoutubeVOS
Download [YoutubeVOS2018](https://youtube-vos.org/dataset/)
Set soft link to dataset directory 

```bash
ln -s path_to_datasets datasets/youtubevos
```

### COCO
Download [COCO2017](http://cocodataset.org/#download)
Set soft link to dataset directory 

```bash
ln -s path_to_datasets datasets/coco2017
```

At _path_to_datasets_:

```File Tree
datasets
└── DAVIS
    ├── Annotations
    |        ├── 480p # annotation for davis2017
    |        └── 480p_2016 # annotation for davis2016
    ├── ImageSets
    └── ...
└── coco2017
    ├── annotations
    |        ├── instances_train2017.json
    |        └── instances_val2017.json
    ├── train2017
    └── val2017
└── youtubevos
    ├── train
    └── valid

```


## Set models
At _path_to_models_:

```File Tree
models
├── siamfcpp
    |── siamfcpp-alexnet-vot-md5_18fd31a2f94b0296c08fff9b0f9ad240.pkl
├── sat
    |── sat_res50_davis17_2518a.pkl
    |── sat_res18_davis17_7ccb1.pkl
├── resnet
    |── res-50-imagenet-fff49.pkl
    |── res-18-imagenet-7ccb1.pkl
```
