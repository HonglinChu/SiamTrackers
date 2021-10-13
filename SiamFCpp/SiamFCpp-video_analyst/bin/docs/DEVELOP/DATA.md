# Data

## Dataset file

```File Tree
project_root/
├── datasets
    ├── COCO  # coco dataset
    │   ├── annotations
    │   │   ├── instances_train2017.json
    │   │   ├── instances_val2017.json
    │   │   ├── ...
    │   ├── train2017
    │   │   ├── 000000000009.jpg
    │   │   ├── 000000000025.jpg
    │   │   ├── ...
    │   └── val2017
    │       ├── 000000000139.jpg
    │       ├── 000000000285.jpg
    │       ├── ...
    ├── GOT-10k  # got10k dataset
    │   ├── train
    │   │   ├── list.txt
    │   │   ├── GOT-10k_Train_000001
    │   │   ├── GOT-10k_Train_000002
    │   │   ├── ...
    │   ├── val
    │   │   ├── list.txt
    │   │   ├── GOT-10k_Val_000001
    │   │   ├── GOT-10k_Val_000002
    │   │   ├── ...
    │   └── test
    │       ├── list.txt
    │       ├── GOT-10k_Test_000001
    │       ├── GOT-10k_Test_000002
    │       ├── ...
    ├── ILSVRC2015  # ILSVRC2015 dataset
    │   ├── Annotations
    │   │   ├── DET
    │   │   │   ├── train
    │   │   │   │   ├── ILSVRC2013_train
    │   │   │   │   │   ├── n00007846
    │   │   │   │   │   │   ├── n00007846_103856.xml
    │   │   │   │   │   │   ├── n00007846_104163.xml
    │   │   │   │   │   │   ├── ...
    │   │   │   │   │   ├── n00141669
    │   │   │   │   │   ├── ...
    │   │   │   │   ├── ILSVRC2014_train_0000
    │   │   │   │   ├── ...
    │   │   │   ├── val
    │   │   │   │   ├── ILSVRC2012_val_00000001.xml
    │   │   │   │   ├── ILSVRC2012_val_00000006.xml
    │   │   │   │   ├── ...
    │   │   └── VID
    │   │       ├── train
    │   │       │   ├── ILSVRC2015_VID_train_0000
    │   │       │   │   ├── ILSVRC2015_train_00000000
    │   │       │   │   │   ├── 000000.xml
    │   │       │   │   │   ├── 000001.xml
    │   │       │   │   │   ├── ...
    │   │       │   │   ├── ILSVRC2015_train_00001000
    │   │       │   │   ├── ...
    │   │       │   ├── ILSVRC2015_VID_train_0001
    │   │       │   ├── ...
    │   │       └── val
    │   │           ├── ILSVRC2015_val_00000000
    │   │           │   ├── 000000.xml
    │   │           │   ├── 000001.xml
    │   │           │   ├── ...
    │   │           ├── ILSVRC2015_val_00000001
    │   │           ├── ...
    │   └── Data
    │       ├── DET
    │       │   ├── train
    │       │   │   ├── ILSVRC2013_train
    │       │   │   │   ├── n00007846
    │       │   │   │   │   ├── n00007846_103856.JPEG
    │       │   │   │   │   ├── n00007846_104163.JPEG
    │       │   │   │   │   ├── ...
    │       │   │   │   ├── n00141669
    │       │   │   │   ├── ...
    │       │   ├── val
    │       │   │   ├── ILSVRC2012_val_00000001.JPEG
    │       │   │   ├── ILSVRC2012_val_00000006.JPEG
    │       │   │   ├── ...
    │       └── VID
    │           ├── train
    │           │   ├── ILSVRC2015_VID_train_0000
    │           │   │   ├── ILSVRC2015_train_00000000
    │           │   │   │   ├── 000000.JPEG
    │           │   │   │   ├── 000001.JPEG
    │           │   │   │   ├── ...
    │           │   │   ├── ILSVRC2015_train_00001000
    │           │   │   ├── ...
    │           │   ├── ILSVRC2015_VID_train_0001
    │           │   ├── ...
    │           └── val
    │               ├── ILSVRC2015_val_00000000
    │               │   ├── 000000.JPEG
    │               │   ├── 000001.JPEG
    │               │   ├── ...
    │               ├── ILSVRC2015_val_00000001
    │               ├── ...
    ├── LaSOT  # LaSOT dataset
    │   ├── airplane
    │   │   ├── airplane-1
    │   │   │   ├── groundtruth.txt
    │   │   │   ├── full_occlusion.txt
    │   │   │   ├── out_of_view.txt
    │   │   │   ├── nlp.txt
    │   │   │   └── img
    │   │   │       ├── 00000001.jpg
    │   │   │       ├── 00000002.jpg
    │   │   │       ├── ...
    │   │   ├── airplane-10
    │   │   ├── ...
    │   └── bicycle
    │       ├── ...
    ├── TrackingNet  # Tracking Net
    │   ├── TRAIN_0
    │   │   ├── frames
    │   │   │   ├── -3TIfnTSM6c_2
    │   │   │   ├── a1qoB1eERn0_0
    │   │   │   ├── ...
    │   │   └── anno
    │   │       ├── -3TIfnTSM6c_2.txt
    │   │       ├── a1qoB1eERn0_0.txt
    │   │       ├── ...
    │   ├── TRAIN_1
    │   ├── ...
    │   └── TEST
    └── VOT  # test entry point
        ├── vot2018  # VOT2018
        │   ├── VOT2018.json  # 
        │   └── VOT2018  # 
        │       ├── ant1  # 
        │       ├── ant3  # 
        │       ├── ...  # 
        └── vot2019  # VOT2019
            ├── VOT2019.json  # 
            └── VOT2019  # 
                ├── agility  # 
                ├── ball3  # 
                ├── ...  # 

```

## Misc

* Please clean cache after moving your dataset folder
  * We cache absolute path for accessing efficiency reason.
