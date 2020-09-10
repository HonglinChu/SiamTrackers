# Pytorch implementation of UpdateNet

## Description
My code:https://github.com/HonglinChu/UpdateNet
 
BiliBili:https://www.bilibili.com/video/bv1Jg4y1B7tL
 
SiamRPNBIG.model:https://pan.baidu.com/s/10v3d3G7BYSRBanIgaL73_Q password: b3b6

2020-05-15 
``` About create_template.py at line:138  'get_axis_aligned_rect' not exist， please comment get_axis_aigned_rect function

    if reset:   #reset=1 (default)            

        #gt_rect = get_axis_aligned_rect(ground_truth[frame])#x，y，w，h

        rect=ground_truth[frame] #topx,topy,w,h

        gt_rect=np.array([rect[0]-1,rect[1]-1,rect[2],rect[3]])#0-based x,y,w,h

        iou = overlap_ratio(gt_rect, res)

        if iou<=0:# you can choose iou<0.2, iou<0.3,  iou<0.4

            break   
``` 

## File tree
```
├── bin
├── dasiamrpn
├── data
├── datasets
├── models
├── results
├── toolkit
└── updatenet
```

## Data peparation
```
python ./updatenet/create_template.py
```

## Train
```
python ./updatenet/train_upd.py
```

## Test
```
python ./bin/my_test.py
```

## Experiment
- How to produce training data set？
![image](../image/template.png)


My result VOT2018 EAO=0.403 

Original result VOT2018 EAO=0.393 

2020-05-21  
EAO on VOT2018
```
Stage one train from scratch

learning rate Lr5-6  

Checkpoint35      EAO-0.360

Checkpoint36      EAO-0.347

Checkpoint39      EAO-0.325

Checkpoint40      EAO-0.363  

Checkpoint41      EAO-0.350

Checkpoint42      EAO-0.334


Stage two load pretrained model checkpoint40 from stage one

learning rate Lr8-9  

Checkpoint2  EAO-0.362

Checkpoint3  EAO-0.370

Checkpoint4  EAO-0.355


Stage  three load pretrained model  checkpoint3 from stage two

learning rate Lr8-9  

Checkpoint 1   EAO-0.343

Checkpoint 2   EAO-0.376

Checkpoint 3   EAO-0.344

Checkpoint 4   EAO-0.348
```
