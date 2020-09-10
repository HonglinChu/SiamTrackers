# Pytorch implementation of UpdateNet

## Description
My code: https://github.com/HonglinChu/UpdateNet
 
BiliBili: https://www.bilibili.com/video/bv1Jg4y1B7tL
- Note: About create_template.py at line:138  'get_axis_aligned_rect' not exist， please comment get_axis_aigned_rect function
```

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
How to produce templates ? You can choose iou<0.2, iou<0.3,  iou<0.4

python ./updatenet/create_template.py
```
![image](../image/template.png)


## Model

SiamRPNBIG.model:https://pan.baidu.com/s/10v3d3G7BYSRBanIgaL73_Q password: b3b6

## Train
```
python ./updatenet/train_upd.py
```

## Test
```
python ./bin/my_test.py
```

## Experiment


- My result VOT2018 EAO=0.403, original result VOT2018 EAO=0.393 

- How to train UpdateNet on VOT2018 ?

- Stage 1.1
```
Generate templates by linear update, train from scratch

you can try learning rate Lr5-6 ,  Lr6-7, Lr7-8

checpoint1   EAO  xxx

...

checkpoint50 EAO  xxx
```
- Stage 1.2
```
Load pretrained model(the best checkpoint from stage 1.1), train from checkpoint

you can try learning rate Lr7-8 ,  Lr8-9, Lr9-10

checpoint1   EAO  xxx

...

checkpoint50 EAO  xxx
```

- Stage 2.1
```
Generate templates by UpdateNet model (choose best checkpoint from stage 1.2) , train from scratch

you can try learning rate Lr5-6 ,  Lr6-7, Lr7-8

checpoint1   EAO  xxx

...

checkpoint50 EAO  xxx
```

- Stage 2.2
```
Load pretrained model(choose best checkpoint from stage 2.1),train from checkpoint

you can try learning rate Lr7-8 ,  Lr8-9, Lr9-10

checpoint1   EAO  xxx

...

checkpoint50 EAO  xxx

```

- Stage 3.1
```
Generate templates by UpdateNet model (choose best checkpoint from stage 2.2) , train from scratch

you can try learning rate Lr5-6 ,  Lr6-7, Lr7-8

checpoint1   EAO  xxx

...

checkpoint50 EAO  xxx
```

- Stage 3.2
```
Load pretrained model(choose best checkpoint from stage 3.1), train from checkpoint

you can try learning rate Lr7-8 ,  Lr8-9, Lr9-10

checpoint1   EAO  xxx

...

checkpoint50 EAO  xxx

```
