# Pytorch implementation of UpdateNet

My code link: https://github.com/HonglinChu/UpdateNet
 
UpdateNet复现视频讲解  https://www.bilibili.com/video/bv1Jg4y1B7tL
 
SiamRPNBIG.model 链接: https://pan.baidu.com/s/10v3d3G7BYSRBanIgaL73_Q 提取码: b3b6

内嵌Dasiamrpn, 训练完UpdateNet,可以直接进行测试和评估,支持;模型更新支持UpdateNet和Linear更新方式

2020-05-15 修正
``` 关于create_template.py文件中138行  'get_axis_aligned_rect'不存在的问题，注释掉get_axis_aigned_rect函数

    if reset:   #reset 默认是1               

        #gt_rect = get_axis_aligned_rect(ground_truth[frame])#x，y，w，h

        rect=ground_truth[frame] #topx,topy,w,h

        gt_rect=np.array([rect[0]-1,rect[1]-1,rect[2],rect[3]])#0-based x,y,w,h

        iou = overlap_ratio(gt_rect, res)

        if iou<=0:#iou<0.2, iou<0.3,  iou<0.4

            break   
``` 

# File Tree
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
# Experiment

2020-06-08
目前已经在第三个阶段, VOT2018上复现出EAO=0.403（VS 0.393）的结果

2020-05-21  
以下EAO均在VOT2018上测试 

第一阶段 train from scratch

学习率 Lr5-6  
Checkpoint35      EAO-0.360

Checkpoint36      EAO-0.347

Checkpoint39      EAO-0.325

Checkpoint40      EAO-0.363  

Checkpoint41      EAO-0.350

Checkpoint42      EAO-0.334


第二阶段 加载预训练 checkpoint40
Lr 8-9 

Checkpoint2  EAO-0.362

Checkpoint3  EAO-0.370

Checkpoint4  EAO-0.355


第三阶段 加载预训练 checkpoint3

Lr 8-9 

Checkpoint 1   EAO-0.343

Checkpoint 2   EAO-0.376

Checkpoint 3   EAO-0.344

Checkpoint 4   EAO-0.348

