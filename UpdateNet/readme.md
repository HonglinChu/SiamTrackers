- 2020-05-14 UpdateNet代码已经复现完成,可训练测试和评估,需要你另外自己下载SiamRPNBIG模型
- 内嵌Dasiamrpn, 训练完UpdateNet,可以直接进行测试和评估,支持 
- 模型更新支持UpdateNet和Linear更新方式
- https://www.bilibili.com/video/bv1Jg4y1B7tL UpdateNet复现视频讲解
- SiamRPNBIG.model 链接: https://pan.baidu.com/s/10v3d3G7BYSRBanIgaL73_Q 提取码: b3b6

- 2020-05-15 修正1
` 关于create_template.py文件中138行  'get_axis_aligned_rect'不存在的问题，注释掉get_axis_aigned_rect函数

if reset:   #reset 默认是1               

    #gt_rect = get_axis_aligned_rect(ground_truth[frame])#x，y，w，h

    rect=ground_truth[frame] #topx,topy,w,h

    gt_rect=np.array([rect[0]-1,rect[1]-1,rect[2],rect[3]])#0-based x,y,w,h

    iou = overlap_ratio(gt_rect, res)

    if iou<=0:#这个条件是不是太宽了 iou<0.5才可以把

        break   
`


`
2020-05-21 

第一阶段 train from scratch

Lr5-6

35 0.360

36 0.347

39 0.325

40  0.363  

41  0.350

42  0.334

第二阶段 加载预训练 checkpoint40

Lr 8-9 

Checkpoint2  0.362

Checkpoint3  0.370

Checkpoint4  0.355


第三阶段 加载预训练 checkpoint3

Lr 8-9 

Checkpoint  1   0.343

Checkpoint 2   0.376

Checkpoint 3    0.344

Checkpoint 4    0.348

`
