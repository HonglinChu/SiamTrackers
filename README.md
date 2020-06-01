# SiamTrackers

The code will come soon！  https://www.bilibili.com/video/BV1Y64y1T7qs/ 

| Trackers     |  Debug   | Train   | Test  |  Evaluation | Comment | Toolkit | GPU | Version |
| :--------- | :--------: | :------: |:------: |:------: |:------: |:------: |  :------: |  :------: | 
| Siamese     | &radic;|  &radic; | &radic;| &radic;   | &radic;|   | &radic;|- |
| SiamFC     |  &radic; |  &radic; |  &radic;| &radic;| &radic;|got10k|&radic; |unofficial|
| SiamRPN     |  &radic; |  &radic; |  &radic;| &radic;| &radic;|got10k|  &radic;|unofficial|
| DaSiamRPN     |  &radic; |       |  &radic;| &radic;| &radic;|pysot| &radic; | official|
| UpdateNet  |  &radic; |  &radic; |  &radic;| &radic;| &radic;|pysot| &radic; | unofficial|
| SiamDW      |  &radic; |  &radic; |  &radic;| &radic;| &radic;|-| &radic; |official|
| SiamRPN++   |  &radic; |  &radic; |  &radic;| &radic;| &radic;|pysot| &radic;|official|
| SiamMask    |  &radic; |  &radic; |  &radic;| &radic;| &radic;|pysot| &radic; |official|
| SiamFC++   |  &radic; |  &radic; |  &radic;| &radic;| &radic;|pysot&got10k| &radic; |official|

# Description

- Siamese 

基于孪生网络的简单人脸分类实现，支持训练和测试,

- 2016-ECCV-SiamFC 

添加got10k评估工具;对接口进行优化;可评估,可训练和测试;使用VID数据集进行训练,训练速度还是比较快的,记不太清了,大概几个hours;复现结果略低于论文(没有进行超参调节); 使用GOT10K数据集进行训练效果要比论文的结果好一些 

- 2018-CVPR-SiamRPN    

添加got10k评估工具;可评估,可训练和测试;使用YTB和VID数据集进行训练，训练时长>24 hours,复现结果略低于论文(没有进行超参调节);

- 2018-ECCV-DaSiamRPN  

支持VScode单步调试,加pysot评估工具;支持一键测试和评估;测试结果和论文一致;不支持训练;

- 2019-ICCV-UpdateNet   

复现updatenet网络;可测试,训练,评估模型;目前复现发现模型对学习率比较敏感,还在摸索中;训练时间<1 hour,测试时间每个epoch~10min

- 2019-CVPR-SiamDW
之前没有关注这个算法,最近看到这个算法速度还是很快的,后面有时间再复现一下

- 2019-CVPR-SiamRPN++ 

支持VScode单步调试 ；对训练和测试的接口进行了优化；对代码进行部分注释; 修改训练模式,将分布式多机多GPU并行;改成单机多GPU并行;使用四个数据集重新训练SiamRPN++(alexnet版本,训练时间3~4days);在没有进行调超参的情况下精度和论文比较接近
 
- 2019-CVPR-SiamMask  
支持VScode单步调试;对训练和测试的接口进行了优化;对代码进行部分注释;

- 2020-AAAI-SiamFC++ 

支持VScode单步调试,对训练和测试的接口进行了优化;对代码进行部分注释;使用GOT10K数据集重新训练alexnet版本,训练时长~20 hours,测试精度和论文比较接近(备注：官方代码封装的非常好,用到了很多的编程技巧,真的非常考验一个人的代码功底, 但是也给学者理解代码带来很大挑战:(,另外敬请期待我后续更新精简的版本吧:) )

# Model

- SiamRPNVOT.model link: https://pan.baidu.com/s/1V7GMgurufuILhzTSJ4LsYA  password: p4ig   

- SiamRPNOTB.model link: https://pan.baidu.com/s/1mpXaIDcf0HXf3vMccaSriw password: 5xm9   

- SiamRPNBIG.model link: https://pan.baidu.com/s/10v3d3G7BYSRBanIgaL73_Q password: b3b6

# Experimental results 

My environment
- GPU Nvidia-1080 8G  
- CPU Intel® Xeon(R) CPU E5-2650 v4 @ 2.20GHz × 24 
- CUDA 9.0
- System ubuntu 16.04  64 bits
- pytorch 1.1.0
- python 3.7.3

Note:Due to the limitation of computer configuration, i only choose some high speed  algorithms for training and testing on several small  tracking datasets

|   Trackers|       | SiamFC   | DaSiamRPN   |DaSiamRPN | SiamRPN++ |SiamRPN|SiamFC++ |
|:---------: |:-----:|:--------:| :------:    |:------:  |:------:   |:------: |:------:|
|           |       |           |            |         |         |       |        |
|  Backbone |   -    | AlexNet |  AlexNet(OTB/VOT)  |AlexNet(BIG)    | AlexNet(DW)    |AlexNet(UP)    |AlexNet|
|     FPS   | fps>120 |   120   |   180         |  140  |   160      |   180       |   140     |
|           |       |           |            |         |         |       |        |
| OTB100    |  AUC   |  0.570    |   0.655   |  0.646   |   0.648  | 0.637  | **0.680**    |
|           |  DP   |   0.767    |   0.880   |  0.859   |  0.853   |0.851   | **0.884**   |
|           |     |           |            |         |         |       |        |
| UAV123    |  AUC  |   0.504    |   0.586   |  0.604   |  0.578   |0.527   |  **0.623**    |
|           |  DP   |    0.702   |   0.796   | **0.801**    |  0.769   |0.748   |  0.781   |
|           |     |           |            |         |         |       |        |
| UAV20L    |  AUC  |  0.410     |         |   0.524  |  **0.530**   |0.454   |  0.516  |
|           |  DP   |   0.566    |         | **0.691**   |  0.695   |0.617   |  0.613   |
|           |     |           |            |         |         |       |        |
| DTB70     |  AUC  |    0.487   |          |  0.554|   0.588  |        | **0.639**   |
|           |  DP   |    0.735   |         |   0.766|   0.797  |        |  **0.826**   |
|           |     |           |            |         |         |       |        |
| UAVDT     |  AUC  |   0.451 |           |  0.593  |  0.566   |       |  **0.632**    |
|           | DP    |   0.710 |           |  0.836  |  0.793   |       |   **0.846**   |
|           |     |           |            |         |         |       |        |
| VisDrone  | AUC   |    0.510|           |   0.547 |  0.572   |       |  **0.588**    |
|           |  DP   |    0.698|           |   0.722 |   0.764  |       |  **0.784**    |
|           |     |           |            |         |         |       |        |
| VOT2016   |  A    |   0.538    |  0.61      |  0.625   |  0.618   |0.56    |  **0.626**    |
|           | R     |    0.424   |  0.22      |  0.224   |  0.238   |0.26    |   **0.144**   |
|           | E     |    0.262   |  0.411     |  0.439   |  0.393   | 0.344  |  **0.460**    |
|           |Lost   |    91      |           |  48      |  51      |       |    31  |
|           |     |           |            |         |         |       |        |
| VOT2018   | A     |     0.501  |   0.56     |  **0.586**   | 0.576    |0.49    | 0.577   |
|           |  R    |    0.534   |   0.34     |  0.276   |  0.290   |0.46    | **0.183**   |
|           | E     |    0.223   |   0.326    | 0.383    |  0.352   |0.244   | **0.385**   |
|           | Lost  |   114      |           |  59      |   62       |       |   39     |

# Dataset

- UAV123 link: https://pan.baidu.com/s/1AhNnfjF4fZe14sUFefU3iA password: 2iq4

- VOT2018 link: https://pan.baidu.com/s/1MOWZ5lcxfF0wsgSuj5g4Yw password: e5eh

- VisDrone2019 link: https://pan.baidu.com/s/1Y6ubKHuYX65mK_iDVSfKPQ password: yxb6 

- OTB2015 link: https://pan.baidu.com/s/1ZjKgRMYSHfR_w3Z7iQEkYA password: t5i1

- DTB70 link: https://pan.baidu.com/s/1kfHrArw0aVhGPSM91WHomw password: e7qm

- ILSVRC2015 VID link: https://pan.baidu.com/s/1CXWgpAG4CYpk-WnaUY5mAQ password: uqzj 

- NFS link: https://pan.baidu.com/s/1ei54oKNA05iBkoUwXPOB7g password: vng1

- GOT10k link: https://pan.baidu.com/s/172oiQPA_Ky2iujcW5Irlow password: uxds

- UAVDT link: https://pan.baidu.com/s/1K8oo53mPYCxUFVMXIGLhVA password: keva

- YTB-VOS link: https://pan.baidu.com/s/1WMB0q9GJson75QBFVfeH5A password: sf1m 

- YTB-Crop511 （used in siamrpn++ and siammask）link: https://pan.baidu.com/s/112zLS_02-Z2ouKGbnPlTjw password: ebq1

- TCColor128 link: https://pan.baidu.com/s/1v4J6zWqZwj8fHi5eo5EJvQ password: 26d4

- DAVIS2017 link: https://pan.baidu.com/s/1JTsumpnkWotEJQE7KQmh6A password: c9qp

- ytb&vid (used in siamrpn) link: https://pan.baidu.com/s/1gF8PSZDzw-7EAVrdYHQwsA password: 6vkz

- trackingnet link  https://pan.baidu.com/s/1PXSRAqcw-KMfBIJYUtI4Aw code: nkb9  (Note that this link is provided by SiamFC++ author)

# [Reference]

   [1] SiamFC 

   [2] SiamRPN

   [3] DaSiamRPN

   [4] UpdateNet
   
   [5] SiamDW

   [6] SiamRPN++

   [7] SiamMask
   
   [8] SiamFC++
