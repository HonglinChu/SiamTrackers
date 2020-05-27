# SiamTrackers

The code will come soon！  https://www.bilibili.com/video/BV1Y64y1T7qs/ 

| Trackers     |  Debug   | Train   | Test  |  Evaluation | Comment | Toolkit | GPU | Version |
| :--------- | :--------: | :------: |:------: |:------: |:------: |:------: |  :------: |  :------: | 
| Siamese     | &radic;|  &radic; | &radic;| &radic;   | &radic;|   | &radic;|- |
| SiamFC     |  &radic; |  &radic; |  &radic;| &radic;| &radic;|got10k|&radic; |unofficial|
| SiamRPN     |  &radic; |  &radic; |  &radic;| &radic;| &radic;|got10k|  &radic;|unofficial|
| DaSiamRPN     |  &radic; |       |  &radic;| &radic;| &radic;|pysot| &radic; | official|
| UpdateNet  |  &radic; |  &radic; |  &radic;| &radic;| &radic;|pysot| &radic; | unofficial|
| SiamRPN++   |  &radic; |  &radic; |  &radic;| &radic;| &radic;|pysot| &radic;|official|
| SiamMask    |  &radic; |  &radic; |  &radic;| &radic;| &radic;|pysot| &radic; |official|
| SiamFC++   |  &radic; |  &radic; |  &radic;| &radic;| &radic;|pysot&got10k| &radic; |official|

# Description

- Siamese 

基于孪生网络的简单人脸分类实现，支持训练和测试,

- 2016-ECCV-SiamFC 

添加got10k评估工具，对接口进行优化 可评估 可训练和测试 复现结果略低于论文

- 2018-CVPR-SiamRPN    

API接口进行优化,添加got10k评估工具,可评估,可训练和测试,复现结果略低于论文; 支持测试; 支持训练 支持 评估

- 2018-ECCV-DaSiamRPN  

API接口优化;支持VScode单步调试加pysot评估工具;支持一键评估;不支持训练，支持测试

- 2019-ICCV-UpdateNet   

复现updatenet网络,可测试,训练,评估自己的模型

- 2019-CVPR-SiamRPN++ 

API接口优化，支持VScode单步调试 ,对训练和测试的输入输出接口进行了优化，方便单步调试对代码进行部分注释 修改训练模式，将多机多GPU并行，改成单机多GPU并行，支持单步调试;
The code will come soon！
 

- 2019-CVPR-SiamMask    
- 2020-AAAI-SiamFC++ 


# Model

- SiamRPNVOT.model link: https://pan.baidu.com/s/1V7GMgurufuILhzTSJ4LsYA  password: p4ig   

- SiamRPNOTB.model link: https://pan.baidu.com/s/1mpXaIDcf0HXf3vMccaSriw password: 5xm9   

- SiamRPNBIG.model link: https://pan.baidu.com/s/10v3d3G7BYSRBanIgaL73_Q password: b3b6

# Results 

|   Trackers|       | SiamFC   | DaSiamRPN   |DaSiamRPN | SiamRPN++ |SiamRPN|SiamFC++ |
|:---------: |:-----:|:--------:| :------:    |:------:  |:------:   |:------: |:------:|
|           |       |           |            |         |         |       |        |
|  Backbone |   -    | AlexNet |  AlexNet(OTB/VOT)  |AlexNet(BIG)    | AlexNet(DW)    |AlexNet(UP)    |AlexNet|
|     FPS   | >120fps |   120   |   200         |  160  |   180      |   200       |   160     |
|           |       |           |            |         |         |       |        |
| OTB100    |  AUC   |  0.570    |   0.655   |  0.646   |   0.648  | 0.637  | 0.656    |
|           |  DP   |   0.767    |   0.880   |  0.859   |  0.853   |0.851   |     |
|           |     |           |            |         |         |       |        |
| UAV123    |  AUC  |   0.504    |   0.586   |  0.604   |  0.578   |0.527   |      |
|           |  DP   |    0.702   |   0.796   | 0.801    |  0.769   |0.748   |     |
|           |     |           |            |         |         |       |        |
| UAV20L    |  AUC  |  0.410     |   0.617   |   0.524  |  0.530   |0.454   |      |
|           |  DP   |   0.566    |   0.838   |  0.691   |  0.695   |0.617   |     |
|           |     |           |            |         |         |       |        |
| DTB70     |  AUC  |    0.487   |          |         |   0.588  |        |     |
|           |  DP   |    0.735   |         |         |   0.797  |        |     |
|           |     |           |            |         |         |       |        |
| UAVDT     |  AUC  |           |           |         |  0.566   |       |      |
|           | DP    |           |           |         |  0.793   |       |      |
|           |     |           |            |         |         |       |        |
| VisDrone  | AUC   |           |           |         |  0.572   |       |      |
|           |  DP   |           |           |         |   0.764  |       |      |
|           |     |           |            |         |         |       |        |
| VOT2016   |  A    |   0.538    |  0.61      |  0.625   |  0.618   |0.56    |      |
|           | R     |    0.424   |  0.22      |  0.224   |  0.238   |0.26    |      |
|           | E     |    0.262   |  0.411     |  0.439   |  0.393   | 0.344  |      |
|           |Lost   |    91      |           |  48      |  51      |       |      |
|           |     |           |            |         |         |       |        |
| VOT2018   | A     |     0.501  |   0.56     |  0.586   | 0.576    |0.49    | 0.556   |
|           |  R    |    0.534   |   0.34     |  0.276   |  0.290   |0.46    | 0.183   |
|           | E     |    0.223   |   0.326    | 0.383    |  0.352   |0.244   | 0.400   |
|           | Lost  |   114      |           |  59      |   62       |       |        |

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

   [5] SiamRPN++

   [6] SiamMask
   
   [7] SiamFC++
