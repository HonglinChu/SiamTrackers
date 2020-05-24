# SiamTrackers

The code will come soon！  https://www.bilibili.com/video/BV1Y64y1T7qs/ （Description）

![image](https://github.com/HonglinChu/SiamTrackers/tree/master/image/deep.jpg)


| Trackers     | Deubug   | Train   | Test  |  Evaluation | Comment | Toolkit | GPU | 
| :--------- | :--------: | :------: |:------: |:------: |:------: |:------: |  :------: | 
| Siamese     | &radic;|  &radic; | &radic;| &radic;   | &radic;|   | &radic;|
| 2016-ECCV-SiamFC      |  &radic; |  &radic; |  &radic;| &radic;| &radic;|got10k|&radic; |
| 2018-CVPR-SiamRPN     |  &radic; |  &radic; |  &radic;| &radic;| &radic;|got10k|  &radic;|
| 2018-ECCV-DaSiamRPN     |  &radic; |       |  &radic;| &radic;| &radic;|pysot| &radic; |
| 2019-ICCV-UpdateNet   |  &radic; |  &radic; |  &radic;| &radic;| &radic;|pysot| &radic; |
| 2019-CVPR-SiamRPN++    |  &radic; |  &radic; |  &radic;| &radic;| &radic;|pysot| &radic;|
| 2019-CVPR-SiamMask    |  &radic; |  &radic; |  &radic;| &radic;| &radic;|pysot| &radic; |
| 2020-AAAI-SiamFC++    |  &radic; |  &radic; |  &radic;| &radic;| &radic;|pysot| &radic; |

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

- 2019-CVPR-SiamMask    

- 2020-AAAI-SiamFC++   

# Model

- SiamRPNVOT.model link: https://pan.baidu.com/s/1V7GMgurufuILhzTSJ4LsYA  password: p4ig   

- SiamRPNOTB.model link: https://pan.baidu.com/s/1mpXaIDcf0HXf3vMccaSriw password: 5xm9   

- SiamRPNBIG.model link: https://pan.baidu.com/s/10v3d3G7BYSRBanIgaL73_Q password: b3b6

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
