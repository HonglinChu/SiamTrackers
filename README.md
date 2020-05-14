# SiamTrackers

The code will come soon！  https://www.bilibili.com/video/BV1Y64y1T7qs/


![image](https://github.com/HonglinChu/SiamTrackers/tree/master/image/deep.jpg)

孪生网络系列跟踪代码复现汇总
我的贡献
| Trackers     | 调试    | 训练   | 测试  |  评估 | 注释 |
| :--------- | :------: | :------: |:------: |:------: |:------: |
| Siamese     | &radic;|  &radic; | &radic;| &radic;| &radic;|
| SiamFC      |  &radic; |  &radic; |  &radic;| &radic;| &radic;|
| SiamRPN     |  &radic; |  &radic; |  &radic;| &radic;| &radic;|
| DaSiamRPN     |  &radic; |        |  &radic;| &radic;| &radic;|
| UpdateNet   |  &radic; |  &radic; |  &radic;| &radic;| &radic;|
| SiamRPN++    |  &radic; |  &radic; |  &radic;| &radic;| &radic;|
| SiamMask    |  &radic; |  &radic; |  &radic;| &radic;| &radic;|


- Siamese 
基于孪生网络的简单人脸分类实现，支持训练和测试

- SiamFC 添加got10k评估工具，对接口进行优化
可评估
可训练和测试
复现结果略低于论文

- SiamRPN    API接口进行优化,添加got10k评估工具,可评估,可训练和测试,复现结果略低于论文;
  &radic; 测试
  &radic; 训练
  &radic; 评估

- DaSiamRPN  API接口优化;支持VScode单步调试加pysot评估工具;支持一键评估;不支持训练，支持测试 ;
- SiamRPN++  API接口优化，支持VScode单步调试 ,对训练和测试的输入输出接口进行了优化，方便单步调试对代码进行部分注释 修改训练模式，将多机多GPU并行，改成单机多GPU并行，支持单步调试; 

- SiamMask    API接口优化,对代码部分注释,支持训练和测试;

- UpdateNet   复现updatenet网络,可测试,训练,评估自己的模型;

# 训练模型

- SiamRPNVOT.model 链接: https://pan.baidu.com/s/1V7GMgurufuILhzTSJ4LsYA 提取码: p4ig   

- SiamRPNOTB.model 链接: https://pan.baidu.com/s/1mpXaIDcf0HXf3vMccaSriw 提取码: 5xm9   

- SiamRPNBIG.model 链接: https://pan.baidu.com/s/10v3d3G7BYSRBanIgaL73_Q 提取码: b3b6

# 数据集

- UAV123 链接: https://pan.baidu.com/s/1AhNnfjF4fZe14sUFefU3iA 提取码: 2iq4

- VOT2018 链接: https://pan.baidu.com/s/1MOWZ5lcxfF0wsgSuj5g4Yw 提取码: e5eh

- VisDrone2019 链接: https://pan.baidu.com/s/1Y6ubKHuYX65mK_iDVSfKPQ 提取码: yxb6 

- OTB2015 链接: https://pan.baidu.com/s/1ZjKgRMYSHfR_w3Z7iQEkYA 提取码: t5i1

- DTB70 链接: https://pan.baidu.com/s/1kfHrArw0aVhGPSM91WHomw 提取码: e7qm

- ILSVRC2015 VID 链接: https://pan.baidu.com/s/1CXWgpAG4CYpk-WnaUY5mAQ 提取码: uqzj 

- NFS 链接: https://pan.baidu.com/s/1ei54oKNA05iBkoUwXPOB7g 提取码: vng1

- GOT10k 链接: https://pan.baidu.com/s/172oiQPA_Ky2iujcW5Irlow 提取码: uxds

- UAVDT 链接: https://pan.baidu.com/s/1K8oo53mPYCxUFVMXIGLhVA 提取码: keva

- YTB-VOS 链接: https://pan.baidu.com/s/1WMB0q9GJson75QBFVfeH5A 提取码: sf1m 

- YTB-Crop511 （pysot中裁剪好的ytb训练集，siamrpn++和siammask需要用到）链接: https://pan.baidu.com/s/112zLS_02-Z2ouKGbnPlTjw 提取码: ebq1

- TCColor128 链接: https://pan.baidu.com/s/1v4J6zWqZwj8fHi5eo5EJvQ 提取码: 26d4

- DAVIS2017 链接: https://pan.baidu.com/s/1JTsumpnkWotEJQE7KQmh6A 提取码: c9qp

- ytb&vid (siamrpn需要用到) 链接: https://pan.baidu.com/s/1gF8PSZDzw-7EAVrdYHQwsA 提取码: 6vkz



# [Reference]

   [1] SiamFC 

   [2] SiamRPN

   [3] DaSiamRPN

   [4] UpdateNet

   [5] SiamRPN++

   [6] SiamMask
