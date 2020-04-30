# SiamTrackers

The code will come soon！  https://www.bilibili.com/video/BV1Y64y1T7qs/


![image](https://github.com/HonglinChu/SiamTrackers/tree/master/image/deep.jpg)

孪生网络系列跟踪代码复现汇总
我的贡献

- Siamese 
基于孪生网络的简单人脸分类实现，支持训练和测试

- SiamFC  
添加got10k评估工具，对接口进行优化
可评估
可训练和测试
复现结果略低于论文

- SiamRPN 
API接口进行优化
添加got10k评估工具
可评估
可训练和测试
复现结果略低于论文

- DaSiamRPN
API接口优化，支持VScode单步调试
添加pysot评估工具，支持一键评估
不支持训练，支持测试

- SiamRPN++ 
API接口优化，支持VScode单步调试
- 对训练和测试的输入输出接口进行了优化，方便单步调试
- 对代码进行部分注释
- 修改训练模式，将多机多GPU并行，改成单机多GPU并行，支持单步调试

- SiamMask
 API接口优化
 对代码部分注释
 支持训练和测试

# [Reference]

   [1] SiamFC 

   [2] SiamRPN

   [3] DaSiamRPN

   [5] UpdateNet

   [6] SiamRPN++

   [7] SiamMask
