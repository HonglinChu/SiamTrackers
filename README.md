# SiamTrackers

- [NanoTrackV1](./NanoTrack)
![image](./image/nanotrack_network.png)

# Experiment

- NanoTrack is a lightweight and high speed tracking network which mainly referring to SiamBAN and LightTrack. It is suitable for deployment on embedded or mobile devices. In fact, V1 and V2 can run at **> 200FPS** on Apple M1 CPU.

| Trackers|  Backbone Size(*.onnx)  |Head Size (*.onnx) | FLOPs| Parameters  |  
| :------------: |:------------: | :------------: | :------------: | :------------: |
| NanoTrackV1  | 752K  | 384K  | 75.6M  | 287.9K  |  
|  NanoTrackV2 | 1.0M  | 712K  |84.6M   |334.1K   |
| NanoTrackV3  | 1.4M| 1.1M|115.6M  | 541.4K  | 

- Experiments show that NanoTrack has good performance on tracking datasets.

| Trackers            |   Backbone   | Model Size(*.pth) | VOT2018 EAO | VOT2019 EAO | GOT-10k-Val AO | GOT-10k-Val SR | DTB70 Success | DTB70 Precision |
| :------------------ | :----------: | :------: | :---------: | :---------: | :------------: | :------------: | :-----------: | :-------------: |
| NanoTrackV1         | MobileNetV3  |  2.4MB   |    0.311    |    0.247    |     0.604      |     0.724      |     0.532     |      0.727      |
| NanoTrackV2         | MobileNetV3  |  2.0MB   |    0.352    |    0.270    |     0.680      |     0.817      |     0.584     |      0.753      |
| NanoTrackV3         | MobileNetV3  |  3.4MB   |    0.449    |    0.296    |     0.719      |     0.848      |     0.628     |      0.815      |
| CVPR2021 LightTrack | MobileNetV3  |  7.7MB   |    0.418    |    0.328    |      0.75      |     0.877      |     0.591     |      0.766      |
| WACV2022 SiamTPN    | ShuffleNetV2 |  62.2MB  |    0.191    |    0.209    |     0.728      |     0.865      |     0.572     |      0.728      |
| ICRA2021 SiamAPN    |   AlexNet    | 118.7MB  |    0.248    |    0.235    |     0.622      |     0.708      |     0.585     |      0.786      |
| IROS2021 SiamAPN++  |   AlexNet    |  187MB   |    0.268    |    0.234    |     0.635      |      0.73      |     0.594     |      0.791      |
- For NanoTrackV1, we provide [Android demo](https://github.com/HonglinChu/NanoTrack/tree/master/ncnn_android_nanotrack) and [MacOS demo](https://github.com/HonglinChu/NanoTrack/tree/master/ncnn_macos_nanotrack) based on ncnn inference framework. 

- We also provide [PyTorch code](https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack). It is friendly for training with much lower GPU memory cost than other models. NanoTrackV1 only uses GOT-10k dataset to train, which only takes two hours on RTX3090.

# OpenCV API

- https://docs.opencv.org/4.x/d8/d69/classcv_1_1TrackerNano.html
  
# Dataset

-  **All json files** [BaiduYun](https://pan.baidu.com/s/1RL1kwdP93fdBVOrPc5y0bQ) parrword: xm5w (The json files are provided by [pysot](https://github.com/STVIR/pysot))

## Test
- **OTB2015**  [BaiduYun](https://pan.baidu.com/s/1ZjKgRMYSHfR_w3Z7iQEkYA) password: t5i1

- **VOT2016** [BaiduYun](https://pan.baidu.com/s/1ihsivizX62WhsKBFxwu84w) password: v7vq 

- **VOT2018** [BaiduYun](https://pan.baidu.com/s/1MOWZ5lcxfF0wsgSuj5g4Yw) password: e5eh

- **VOT2019** [BaiduYun](https://pan.baidu.com/s/1HqugngSFKfGl8NGXiRlR_Q) password: p4fi 

- **VOT2020** [BaiduYun](https://pan.baidu.com/s/14KqEVJA10ykO4w4L5gtTjA) password: x93i 

- **UAV123**  [BaiduYun](https://pan.baidu.com/s/1AhNnfjF4fZe14sUFefU3iA) password: 2iq4

- **DTB70**  [BaiduYun](https://pan.baidu.com/s/1kfHrArw0aVhGPSM91WHomw) password: e7qm

- **UAVDT** [BaiduYun](https://pan.baidu.com/s/1K8oo53mPYCxUFVMXIGLhVA) password: keva

- **VisDrone2019** [BaiduYun](https://pan.baidu.com/s/1Y6ubKHuYX65mK_iDVSfKPQ) password: yxb6 

- **TColor128** [BaiduYun](https://pan.baidu.com/s/1v4J6zWqZwj8fHi5eo5EJvQ) password: 26d4

- **NFS** [BaiduYun](https://pan.baidu.com/s/1ei54oKNA05iBkoUwXPOB7g) password: vng1


## Train
- **GOT10k** [BaiduYun](https://pan.baidu.com/s/172oiQPA_Ky2iujcW5Irlow) password: uxds 

- **LaSOT** [BaiduYun](https://pan.baidu.com/s/1A_QWSzNdr4G9CR6rZ7n9Mg) password: ygtx  

- **ILSVRC2015 VID** [BaiDuYun](https://pan.baidu.com/s/1CXWgpAG4CYpk-WnaUY5mAQ) password: uqzj 

- **ILSVRC2015 DET** [BaiDuYun](https://pan.baidu.com/s/1t2IgiYGRu-sdfOYwfeemaQ) password: 6fu7 

- **YTB-Crop511** [BaiduYun](https://pan.baidu.com/s/112zLS_02-Z2ouKGbnPlTjw) password: ebq1 

- **COCO** [BaiduYun](https://pan.baidu.com/s/17AMGS2ezLVd8wFI2NbJQ3w) password: ggya

- **TrackingNet** [BaiduYun](https://pan.baidu.com/s/1PXSRAqcw-KMfBIJYUtI4Aw) password: nkb9  (Note that this link is provided by SiamFCpp author)

## Mask 

- **YTB-VOS** [BaiduYun](https://pan.baidu.com/s/1WMB0q9GJson75QBFVfeH5A) password: sf1m  

- **DAVIS2017** [BaiduYun](https://pan.baidu.com/s/1JTsumpnkWotEJQE7KQmh6A) password: c9qp 


# Toolkit
### Matlab version

- **OTB2013/2015**  [Github](https://github.com/HonglinChu/visual_tracker_benchmark)

- **UAVDT** [BaiduYun](https://pan.baidu.com/s/1NdpaWZxv5hGfKnIqJznWYA) password: ehit

- **VOT2016-toolkit** [BaiduYun](https://pan.baidu.com/s/1RbmH-fVExBpHv3TgjHzYGg) password: 272e

- **VOT2018-toolkit** [BaiduYun](https://pan.baidu.com/s/1crv4XSFK6zQp2LiZtJcrPw) password: xpkb 

### Python version

- **pysot-toolkit**： OTB, VOT, UAV, NfS, LaSOT are supported.[BaiduYun](https://pan.baidu.com/s/1H2Hc4VXsWahgNjDZJP8jaA) password: 2t2q

- **got10k-toolkit**：GOT-10k, OTB, VOT, UAV, TColor, DTB, NfS, LaSOT and TrackingNet are supported.[BaiduYun](https://pan.baidu.com/s/1OS80_OPtZoo0ZFKzfCOFzg) password: vsar

# Papers

[BaiduYun](https://pan.baidu.com/s/1nyXMesdAUHzdSQkM88AvWQ) password: fukj

# Reference

```
[1] SiamFC

Bertinetto L, Valmadre J, Henriques J F, et al. Fully-convolutional siamese networks for object tracking.European conference on computer vision. Springer, Cham, 2016: 850-865.
   
[2] SiamRPN

Li B, Yan J, Wu W, et al. High performance visual tracking with siamese region proposal network.Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 8971-8980.

[3] DaSiamRPN

Zhu Z, Wang Q, Li B, et al. Distractor-aware siamese networks for visual object tracking.Proceedings of the European Conference on Computer Vision (ECCV). 2018: 101-117.

[4] UpdateNet

Zhang L, Gonzalez-Garcia A, Weijer J, et al. Learning the Model Update for Siamese Trackers. Proceedings of the IEEE International Conference on Computer Vision. 2019: 4010-4019.
   
[5] SiamDW

Zhang Z, Peng H. Deeper and wider siamese networks for real-time visual tracking. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 4591-4600.

[6] SiamRPNpp

Li B, Wu W, Wang Q, et al. SiamRPNpp: Evolution of siamese visual tracking with very deep networks.Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 4282-4291.

[7] SiamMask

Wang Q, Zhang L, Bertinetto L, et al. Fast online object tracking and segmentation: A unifying approach. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 1328-1338.
   
[8] SiamFCpp

Xu Y, Wang Z, Li Z, et al. SiamFCpp: Towards Robust and Accurate Visual Tracking with Target Estimation Guidelines. AAAI, 2020.

[9] SiamCAR
Guo D ,  Wang J ,  Cui Y , et al. SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.2020.

[10] SiamBAN
Chen Z, Zhong B, Li G, et al. Siamese box adaptive network for visual tracking[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 6668-6677.

[11] TrTr 
Zhao M, Okada K, Inaba M. TrTr: Visual Tracking with Transformer[J]. arXiv preprint arXiv:2105.03817, 2021.

[12] LightTrack 
Yan B, Peng H, Wu K, et al. Lighttrack: Finding lightweight neural networks for object tracking via one-shot architecture search[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 15180-15189.

```
