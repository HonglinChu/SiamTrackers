# NanoTrack 

![network](./image/nanotrack_network.png)

- NanoTrack is a FCOS-style one-stage anchor-free object trakcing model which mainly referring to SiamBAN and LightTrack

- NanoTrack is a simple, lightweight and high speed tracking network, and it is very suitable for deployment on embedded or mobile devices. We provide [Android demo](https://github.com/HonglinChu/NanoTrack/ncnn_android_nanotrack) and [MacOS demo](https://github.com/HonglinChu/NanoTrack/ncnn_macos_nanotrack) based on ncnn inference framework. 
![macs](./image/calculate.png) 

- We provide [PyTorch code](https://github.com/HonglinChu/SiamTrackers/NanoTrack). It is very friendly for training with much lower GPU memory cost than other models. We only use GOT-10k as tranining set, and it only takes two hours to train on GPU3090
    ```
    NanoTrack  VOT2018 EAO 0.301

    LightTrack VOT2018 EAO 0.42x
    ```

# NanoTrack for MacOS 

[PC demo](https://www.bilibili.com/video/BV1HY4y1q7B6?spm_id_from=333.999.0.0)


- 1. Modify your own CMakeList.txt

- 2. Build (Apple M1 CPU) 

    ```
    $ sh make_macos_arm64.sh 
    ```

# NanoTrack for Android

[Android demo](https://www.bilibili.com/video/BV1eY4y1p7Cb?spm_id_from=333.999.0.0)

- 1. Modify your own CMakeList.txt

- 2. [Download](https://pan.baidu.com/s/1Yu1bpSKG-02fC5qekWXcLw)(password: 6cdd) OpenCV and NCNN libraries for Android 

# Reference  

https://github.com/Tencent/ncnn

https://github.com/hqucv/siamban

https://github.com/researchmm/LightTrack

https://github.com/Z-Xiong/LightTrack-ncnn

https://github.com/FeiGeChuanShu/ncnn_Android_LightTrack
