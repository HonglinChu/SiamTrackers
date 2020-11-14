# Setup

Followings are steps that need to be taken before running any part of our codebase.

## Install requirements

- Linux or MacOS
- Python >= 3.5
- GCC >= 4.9

Fetch code from our repo

```Bash
git clone https://github.com/MegviiDetection/video_analyst.git
cd video_analyst
```

You can choose either using native python (with pip/pip3) or using virtual environment.

```Bash
pip3 install -U -r requirements.txt
```



## Misc issue

### python-tkinter

In case of:

```Python
ModuleNotFoundError: No module named 'tkinter'
```

Please install python3-tk by running:

```Bash
sudo apt-get install python3-tk
```

### pycocotools

The following code ensure the installation of cocoapi in any case.

pip3 install --user pycocotools -i https://pypi.tuna.tsinghua.edu.cn/simple

However, the pycocotools hosted on Pypi may be incompatible with numpy==1.18.0. This issue has been fixed in [this commit](https://github.com/cocodataset/cocoapi/commit/6c3b394c07aed33fd83784a8bf8798059a1e9ae4). If you have numpy==1.18.0 installed (instead numpy==1.16.0 in our requirements.txt), please install pycocotools from the [official repo](https://github.com/cocodataset/cocoapi) on Github.

```Bash
git clone https://github.com/cocodataset/cocoapi
cd cocoapi
make
make install  # may need sudo if it fails
```
