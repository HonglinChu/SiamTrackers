# MODEL_ZOO

## Download links

Models & Raw results:

* [Google Drive](https://drive.google.com/open?id=1UXshq4k9WKx4hNkdpOagJLXPR57ZkBkg)
* [baidu yun](https://pan.baidu.com/s/1uZ26iZyVJm50dJ3GoLCQ9w), code: rcsn

## Models

### Davis2017 val

VOS test configuration directory: _experiments/sat/test/

| MODEL | Pipeline | Dataset | J&F-Mean | J-Mean | J-Recall| J-Decay| F-Mean | F-Recall| F-Decaly | FPS@GTX2080Ti |Config. Filename|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SAT-Res50|StateAwareTracker| DAVIS2017_val | 0.712  |0.676  |0.781 | 0.144 | 0.748  | 0.854 | 0.18|~35|sat_res50-davis17.yaml 
| SAT-Res18|StateAwareTracker| DAVIS2017_val | 0.696  |0.665  |0.761 | 0.162 | 0.727  | 0.820 | 0.191|~50|sat_res18-davis17.yaml 
| SAT-Res50|StateAwareTracker| DAVIS2016_val | 0.821 |0.818  |0.938 | 0.022 | 0.824  | 0.931 | 0.05|~35|sat_res50-davis16.yaml 

__Nota__:

[1]  We reimplement SAT with pytorch. The J&F is slightly lower than what we trained with internal framework. We will continue refining the training code.