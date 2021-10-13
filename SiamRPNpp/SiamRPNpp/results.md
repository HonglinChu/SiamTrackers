
## Visual Tracking Baselines

# 在项目根目录下运行
python setup.py build_ext —-inplace
### Short-term Tracking

| <sub>Model</br>（arch+backbone+xcorr）</sub> | <sub>VOT16</br> (EAO/A/R) </sub> | <sub>VOT18</br> (EAO/A/R) </sub> | <sub>VOT19</br> (EAO/A/R) </sub> | <sub>OTB2015</br> (AUC/Prec.) </sub> | <sub>VOT18-LT</br>(F1)</sub> | <sub>Speed</br> (fps) </sub> | <sub>url</sub> |
|:---------------------------------:|:-:|:------------------------:|:--------------------:|:----------------:|:--------------:|:------------:|:-----------:|
|      <sub>siamrpn_alex_dwxcorr</sub>     | <sub>0.393/0.618/0.238</sub> | <sub>0.352/0.576/0.290</sub> | <sub>0.260/0.573/0.547</sub>|             -        |         -        | <sub>180</sub> | [link](https://drive.google.com/open?id=1t62x56Jl7baUzPTo0QrC4jJnwvPZm-2m) |
|    <sub>siamrpn_alex_dwxcorr_otb</sub>   |              -               |             -                | - |<sub>0.666/0.876</sub> |         -        | <sub>180</sub> | [link](https://drive.google.com/open?id=1gCpmR85Qno3C-naR3SLqRNpVfU7VJ2W0) |
|    <sub>siamrpn_r50_l234_dwxcorr</sub>   | <sub>0.464/0.642/0.196</sub> | <sub>0.415/0.601/0.234</sub> | <sub>0.287/0.595/0.467</sub> |            -        |         -        | <sub>35</sub>  | [link](https://drive.google.com/open?id=1Q4-1563iPwV6wSf_lBHDj5CPFiGSlEPG) |
|  <sub>siamrpn_r50_l234_dwxcorr_otb</sub> |              -               |             -                | - |<sub>0.696/0.914</sub> |         -        | <sub>35</sub>  | [link](https://drive.google.com/open?id=1Cx_oHu6o0gNeH7F9zZrgevfAGdyWC4D5) |
|<sub>siamrpn_mobilev2_l234_dwxcorr</sub>| <sub>0.455/0.624/0.214</sub> | <sub>0.410/0.586/0.229</sub> | <sub>0.292/0.580/0.446</sub>|            -        |         -        | <sub>75</sub>  | [link](https://drive.google.com/open?id=1JB94pZTvB1ZByU-qSJn4ZAIfjLWE5EBJ) |
|  <sub>siammask_r50_l3</sub>        | <sub>0.455/0.634/0.219</sub> | <sub>0.423/0.615/0.248</sub> | <sub>0.283/0.597/0.461</sub> |            -        |         -        | <sub>56</sub>  | [link](https://drive.google.com/open?id=1YbPUQVTYw_slAvk_DchvRY-7B6rnSXP9) |
|  <sub>siamrpn_r50_l234_dwxcorr_lt</sub>  |              -               |             -                | - |            -        | <sub>0.629</sub> | <sub>20</sub>  | [link](https://drive.google.com/open?id=1lOOTedwGLbGZ7MAbqJimIcET3ANJd29A) |

The models can also be downloaded from [Baidu Yun](https://pan.baidu.com/s/1GB9-aTtjG57SebraVoBfuQ) Extraction Code: j9yb
