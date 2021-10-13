## Pipeline

Basic sot tracker pipelines:
* _videoanalyst/pipeline/tracker_impl/siamfcpp_track.py_

### Basic API

* init(im, state)
* update(im[, rect])

P.S. state: bounding box (format: xywh)

### Internal State Description

Following internal states (e.g. intermediate scores/bboxes) are updated every time the _update_ method is called. In cases where some of them are needed in your application, please collect them before the next call.

* self._state['x_crop']
  * cropped resized search image patch, shape=(x_size, x_size)
  * shape=(x_size, x_size, 3)
* self._state['bbox_pred_in_crop']
  * bbox on _x_crop_, format: xyxy
* self._state['score']
  * original score 
  * shape=(score_size * score_size, )
* self._state['pscore'] = pscore[best_pscore_id]
  * maximum penalized score
  * scalar
* self._state['all_box'] = box
  * all the predicted box on _x_crop_
  * shape=(score_size * score_size, 4), format: xyxy
* self._state['cls'] = cls
  * all the classification score
  * shape=(score_size * score_size, )
* self._state['ctr'] = ctr
  * all the quality asseessment score
  * shape=(score_size * score_size, )

### Hyper-parameter

Description for some hyper-parameter configuration (model/pipeline).

e.g. _experiments/siamfcpp/test/got10k/siamfcpp_googlenet-got.yaml_

* model
  * task_head
    * DenseboxHead
      * total_stride: downsample ratio (8 for SOT)
      * score_size: size of final dense prediction size
      * x_size: search image input size
      * num_conv3x3: number of conv3x3 in head. Note that each conv3x3 brings a shrinkage of 2 pixel in _score_size_
      * head_conv_bn: List[bool], control BN config in head
* pipeline
  * SiamFCppTracker:
    * test_lr: control the bbox smoothing factor, larger test_lr -> less smoothing
    * window_influence: control the penalization on spatial motion    
    * penalty_k: control the penalization on bbox prediction change
    * x_size: keep as the same in _model_
    * num_conv3x3: keep as the same in _model_


