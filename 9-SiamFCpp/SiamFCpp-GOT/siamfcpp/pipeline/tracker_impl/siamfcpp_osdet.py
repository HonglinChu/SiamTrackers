# -*- coding: utf-8 -*

import numpy as np

import torch
import torch.nn as nn

from siamfcpp.pipeline.pipeline_base import TRACK_PIPELINES, PipelineBase
from siamfcpp.pipeline.utils import (cxywh2xywh, cxywh2xyxy, get_crop,
                                         get_subwindow_tracking,
                                         imarray_to_tensor, tensor_to_numpy,
                                         xywh2cxywh, xyxy2cxywh)


# ============================== Tracker definition ============================== #
@TRACK_PIPELINES.register
class SiamFCppOneShotDetector(PipelineBase):
    r"""
    One-shot detector
    Based on Basic SiamFC++ tracker

    Hyper-parameters
    ----------------
        total_stride: int
            stride in backbone
        context_amount: float
            factor controlling the image patch cropping range. Set to 0.5 by convention.
        test_lr: float
            factor controlling target size updating speed
        penalty_k: float
            factor controlling the penalization on target size (scale/ratio) change
        window_influence: float
            factor controlling spatial windowing on scores
        windowing: str
            windowing type. Currently support: "cosine"
        z_size: int
            template image size
        x_size: int
            search image size
        num_conv3x3: int
            number of conv3x3 tiled in head
        min_w: float
            minimum width
        min_h: float
            minimum height
        phase_init: str
            phase name for template feature extraction
        phase_track: str
            phase name for target search

    Hyper-parameters (to be calculated at runtime)
    ----------------------------------------------
    score_size: int
        final feature map
    score_offset: int
        final feature map
    """
    default_hyper_params = dict(
        total_stride=8,
        score_size=17,
        score_offset=87,
        context_amount=0.5,
        test_lr=0.52,
        penalty_k=0.04,
        window_influence=0.21,
        windowing="cosine",
        z_size=127,
        x_size=303,
        num_conv3x3=3,
        min_w=10,
        min_h=10,
        phase_init="feature",
        phase_track="track",
    )

    def __init__(self, *args, **kwargs):
        super(SiamFCppOneShotDetector, self).__init__(*args, **kwargs)
        self.update_params()

        # set underlying model to device
        self.device = torch.device("cpu")
        self.debug = False
        self.set_model(self._model)

    def set_model(self, model):
        """model to be set to pipeline. change device & turn it into eval mode
        
        Parameters
        ----------
        model : ModuleBase
            model to be set to pipeline
        """
        self._model = model.to(self.device)
        self._model.eval()

    def set_device(self, device):
        self.device = device
        self._model = self._model.to(device)

    def update_params(self):
        hps = self._hyper_params
        hps['score_size'] = (
            hps['x_size'] -
            hps['z_size']) // hps['total_stride'] + 1 - hps['num_conv3x3'] * 2
        hps['score_offset'] = (
            hps['x_size'] - 1 -
            (hps['score_size'] - 1) * hps['total_stride']) // 2
        self._hyper_params = hps

    def feature(self, im: np.array, target_pos, target_sz, avg_chans=None):
        """Extract feature

        Parameters
        ----------
        im : np.array
            initial frame
        target_pos : 
            target position (x, y)
        target_sz : [type]
            target size (w, h)
        avg_chans : [type], optional
            channel mean values, (B, G, R), by default None
        
        Returns
        -------
        [type]
            [description]
        """
        if avg_chans is None:
            avg_chans = np.mean(im, axis=(0, 1))

        z_size = self._hyper_params['z_size']
        context_amount = self._hyper_params['context_amount']

        im_z_crop, _ = get_crop(
            im,
            target_pos,
            target_sz,
            z_size,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        phase = self._hyper_params['phase_init']
        with torch.no_grad():
            features = self._model(imarray_to_tensor(im_z_crop).to(self.device),
                                   phase=phase)

        return features, im_z_crop, avg_chans

    def init(self, im, state):
        r"""
        Initialize tracker
            Internal target state representation: self._state['state'] = (target_pos, target_sz)
        :param im: initial frame image
        :param state: bbox, format: xywh
        :return: None
        """
        rect = state  # bbox in xywh format is given for initialization in case of tracking
        box = xywh2cxywh(rect)
        target_pos, target_sz = box[:2], box[2:]

        self._state['im_h'] = im.shape[0]
        self._state['im_w'] = im.shape[1]

        # extract template feature
        features, im_z_crop, avg_chans = self.feature(im, target_pos, target_sz)

        score_size = self._hyper_params['score_size']
        # if self._hyper_params['windowing'] == 'cosine':
        #     window = np.outer(np.hanning(score_size), np.hanning(score_size))
        #     window = window.reshape(-1)
        # elif self._hyper_params['windowing'] == 'uniform':
        #     window = np.ones((score_size, score_size))
        # else:
        #     window = np.ones((score_size, score_size))

        self._state['z_crop'] = im_z_crop
        self._state['avg_chans'] = avg_chans
        self._state['features'] = features
        # self._state['window'] = window
        # self.state['target_pos'] = target_pos
        # self.state['target_sz'] = target_sz
        self._state['state'] = (target_pos, target_sz)

    def track(self,
              im_x,
              target_pos,
              target_sz,
              features,
              update_state=False,
              **kwargs):
        if 'avg_chans' in kwargs:
            avg_chans = kwargs['avg_chans']
        else:
            avg_chans = self._state['avg_chans']

        z_size = self._hyper_params['z_size']
        x_size = self._hyper_params['x_size']
        context_amount = self._hyper_params['context_amount']
        phase_track = self._hyper_params['phase_track']
        im_x_crop, scale_x = get_crop(
            im_x,
            target_pos,
            target_sz,
            z_size,
            x_size=x_size,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        # store crop information
        self._state["crop_info"] = dict(
            target_pos=target_pos,
            target_sz=target_sz,
            scale_x=scale_x,
            avg_chans=avg_chans,
        )
        with torch.no_grad():
            score, box, cls, ctr, *args = self._model(
                imarray_to_tensor(im_x_crop).to(self.device),
                *features,
                phase=phase_track)

        box = tensor_to_numpy(box[0])
        score = tensor_to_numpy(score[0])[:, 0]
        cls = tensor_to_numpy(cls[0])
        ctr = tensor_to_numpy(ctr[0])
        box_wh = xyxy2cxywh(box)

        # score post-processing
        best_pscore_id, pscore, penalty = self._postprocess_score(
            score, box_wh, target_sz, scale_x)
        # box post-processing
        new_target_pos, new_target_sz = self._postprocess_box(
            best_pscore_id, score, box_wh, target_pos, target_sz, scale_x,
            x_size, penalty)

        if self.debug:
            box = self._cvt_box_crop2frame(box_wh, target_pos, x_size, scale_x)

        # restrict new_target_pos & new_target_sz
        # new_target_pos, new_target_sz = self._restrict_box(
        #     new_target_pos, new_target_sz)

        # record basic mid-level info
        self._state['x_crop'] = im_x_crop
        # bbox_pred_in_crop = np.rint(box[best_pscore_id]).astype(np.int)
        bbox_pred_in_crop = box[best_pscore_id]
        self._state['bbox_pred_in_crop'] = bbox_pred_in_crop
        self._state['bbox_pred_in_frame'] = bbox_pred_in_crop

        # record optional mid-level info
        if update_state:
            self._state['score'] = score
            self._state['pscore'] = pscore
            self._state['all_box'] = box
            self._state['cls'] = cls
            self._state['ctr'] = ctr

        return new_target_pos, new_target_sz

    def update(self, im):

        # get track
        target_pos_prior, target_sz_prior = self._state['state']
        features = self._state['features']

        # forward inference to estimate new state
        target_pos, target_sz = self.track(im,
                                           target_pos_prior,
                                           target_sz_prior,
                                           features,
                                           update_state=True)

        # save underlying state
        self._state['state'] = target_pos, target_sz

        # return rect format
        track_rect = cxywh2xywh(np.concatenate([target_pos, target_sz],
                                               axis=-1))
        return track_rect

    # ======== tracking processes ======== #

    def _postprocess_score(self, score, box_wh, target_sz, scale_x):
        r"""
        Perform SiameseRPN-based tracker's post-processing of score
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_sz: previous state (w & h)
        :param scale_x:
        :return:
            best_pscore_id: index of chosen candidate along axis HW
            pscore: (HW, ), penalized score
            penalty: (HW, ), penalty due to scale/ratio change
        """
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        penalty_k = self._hyper_params['penalty_k']
        target_sz_in_crop = target_sz * scale_x
        s_c = change(
            sz(box_wh[:, 2], box_wh[:, 3]) /
            (sz_wh(target_sz_in_crop)))  # scale penalty
        r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) /
                     (box_wh[:, 2] / box_wh[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score

        # ipdb.set_trace()
        # cos window (motion model)
        # window_influence = self._hyper_params['window_influence']
        # pscore = pscore * (
        #     1 - window_influence) + self._state['window'] * window_influence
        best_pscore_id = np.argmax(pscore)

        return best_pscore_id, pscore, penalty

    def _postprocess_box(self, best_pscore_id, score, box_wh, target_pos,
                         target_sz, scale_x, x_size, penalty):
        r"""
        Perform SiameseRPN-based tracker's post-processing of box
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_pos: (2, ) previous position (x & y)
        :param target_sz: (2, ) previous state (w & h)
        :param scale_x: scale of cropped patch of current frame
        :param x_size: size of cropped patch
        :param penalty: scale/ratio change penalty calculated during score post-processing
        :return:
            new_target_pos: (2, ), new target position
            new_target_sz: (2, ), new target size
        """
        pred_in_crop = box_wh[best_pscore_id, :] / np.float32(scale_x)
        # about np.float32(scale_x)
        # attention!, this casting is done implicitly
        # which can influence final EAO heavily given a model & a set of hyper-parameters

        # box post-postprocessing
        # test_lr = self._hyper_params['test_lr']
        # lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
        lr = 1.0  # no EMA smoothing, size directly determined by prediction
        res_x = pred_in_crop[0] + target_pos[0] - (x_size // 2) / scale_x
        res_y = pred_in_crop[1] + target_pos[1] - (x_size // 2) / scale_x
        res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

        new_target_pos = np.array([res_x, res_y])
        new_target_sz = np.array([res_w, res_h])

        return new_target_pos, new_target_sz

    def _restrict_box(self, target_pos, target_sz):
        r"""
        Restrict target position & size
        :param target_pos: (2, ), target position
        :param target_sz: (2, ), target size
        :return:
            target_pos, target_sz
        """
        target_pos[0] = max(0, min(self._state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(self._state['im_h'], target_pos[1]))
        target_sz[0] = max(self._hyper_params['min_w'],
                           min(self._state['im_w'], target_sz[0]))
        target_sz[1] = max(self._hyper_params['min_h'],
                           min(self._state['im_h'], target_sz[1]))

        return target_pos, target_sz

    def _cvt_box_crop2frame(self, box_in_crop, target_pos, scale_x, x_size):
        r"""
        Convert box from cropped patch to original frame
        :param box_in_crop: (4, ), cxywh, box in cropped patch
        :param target_pos: target position
        :param scale_x: scale of cropped patch
        :param x_size: size of cropped patch
        :return:
            box_in_frame: (4, ), cxywh, box in original frame
        """
        x = (box_in_crop[..., 0]) / scale_x + target_pos[0] - (x_size //
                                                               2) / scale_x
        y = (box_in_crop[..., 1]) / scale_x + target_pos[1] - (x_size //
                                                               2) / scale_x
        w = box_in_crop[..., 2] / scale_x
        h = box_in_crop[..., 3] / scale_x
        box_in_frame = np.stack([x, y, w, h], axis=-1)

        return box_in_frame

    def _transform_bbox_from_crop_to_frame(self, bbox_in_crop, crop_info=None):
        r"""Transform bbox from crop to frame, 
            Based on latest detection setting (cropping position / cropping scale)
        
        Arguments
        ---------
        bbox_in_crop:
            bboxes on crop that will be transformed on bboxes on frame
            object able to be reshaped into (-1, 4), xyxy, 
        crop_info: Dict
            dictionary containing cropping information. Transform will be performed based on crop_info
            target_pos: cropping position
            target_sz: target size based on which cropping range was calculated
            scale_x: cropping scale, length on crop / length on frame
        
        Returns
        -------
        np.array
            bboxes on frame. (N, 4)
        """
        if crop_info is None:
            crop_info = self._state["crop_info"]
        target_pos = crop_info["target_pos"]
        target_sz = crop_info["target_sz"]
        scale_x = crop_info["scale_x"]
        x_size = self._hyper_params["x_size"]

        bbox_in_crop = np.array(bbox_in_crop).reshape(-1, 4)
        pred_in_crop = xyxy2cxywh(bbox_in_crop)
        pred_in_crop = pred_in_crop / np.float32(scale_x)

        lr = 1.0  # no EMA smoothing, size directly determined by prediction
        res_x = pred_in_crop[..., 0] + target_pos[0] - (x_size // 2) / scale_x
        res_y = pred_in_crop[..., 1] + target_pos[1] - (x_size // 2) / scale_x
        res_w = target_sz[0] * (1 - lr) + pred_in_crop[..., 2] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[..., 3] * lr

        bbox_in_frame = cxywh2xyxy(
            np.stack([res_x, res_y, res_w, res_h], axis=1))
        return bbox_in_frame
