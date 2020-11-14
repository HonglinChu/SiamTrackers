# -*- coding: utf-8 -*
import copy

import numpy as np

import torch
import torch.nn as nn

from siamfcpp.pipeline.pipeline_base import TRACK_PIPELINES, PipelineBase
from siamfcpp.pipeline.tracker_impl.siamfcpp_track import SiamFCppTracker
from siamfcpp.pipeline.utils import (cxywh2xywh, get_crop,
                                         get_subwindow_tracking,
                                         imarray_to_tensor, tensor_to_numpy,
                                         xywh2cxywh, xyxy2cxywh)

eps = 1e-7


# ============================== Tracker definition ============================== #
@TRACK_PIPELINES.register
class SiamFCppMultiTempTracker(SiamFCppTracker):
    r"""
    Multi-template SiamFC++ tracker.
    Currently using naive short-time template averaging strategy

    Hyper-parameters
    ----------------
    mem_step: int
        short-time template sampling frequency (e.g. one sampling every mem_step frames )
    mem_len: int
        template memory length
    mem_coef: str
        short-time memory coefficient
        e.g. final_score = (1-mem_coef * init_score + mem_coef * mean(st_mem_score[])
    mem_sink_idx: str
        template index to dequeue
    """
    extra_hyper_params = dict(
        mem_step=5,
        mem_len=5,
        mem_coef=0.7,
        mem_sink_idx=1,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_params()

    def init(self, im, state):
        super().init(im, state)
        self._state['frame_cnt'] = 0
        self._state['z_crop'] = [self._state['z_crop']
                                 ] * self._hyper_params['mem_len']
        self._state['features'] = [self._state['features']
                                   ] * self._hyper_params['mem_len']

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

        # process batch of templates
        score_list = []
        box_list = []
        cls_list = []
        ctr_list = []
        fms_x = None
        for ith in range(self._hyper_params['mem_len']):
            if fms_x is None:
                with torch.no_grad():
                    score, box, cls, ctr, extra = self._model(
                        imarray_to_tensor(im_x_crop).to(self.device),
                        *(features[ith]),
                        phase=phase_track)
                fms_x = [extra['c_x'], extra['r_x']]
            else:
                with torch.no_grad():
                    score, box, cls, ctr, extra = self._model(*(features[ith]),
                                                              fms_x[0],
                                                              fms_x[1],
                                                              phase=phase_track)
            box = tensor_to_numpy(box[0])
            score = tensor_to_numpy(score[0])[:, 0]
            cls = tensor_to_numpy(cls[0])[:, 0]
            ctr = tensor_to_numpy(ctr[0])[:, 0]
            # append to list
            box_list.append(box)
            score_list.append(score)
            cls_list.append(cls)
            ctr_list.append(ctr)

        # fusion
        if self._hyper_params['mem_len'] > 1:
            score = score_list[0] * (1-self._hyper_params['mem_coef']) + \
                    np.stack(score_list[1:], axis=0).mean(axis=0) * self._hyper_params['mem_coef']
        else:
            # single template
            score = score_list[0]
        box = box_list[0]
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
        new_target_pos, new_target_sz = self._restrict_box(
            new_target_pos, new_target_sz)

        # record basic mid-level info
        self._state['x_crop'] = im_x_crop
        bbox_pred_in_crop = np.rint(box[best_pscore_id]).astype(np.int)
        self._state['bbox_pred_in_crop'] = bbox_pred_in_crop
        # record optional mid-level info
        if update_state:
            self._state['score'] = score
            self._state['pscore'] = pscore
            self._state['all_box'] = box
            self._state['cls'] = cls
            self._state['ctr'] = ctr

        return new_target_pos, new_target_sz


'''

    def update(self, im):
        self._state['frame_cnt'] = self._state['frame_cnt'] + 1

        track_rect = super().update(im)

        if ((self._state['frame_cnt']%self._hyper_params['mem_step']) == 0) and \
                (self._hyper_params['mem_len'] > 1):
            target_pos, target_sz = self._state['state']
            features_curr, im_z_crop_curr, _ = self.feature(
                im, target_pos, target_sz, avg_chans=self._state['avg_chans'])

            self._state['z_crop'].pop(self._hyper_params['mem_sink_idx'])
            self._state['z_crop'].append(im_z_crop_curr)
            self._state['features'].pop(self._hyper_params['mem_sink_idx'])
            self._state['features'].append(features_curr)

        return track_rect
'''

SiamFCppMultiTempTracker.default_hyper_params = copy.deepcopy(
    SiamFCppMultiTempTracker.default_hyper_params)
SiamFCppMultiTempTracker.default_hyper_params.update(
    SiamFCppMultiTempTracker.extra_hyper_params)
