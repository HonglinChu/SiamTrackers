# -*- coding: utf-8 -*

from copy import deepcopy

import cv2
import numpy as np
from loguru import logger

import torch
import torch.nn as nn

from siamfcpp.pipeline.pipeline_base import VOS_PIPELINES, PipelineBase
from siamfcpp.pipeline.utils import (cxywh2xywh, get_crop,
                                         get_subwindow_tracking,
                                         imarray_to_tensor, tensor_to_numpy,
                                         xywh2cxywh, xyxy2cxywh)


# ============================== Tracker definition ============================== #
@VOS_PIPELINES.register
class StateAwareTracker(PipelineBase):
    r"""
    Basic State-Aware Tracker for vos

    Hyper-parameters
    ----------------
        z_size: int
            template image size
        save_patch: bool
            save and visualize the predicted mask for saliency image patch
        mask_pred_thresh: float
            threshold to binarize predicted mask for final decision
        mask_filter_thresh: float
            threshold to binarize predicted mask for filter the patch of global modeling loop
        GMP_image_size: int
            image size of the input of global modeling loop
        saliency_image_size: int
            image size of saliency image
        saliency_image_field: int
            corresponding fields of saliency image
        cropping_strategy: bool
            use cropping strategy or not
        state_score_thresh: float
            threshhold for state score
        global_modeling: bool
            use global modeling loop or not
        seg_ema_u: float
            hyper-parameter u for global feature updating
        seg_ema_s: float
            hyper-parameter s for global feature updating
        track_failed_score_th: float
            if tracker score < th, then the mask will be ignored
        update_global_fea_th: float
            if state score > th, the global fea will be updated 

    """
    default_hyper_params = dict(
        z_size=127,
        save_patch=True,
        mask_pred_thresh=0.4,
        mask_filter_thresh=0.5,
        GMP_image_size=129,
        saliency_image_size=257,
        saliency_image_field=129,
        cropping_strategy=True,
        state_score_thresh=0.9,
        global_modeling=True,
        seg_ema_u=0.5,
        seg_ema_s=0.5,
        context_amount=0.5,
        mask_rect_lr=1.0,
        track_failed_score_th=0.0,
        update_global_fea_th=0.0,
    )

    def __init__(self, segmenter, tracker):

        self._hyper_params = deepcopy(
            self.default_hyper_params)  # mapping-like object
        self._state = dict()  # pipeline state
        self._segmenter = segmenter
        self._tracker = tracker

        self.update_params()

        # set underlying model to device
        self.device = torch.device("cpu")
        self.debug = False
        self.set_model(self._segmenter, self._tracker)

    def set_model(self, segmenter, tracker):
        """model to be set to pipeline. change device & turn it into eval mode
        
        Parameters
        ----------
        model : ModuleBase
            model to be set to pipeline
        """
        self._segmenter = segmenter.to(self.device)
        self._segmenter.eval()
        self._tracker.set_device(self.device)

    def set_device(self, device):
        self.device = device
        self._segmenter = self._segmenter.to(device)
        self._tracker.set_device(self.device)

    def init(self, im, state, init_mask):
        """
        initialize the whole pipeline :
        tracker init => global modeling loop init

        :param im: init frame
        :param state: bbox in xywh format
        :param init_mask: binary mask of target object in shape (h,w)
        """

        #========== SiamFC++ init ==============
        self._tracker.init(im, state)
        avg_chans = self._tracker.get_avg_chans()
        self._state['avg_chans'] = avg_chans

        rect = state  # bbox in xywh format is given for initialization in case of tracking
        box = xywh2cxywh(rect)
        target_pos, target_sz = box[:2], box[2:]
        self._state['state'] = (target_pos, target_sz)
        self._state['im_h'] = im.shape[0]
        self._state['im_w'] = im.shape[1]

        # ========== Global Modeling Loop init ==============
        init_image, _ = get_crop(
            im,
            target_pos,
            target_sz,
            z_size=self._hyper_params["z_size"],
            x_size=self._hyper_params["GMP_image_size"],
            avg_chans=avg_chans,
            context_amount=self._hyper_params["context_amount"],
            func_get_subwindow=get_subwindow_tracking,
        )
        init_mask_c3 = np.stack([init_mask, init_mask, init_mask],
                                -1).astype(np.uint8)
        init_mask_crop_c3, _ = get_crop(
            init_mask_c3,
            target_pos,
            target_sz,
            z_size=self._hyper_params["z_size"],
            x_size=self._hyper_params["GMP_image_size"],
            avg_chans=avg_chans * 0,
            context_amount=self._hyper_params["context_amount"],
            func_get_subwindow=get_subwindow_tracking,
        )
        init_mask_crop = init_mask_crop_c3[:, :, 0]
        init_mask_crop = (init_mask_crop >
                          self._hyper_params['mask_filter_thresh']).astype(
                              np.uint8)
        init_mask_crop = np.expand_dims(init_mask_crop,
                                        axis=-1)  #shape: (129,129,1)
        filtered_image = init_mask_crop * init_image
        self._state['filtered_image'] = filtered_image  #shape: (129,129,3)

        with torch.no_grad():
            deep_feature = self._segmenter(imarray_to_tensor(filtered_image).to(
                self.device),
                                           phase='global_feature')[0]

        self._state['seg_init_feature'] = deep_feature  #shape : (1,256,5,5)
        self._state['seg_global_feature'] = deep_feature
        self._state['gml_feature'] = deep_feature
        self._state['conf_score'] = 1

    def global_modeling(self):
        """
        always runs after seg4vos, takes newly predicted filtered image,
        extracts high-level feature and updates the global feature based on confidence score

        """
        filtered_image = self._state['filtered_image']  # shape: (129,129,3)
        with torch.no_grad():
            deep_feature = self._segmenter(imarray_to_tensor(filtered_image).to(
                self.device),
                                           phase='global_feature')[0]

        seg_global_feature = self._state['seg_global_feature']
        seg_init_feature = self._state['seg_init_feature']
        u = self._hyper_params['seg_ema_u']
        s = self._hyper_params['seg_ema_s']
        conf_score = self._state['conf_score']

        u = u * conf_score
        seg_global_feature = seg_global_feature * (1 - u) + deep_feature * u
        gml_feature = seg_global_feature * s + seg_init_feature * (1 - s)

        self._state['seg_global_feature'] = seg_global_feature
        self._state['gml_feature'] = gml_feature

    def joint_segmentation(self, im_x, target_pos, target_sz, corr_feature,
                           gml_feature, **kwargs):
        r"""
        segment the current frame for VOS
        crop image => segmentation =>  params updation

        :param im_x: current image
        :param target_pos: target position (x, y)
        :param target_sz: target size (w, h)
        :param corr_feature: correlated feature produced by siamese encoder
        :param gml_feature: global feature produced by gloabl modeling loop
        :return: pred_mask  mask prediction in the patch of saliency image
        :return: pred_mask_b binary mask prediction in the patch of saliency image
        """

        if 'avg_chans' in kwargs:
            avg_chans = kwargs['avg_chans']
        else:
            avg_chans = self._state['avg_chans']

        # crop image for saliency encoder
        saliency_image, scale_seg = get_crop(
            im_x,
            target_pos,
            target_sz,
            z_size=self._hyper_params["z_size"],
            output_size=self._hyper_params["saliency_image_size"],
            x_size=self._hyper_params["saliency_image_field"],
            avg_chans=avg_chans,
            context_amount=self._hyper_params["context_amount"],
            func_get_subwindow=get_subwindow_tracking,
        )
        self._state["scale_x"] = scale_seg
        # mask prediction
        pred_mask = self._segmenter(imarray_to_tensor(saliency_image).to(
            self.device),
                                    corr_feature,
                                    gml_feature,
                                    phase='segment')[0]  #tensor(1,1,257,257)

        pred_mask = tensor_to_numpy(pred_mask[0]).transpose(
            (1, 2, 0))  #np (257,257,1)

        # post processing
        mask_filter = (pred_mask >
                       self._hyper_params['mask_filter_thresh']).astype(
                           np.uint8)
        pred_mask_b = (pred_mask >
                       self._hyper_params['mask_pred_thresh']).astype(np.uint8)

        if self._hyper_params['save_patch']:
            mask_red = np.zeros_like(saliency_image)
            mask_red[:, :, 0] = mask_filter[:, :, 0] * 255
            masked_image = saliency_image * 0.5 + mask_red * 0.5
            self._state['patch_prediction'] = masked_image

        filtered_image = saliency_image * mask_filter
        filtered_image = cv2.resize(filtered_image,
                                    (self._hyper_params["GMP_image_size"],
                                     self._hyper_params["GMP_image_size"]))
        self._state['filtered_image'] = filtered_image

        if pred_mask_b.sum() > 0:
            conf_score = (pred_mask * pred_mask_b).sum() / pred_mask_b.sum()
        else:
            conf_score = 0
        self._state['conf_score'] = conf_score
        mask_in_full_image = self._mask_back(
            pred_mask,
            size=self._hyper_params["saliency_image_size"],
            region=self._hyper_params["saliency_image_field"])
        self._state['mask_in_full_image'] = mask_in_full_image
        if self._tracker.get_track_score(
        ) < self._hyper_params["track_failed_score_th"]:
            self._state['mask_in_full_image'] *= 0
        return pred_mask, pred_mask_b

    def get_global_box_from_masks(self, cnts):
        boxes = np.zeros((len(cnts), 4))
        for i, cnt in enumerate(cnts):
            rect = cv2.boundingRect(cnt.reshape(-1, 2))
            boxes[i] = rect
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
        global_box = [
            np.amin(boxes[:, 0]),
            np.amin(boxes[:, 1]),
            np.amax(boxes[:, 2]),
            np.amax(boxes[:, 3])
        ]
        global_box = np.array(global_box)
        global_box[2:] = global_box[2:] - global_box[:2]
        return global_box

    def cropping_strategy(self, p_mask_b, track_pos=None, track_size=None):
        r"""
        swithes the bbox prediction strategy based on the estimation of predicted mask.
        returns newly predicted target position and size

        :param p_mask_b: binary mask prediction in the patch of saliency image
        :param target_pos: target position (x, y)
        :param target_sz: target size (w, h)
        :return: new_target_pos, new_target_sz
        """

        new_target_pos, new_target_sz = self._state["state"]
        conf_score = self._state['conf_score']
        self._state["track_score"] = self._tracker.get_track_score()
        new_target_pos, new_target_sz = track_pos, track_size

        if conf_score > self._hyper_params['state_score_thresh']:
            contours, _ = cv2.findContours(p_mask_b, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            cnt_area = [cv2.contourArea(cnt) for cnt in contours]

            if len(contours) != 0 and np.max(cnt_area) > 10:
                pbox = self.get_global_box_from_masks(contours)
                rect_full, cxywh_full = self._coord_back(
                    pbox,
                    size=self._hyper_params["saliency_image_size"],
                    region=self._hyper_params["saliency_image_field"])
                mask_pos, mask_sz = cxywh_full[:2], cxywh_full[2:]

                conc_score = np.max(cnt_area) / sum(cnt_area)
                state_score = conf_score * conc_score
                self._state['conc_score'] = conc_score
                self._state['state_score'] = state_score

                if state_score > self._hyper_params['state_score_thresh']:
                    new_target_pos = mask_pos
                    lr = self._hyper_params["mask_rect_lr"]
                    new_target_sz = self._state["state"][1] * (
                        1 - lr) + mask_sz * lr
                else:
                    if self._state["track_score"] > self._hyper_params[
                            "track_failed_score_th"]:
                        new_target_pos, new_target_sz = track_pos, track_size

                self._state['mask_rect'] = rect_full

            else:  # empty mask
                self._state['mask_rect'] = [-1, -1, -1, -1]
                self._state['state_score'] = 0

        else:  # empty mask
            self._state['mask_rect'] = [-1, -1, -1, -1]
            self._state['state_score'] = 0

        return new_target_pos, new_target_sz

    def update(self, im):

        # get track
        target_pos_prior, target_sz_prior = self._state['state']
        self._state['current_state'] = deepcopy(self._state['state'])

        # forward inference to estimate new state
        # tracking for VOS returns regressed box and correlation feature
        self._tracker.set_state(self._state["state"])
        target_pos_track, target_sz_track, corr_feature = self._tracker.update(
            im)

        # segmentation returnd predicted masks
        gml_feature = self._state['gml_feature']
        pred_mask, pred_mask_b = self.joint_segmentation(
            im, target_pos_prior, target_sz_prior, corr_feature, gml_feature)

        # cropping strategy loop swtiches the coordinate prediction method
        if self._hyper_params['cropping_strategy']:
            target_pos, target_sz = self.cropping_strategy(
                pred_mask_b, target_pos_track, target_sz_track)
        else:
            target_pos, target_sz = target_pos_track, target_sz_track

        # global modeling loop updates global feature for next frame's segmentation
        if self._hyper_params['global_modeling']:
            if self._state["state_score"] > self._hyper_params[
                    "update_global_fea_th"]:
                self.global_modeling()
        # save underlying state
        self._state['state'] = target_pos, target_sz
        track_rect = cxywh2xywh(
            np.concatenate([target_pos_track, target_sz_track], axis=-1))
        self._state['track_box'] = track_rect
        return self._state['mask_in_full_image']

    # ======== vos processes ======== #

    def _mask_back(self, p_mask, size=257, region=129):
        """
        Warp the predicted mask from cropped patch back to original image.

        :param p_mask: predicted_mask (h,w)
        :param size: image size of cropped patch
        :param region: region size with template = 127
        :return: mask in full image
        """

        target_pos, target_sz = self._state['current_state']
        scale_x = self._state['scale_x']

        zoom_ratio = size / region
        scale = scale_x * zoom_ratio
        cx_f, cy_f = target_pos[0], target_pos[1]
        cx_c, cy_c = (size - 1) / 2, (size - 1) / 2

        a, b = 1 / (scale), 1 / (scale)

        c = cx_f - a * cx_c
        d = cy_f - b * cy_c

        mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
        mask_in_full_image = cv2.warpAffine(
            p_mask,
            mapping, (self._state['im_w'], self._state['im_h']),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)
        return mask_in_full_image

    def _coord_back(self, rect, size=257, region=129):
        """
        Warp the predicted coordinates from cropped patch back to original image.

        :param rect: rect with coords in cropped patch
        :param size: image size of cropped patch
        :param region: region size with template = 127
        :return: rect(xywh) and cxywh in full image
        """

        target_pos, _ = self._state['current_state']
        scale_x = self._state['scale_x']

        zoom_ratio = size / region
        scale = scale_x * zoom_ratio
        cx_f, cy_f = target_pos[0], target_pos[1]
        cx_c, cy_c = (size - 1) / 2, (size - 1) / 2

        a, b = 1 / (scale), 1 / (scale)

        c = cx_f - a * cx_c
        d = cy_f - b * cy_c

        x1, y1, w, h = rect[0], rect[1], rect[2], rect[3]

        x1_t = a * x1 + c
        y1_t = b * y1 + d
        w_t, h_t = w * a, h * b
        return [x1_t, y1_t, w_t, h_t], xywh2cxywh([x1_t, y1_t, w_t, h_t])
