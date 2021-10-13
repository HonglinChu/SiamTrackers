# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F

#----------------------------------------------------SiamMask--------------------------------------------------#
from torch.autograd import Variable
import cv2
#----------------------------------------------------SiamMask--------------------------------------------------#

def get_cls_loss(pred, label, select):
    if len(select.size()) == 0: 
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)

def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5

def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)

#----------------------------------------------------SiamMask--------------------------------------------------#
def select_mask_logistic_loss(p_m, mask, weight, o_sz=63, g_sz=127, padding=32):
    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    if pos.nelement() == 0: return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0

    # print("p_m1:", p_m.shape)  # p_m1: torch.Size([16, 3969, 25, 25])
    if len(p_m.shape) == 4:
        # print("p_m1:", p_m.shape)  # p_m1: torch.Size([16, 3969, 25, 25])
        p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
        p_m = torch.index_select(p_m, 0, pos)
        p_m = torch.nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
        p_m = p_m.view(-1, g_sz * g_sz)
        # print("p_m:", p_m.shape)  # p_m: torch.Size([47, 16129])
    else:
        # print("p_m1:", p_m.shape)  # p_m1: torch.Size([576, 16129])  p_m1: torch.Size([9, 16129])
        p_m = torch.index_select(p_m, 0, pos)
        # print("p_m:", p_m.shape)  # p_m: torch.Size([339, 16129])

    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=padding, stride=8)
    # print("mask_uf:", mask_uf.shape)  # mask_uf: torch.Size([16, 16129, 625])
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)
    # print("mask_uf:", mask_uf.shape)  # mask_uf: torch.Size([10000, 16129])
    mask_uf = torch.index_select(mask_uf, 0, pos)
    # print("mask_uf:", mask_uf.shape)  # mask_uf: torch.Size([47, 16129])
    loss = F.soft_margin_loss(p_m, mask_uf)
    iou_m, iou_5, iou_7 = iou_measure(p_m, mask_uf)
    # print("ioum:", iou_m)
    return loss, iou_m, iou_5, iou_7 

def iou_measure(pred, label):
    pred = pred.ge(0).type(torch.uint8)
    mask_sum = pred.eq(1).type(torch.uint8).add(label.eq(1).type(torch.uint8))

    # pred = pred.ge(0)
    # mask_sum = pred.eq(1).add(label.eq(1))

    intxn = torch.sum(mask_sum == 2, dim=1).float()
    union = torch.sum(mask_sum > 0, dim=1).float()
    iou = intxn/union
    return torch.mean(iou), (torch.sum(iou > 0.5).float()/iou.shape[0]), (torch.sum(iou > 0.7).float()/iou.shape[0])
    
# def iou_measure(pred, label):
#     pred = pred.ge(0)
#     mask_sum = pred.eq(1).add(label.eq(1))
#     # pred = pred.ge(0).type(torch.uint8)
#     # mask_sum = (pred.eq(1).type(torch.uint8)).add(label.eq(1).type(torch.uint8))
#     # print('mask_sum :', mask_sum.shape)  # mask_sum : torch.Size([xx, 16129])

#     # preds = pred.eq(1)
#     # intxp = torch.sum(preds == 1, dim=1).float()
#     # print('intxp:', intxp)
#     # print('intxp:', intxp.shape)

#     # label_mask_sum = label.eq(1)
#     # intx = torch.sum(label_mask_sum == 1, dim=1).float()
#     # print('intx:', intx)
#     # print('intx:', intx.shape)

#     intxn = torch.sum(mask_sum == 2, dim=1).float()
#     # print('intxn:', intxn)
#     # intxn: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0')

#     union = torch.sum(mask_sum > 0, dim=1).float()
#     # print('union:', union)
#     iou = intxn/union
#     return torch.mean(iou), (torch.sum(iou > 0.5).float()/iou.shape[0]), (torch.sum(iou > 0.7).float()/iou.shape[0])

#----------------------------------------------------SiamMask--------------------------------------------------#
