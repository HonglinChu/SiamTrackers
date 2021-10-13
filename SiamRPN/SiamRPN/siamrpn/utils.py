import torch
import numpy as np
import cv2
import time
import os 
from IPython import embed


# freeze layers
import torch.nn as  nn 
def freeze_layers(model):
    print('------------------------------------------------------------------------------------------------')
    for layer in model.featureExtract[:10]:
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()#
            for k, v in layer.named_parameters():#‘weight’，‘bias’
                v.requires_grad = False
        elif isinstance(layer, nn.Conv2d):
            for k, v in layer.named_parameters(): #参数容器的名字‘weight’，‘bias’
                v.requires_grad = False           #默认是True，改为False，则不需要反向传播
        elif isinstance(layer, nn.MaxPool2d):
            continue
        elif isinstance(layer, nn.ReLU):
            continue
        else:
            raise KeyError('error in fixing former 3 layers')

    #print("fixed layers:")
    # print(model.featureExtract[:10])

#获取所有像素点的anchors
#base_size=8; scales=8; ratios=0.33, 0.5, 1, 2, 3
def generate_anchors(total_stride, base_size, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = base_size * base_size
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1
    anchor= np.tile(anchor, score_size * score_size)# [5,1444]
    #tile是瓷砖的意思，就是将原矩阵横向、纵向地复制 tile（matrix，（1,4)=tile（matrix，4）横向复制
    anchor = anchor.reshape((-1, 4))#[5x17x17,4]  #
    # (5,17x17x4) to (17x17x5,4)
    ori = - (score_size // 2) * total_stride #??
    # the left displacement
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    # (15,15) or (17,17) or (19,19)
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), np.tile(yy.flatten(), (anchor_num, 1)).flatten()

    # (15,15) to (225,1) to (5,225) to (225x5,1)
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor
def get_center(x):
    return (x - 1.) / 2.


def xyxy2cxcywh(bbox):
    return get_center(bbox[0] + bbox[2]), \
           get_center(bbox[1] + bbox[3]), \
           (bbox[2] - bbox[0]), \
           (bbox[3] - bbox[1])

def cxcywh2xyxy(bboxes):
    if len(np.array(bboxes).shape) == 1:
        bboxes = np.array(bboxes)[None, :]
    else:
        bboxes = np.array(bboxes)
    x1 = bboxes[:, 0:1] + 1 / 2 - bboxes[:, 2:3] / 2
    x2 = x1 + bboxes[:, 2:3] - 1
    y1 = bboxes[:, 1:2] + 1 / 2 - bboxes[:, 3:4] / 2
    y2 = y1 + bboxes[:, 3:4] - 1
    return np.concatenate([x1, y1, x2, y2], 1)


def nms(bboxes, scores, num, threshold=0.7):
    sort_index = np.argsort(scores)[::-1] #[::-1]表示反向排列，之后得到的sort—index是从大到小排列
    sort_boxes = bboxes[sort_index]
    selected_bbox = [sort_boxes[0]]
    selected_index = [sort_index[0]]
    for i, bbox in enumerate(sort_boxes):
        iou = compute_iou(selected_bbox, bbox)
        # print(iou, bbox, selected_bbox)
        if np.max(iou) < threshold: #求序列的最值，至少接收一个参数
            selected_bbox.append(bbox)
            selected_index.append(sort_index[i])
            if len(selected_bbox) >= num:
                break
    return selected_index


def nms_worker(x, threshold=0.7):
    bboxes, scores, num = x
    if len(bboxes) == 0:
        selected_index = [0]
    else:
        sort_index = np.argsort(scores)[::-1]
        sort_boxes = bboxes[sort_index]
        selected_bbox = [sort_boxes[0]]
        selected_index = [sort_index[0]]
        for i, bbox in enumerate(sort_boxes):
            iou = compute_iou(selected_bbox, bbox)
            # print(iou, bbox, selected_bbox)
            if np.max(iou) < threshold:
                selected_bbox.append(bbox)
                selected_index.append(sort_index[i])
                if len(selected_bbox) >= num:
                    break
    return selected_index


def round_up(value):
    # 替换内置round函数,实现保留2位小数的精确四舍五入
    return round(value + 1e-6 + 1000) - 1000


def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    im_h, im_w, _ = img.shape

    xmin = cx - (original_sz - 1) / 2
    xmax = xmin + original_sz - 1
    ymin = cy - (original_sz - 1) / 2
    ymax = ymin + original_sz - 1
    #边界部分要填充的像素
    left = int(round_up(max(0., -xmin)))
    top = int(round_up(max(0., -ymin)))
    right = int(round_up(max(0., xmax - im_w + 1)))
    bottom = int(round_up(max(0., ymax - im_h + 1)))

    xmin = int(round_up(xmin + left))
    xmax = int(round_up(xmax + left))
    ymin = int(round_up(ymin + top))
    ymax = int(round_up(ymax + top)) #填充之后的坐标
    r, c, k = img.shape
    if any([top, bottom, left, right]): #如果全部是false，则返回false，否则返回true
        te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)  # 0 is better than 1 initialization
        te_im[top:top + r, left:left + c, :] = img
        if top:
            te_im[0:top, left:left + c, :] = img_mean
        if bottom:
            te_im[r + top:, left:left + c, :] = img_mean
        if left:
            te_im[:, 0:left, :] = img_mean
        if right:
            te_im[:, c + left:, :] = img_mean
        im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    else:
        im_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
    else:
        im_patch = im_patch_original
    scale = model_sz / im_patch_original.shape[0]
    return im_patch, scale


def get_exemplar_image(img, bbox, size_z, context_amount, img_mean=None):
    cx, cy, w, h = bbox
    #cx, cy, w, h = xyxy2cxcywh(bbox)
    wc_z = w + context_amount * (w + h) #模板图像扩充  
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z) #
    scale_z = size_z / s_z
    exemplar_img, _ = crop_and_pad(img, cx, cy, size_z, s_z, img_mean)
    return exemplar_img, scale_z, s_z

#   size_z=127 模板大小； size_x 搜索区域大小； 
def get_instance_image(img, bbox, size_z, size_x, context_amount, img_mean=None):
    cx, cy, w, h = bbox  # float type
    #cx, cy, w, h = xyxy2cxcywh(bbox)
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)  # the width of the crop box
    scale_z = size_z / s_z

    s_x = s_z * size_x / size_z
    instance_img, scale_x = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    w_x = w * scale_x
    h_x = h * scale_x # 尺度因子
    # point_1 = (size_x + 1) / 2 - w_x / 2, (size_x + 1) / 2 - h_x / 2
    # point_2 = (size_x + 1) / 2 + w_x / 2, (size_x + 1) / 2 + h_x / 2
    # frame = cv2.rectangle(instance_img, (int(point_1[0]),int(point_1[1])), (int(point_2[0]),int(point_2[1])), (0, 255, 0), 2)
    # cv2.imwrite('1.jpg', frame)
    return instance_img, w_x, h_x, scale_x


def box_transform(anchors, gt_box):
    anchor_xctr = anchors[:, :1]  #cx
    anchor_yctr = anchors[:, 1:2] #cy
    anchor_w = anchors[:, 2:3]   
    anchor_h = anchors[:, 3:]
    gt_cx, gt_cy, gt_w, gt_h = gt_box

    target_x = (gt_cx - anchor_xctr) / anchor_w # offset-x
    target_y = (gt_cy - anchor_yctr) / anchor_h # offset-y
    target_w = np.log(gt_w / anchor_w) #offset-w
    target_h = np.log(gt_h / anchor_h) #offset-h
    regression_target = np.hstack((target_x, target_y, target_w, target_h)) #hstack((1805,1),(1805,1),(1805,1),(1805,1))左右拼接
    return regression_target


def box_transform_inv(anchors, offset):
    anchor_xctr = anchors[:, :1] #cx
    anchor_yctr = anchors[:, 1:2]#cy
    anchor_w = anchors[:, 2:3] #w
    anchor_h = anchors[:, 3:]  #h
    offset_x, offset_y, offset_w, offset_h = offset[:, :1], offset[:, 1:2], offset[:, 2:3], offset[:, 3:],

    box_cx = anchor_w * offset_x + anchor_xctr
    box_cy = anchor_h * offset_y + anchor_yctr
    box_w = anchor_w * np.exp(offset_w)
    box_h = anchor_h * np.exp(offset_h)
    box = np.hstack([box_cx, box_cy, box_w, box_h]) #水平方向堆叠  np.vstack竖直方向堆叠
    return box

#获取前k个得分最高的box
def get_topk_box(cls_score, pred_regression, anchors, topk=10):
    # anchors xc,yc,w,h
    regress_offset = pred_regression.cpu().detach().numpy()

    scores, index = torch.topk(cls_score, topk, )
    index = index.view(-1).cpu().detach().numpy()

    topk_offset = regress_offset[index, :]
    anchors = anchors[index, :]
    pred_box = box_transform_inv(anchors, topk_offset)
    return pred_box


def compute_iou(anchors, box):
    if np.array(anchors).ndim == 1:#几个维度
        anchors = np.array(anchors)[None, :]
    else:
        anchors = np.array(anchors)
    if np.array(box).ndim == 1:    #几个维度
        box = np.array(box)[None, :]
    else:
        box = np.array(box)
    gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))#重复数组来构建新的数组[1805,4]

    anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5 # cx-(w-1)/2=x1
    anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5 # cx+(w-1)/2=x2
    anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5 # cy-(h-1)/2=y1
    anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5 # cy+(h-1)/2=y2

    gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
    gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
    gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
    gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

    xx1 = np.max([anchor_x1, gt_x1], axis=0)
    xx2 = np.min([anchor_x2, gt_x2], axis=0)
    yy1 = np.max([anchor_y1, gt_y1], axis=0)
    yy2 = np.min([anchor_y2, gt_y2], axis=0)
    #计算相交的区域
    inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                           axis=0)
    area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
    return iou


def get_pyramid_instance_image(img, center, size_x, size_x_scales, img_mean=None):
    if img_mean is None:
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
    pyramid = [crop_and_pad(img, center[0], center[1], size_x, size_x_scale, img_mean)
               for size_x_scale in size_x_scales]
    return pyramid


def add_box_img(img, boxes, color=(0, 255, 0)):
    # boxes (x,y,w,h)
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    img = img.copy()
    img_ctx = (img.shape[1] - 1) / 2 #w
    img_cty = (img.shape[0] - 1) / 2 #h
    for box in boxes:
        point_1 = [img_ctx - box[2]/ 2 + box[0] + 0.5, img_cty - box[3] / 2 + box[1] + 0.5]  # x1，y1
        point_2 = [img_ctx + box[2] / 2 + box[0] - 0.5, img_cty + box[3] / 2 + box[1] - 0.5] # x2，y2
        point_1[0] = np.clip(point_1[0], 0, img.shape[1]) #w
        point_2[0] = np.clip(point_2[0], 0, img.shape[1]) #w
        point_1[1] = np.clip(point_1[1], 0, img.shape[0]) #h
        point_2[1] = np.clip(point_2[1], 0, img.shape[0]) #h
        img = cv2.rectangle(img, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])),
                            color, 2)
    return img


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']
def add_box_img_left_top(img, boxes, color=(0, 255, 0)):
    # boxes (x,y,w,h)
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    img = img.copy()
    for i, box in enumerate(boxes):
        point_1 = [- box[2] / 2 + box[0] + 0.5, - box[3] / 2 + box[1] + 0.5]
        point_2 = [+ box[2] / 2 + box[0] - 0.5, + box[3] / 2 + box[1] - 0.5]
        point_1[0] = np.clip(point_1[0], 0, img.shape[1])
        point_2[0] = np.clip(point_2[0], 0, img.shape[1])
        point_1[1] = np.clip(point_1[1], 0, img.shape[0])
        point_2[1] = np.clip(point_2[1], 0, img.shape[0])
        img = cv2.rectangle(img, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])),
                            color, 2)
    return img

def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    
    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
    
    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
    
    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img
        


