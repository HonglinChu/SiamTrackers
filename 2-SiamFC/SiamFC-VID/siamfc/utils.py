import numpy as np
import cv2

def get_center(x):
    return (x - 1.) / 2.
#top-left bottom-right坐标转换成 cx，cy,w,h
def xyxy2cxcywh(bbox):
    return get_center(bbox[0]+bbox[2]), \
           get_center(bbox[1]+bbox[3]), \
           (bbox[2]-bbox[0]), \
           (bbox[3]-bbox[1])
#model_sz=127 original_sz= 获取  original_sz 大小的图像块，然后 resize 到 model-sz 大小
def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    xmin = cx - original_sz // 2
    xmax = cx + original_sz // 2
    ymin = cy - original_sz // 2
    ymax = cy + original_sz // 2
    im_h, im_w, _ = img.shape

    left = right = top = bottom = 0
    if xmin < 0:
        left = int(abs(xmin))#左边越界多少
    if xmax > im_w:
        right = int(xmax - im_w)#右边越界多少
    if ymin < 0:
        top = int(abs(ymin)) #上边越界多少
    if ymax > im_h:
        bottom = int(ymax - im_h)#下边越界多少
    #防止图像越界
    xmin = int(max(0, xmin))
    xmax = int(min(im_w, xmax))
    ymin = int(max(0, ymin))
    ymax = int(min(im_h, ymax))
    im_patch = img[ymin:ymax, xmin:xmax]#扣取图像块
    if left != 0 or right !=0 or top!=0 or bottom!=0:
        if img_mean is None:
            img_mean = tuple(map(int, img.mean(axis=(0, 1))))#求每个通道的均值
        im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right, #使用图像均值进行边缘的填充
                cv2.BORDER_CONSTANT, value=img_mean)
    if model_sz != original_sz:
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))  #将原始目标缩放到127x127的大小
    return im_patch
#   size_z=127
def get_exemplar_image(img, bbox, size_z, context_amount, img_mean=None):
    cx, cy, w, h = xyxy2cxcywh(bbox)
    wc_z = w + context_amount * (w+h)
    hc_z = h + context_amount * (w+h)
    s_z = np.sqrt(wc_z * hc_z) 
    scale_z = size_z / s_z   # 0.75
    exemplar_img = crop_and_pad(img, cx, cy, size_z, s_z, img_mean) #127*127
    return exemplar_img, scale_z, s_z

def get_instance_image(img, bbox, size_z, size_x, context_amount, img_mean=None):
    cx, cy, w, h = xyxy2cxcywh(bbox) #top-left bottom-right坐标转换成 cx，cy,w,h
    wc_z = w + context_amount * (w+h)# w+(w+h)*0.5
    hc_z = h + context_amount * (w+h)# h+(w+h)*0.5
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z #求尺度因子
    d_search = (size_x - size_z) / 2 #需要扩大的搜索区域到255x255
    pad = d_search / scale_z #pad的大小是32，在top bottom，left，right需要填充的像素
    s_x = s_z + 2 * pad
    scale_x = size_x / s_x
    instance_img = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    return instance_img, scale_x, s_x
#s-x是原始搜索区域大小，size-x 最终（理想）搜索区域大小，size-x-scales多尺度变换之后搜索区域大小，
def get_pyramid_instance_image(img, center, size_x, size_x_scales, img_mean=None):
    if img_mean is None:
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))#求每一个通道的均值
    pyramid = [crop_and_pad(img, center[0], center[1], size_x, size_x_scale, img_mean) #获取不同尺度下的instance
            for size_x_scale in size_x_scales]
    return pyramid

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