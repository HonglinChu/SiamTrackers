import torch
import numpy as np
import cv2
import random
import sys
import os
import math 
import matplotlib.pyplot as plt
from dasiamrpn.bbox_util import *
#--------------------------------------------------------------------------------------------#

class RandomStretch(object):
    def __init__(self, max_stretch=0.05):
        """Random resize image according to the stretch
        Args:
            max_stretch(float): 0 to 1 value   
        """
        self.max_stretch = max_stretch

    def __call__(self, sample):
        """
        Args:
            sample(numpy array): 3 or 1 dim image
        """
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        h, w = sample.shape[:2]
        shape = int(w * scale_w), int(h * scale_h)
        return cv2.resize(sample, shape, cv2.INTER_LINEAR)

class CenterCrop(object):
    def __init__(self, size):
        """Crop the image in the center according the given size 
            if size greater than image size, zero padding will adpot
        Args:
            size (tuple): desired size
        """
        self.size = size

    def __call__(self, sample):
        """
        Args:
            sample(numpy array): 3 or 1 dim image
        """
        shape = sample.shape[:2]
        cy, cx = (shape[0] - 1) // 2, (shape[1] - 1) // 2
        ymin, xmin = cy - self.size[0] // 2, cx - self.size[1] // 2
        ymax, xmax = cy + self.size[0] // 2 + self.size[0] % 2, \
                     cx + self.size[1] // 2 + self.size[1] % 2
        left = right = top = bottom = 0
        im_h, im_w = shape
        if xmin < 0:
            left = int(abs(xmin))
        if xmax > im_w:
            right = int(xmax - im_w)
        if ymin < 0:
            top = int(abs(ymin))
        if ymax > im_h:
            bottom = int(ymax - im_h)

        xmin = int(max(0, xmin))
        xmax = int(min(im_w, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(im_h, ymax))
        im_patch = sample[ymin:ymax, xmin:xmax]
        if left != 0 or right != 0 or top != 0 or bottom != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=0)
        return im_patch


class RandomCrop(object):
    def __init__(self, size, max_translate):
        """Crop the image in the center according the given size 
            if size greater than image size, zero padding will adpot
        Args:
            size (tuple): desired size
            max_translate: max translate of random shift
        """
        self.size = size
        self.max_translate = max_translate

    def __call__(self, sample):
        """
        Args:
            sample(numpy array): 3 or 1 dim image
        """
        shape = sample.shape[:2]
        cy_o = (shape[0] - 1) // 2
        cx_o = (shape[1] - 1) // 2
        cy = np.random.randint(cy_o - self.max_translate,
                               cy_o + self.max_translate + 1)
        cx = np.random.randint(cx_o - self.max_translate,
                               cx_o + self.max_translate + 1)
        assert abs(cy - cy_o) <= self.max_translate and \
               abs(cx - cx_o) <= self.max_translate
        ymin = cy - self.size[0] // 2
        xmin = cx - self.size[1] // 2
        ymax = cy + self.size[0] // 2 + self.size[0] % 2
        xmax = cx + self.size[1] // 2 + self.size[1] % 2
        left = right = top = bottom = 0
        im_h, im_w = shape
        if xmin < 0:
            left = int(abs(xmin))
        if xmax > im_w:
            right = int(xmax - im_w)
        if ymin < 0:
            top = int(abs(ymin))
        if ymax > im_h:
            bottom = int(ymax - im_h)

        xmin = int(max(0, xmin))
        xmax = int(min(im_w, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(im_h, ymax))
        im_patch = sample[ymin:ymax, xmin:xmax]
        if left != 0 or right != 0 or top != 0 or bottom != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=0)
        return im_patch


class ColorAug(object):
    def __init__(self, type_in='z'):
        if type_in == 'z':
            rgb_var = np.array([[3.2586416e+03, 2.8992207e+03, 2.6392236e+03],
                                [2.8992207e+03, 3.0958174e+03, 2.9321748e+03],
                                [2.6392236e+03, 2.9321748e+03, 3.4533721e+03]])
        if type_in == 'x':
            rgb_var = np.array([[2.4847285e+03, 2.1796064e+03, 1.9766885e+03],
                                [2.1796064e+03, 2.3441289e+03, 2.2357402e+03],
                                [1.9766885e+03, 2.2357402e+03, 2.7369697e+03]])
        self.v, _ = np.linalg.eig(rgb_var)
        self.v = np.sqrt(self.v)

    def __call__(self, sample):
        return sample + 0.1 * self.v * np.random.randn(3)


class RandomBlur(object):
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, sample):
        if np.random.rand(1) < self.ratio:
            # random kernel size
            kernel_size = np.random.choice([3, 5, 7])
            # random gaussian sigma
            sigma = np.random.rand() * 5
            return cv2.GaussianBlur(sample, (kernel_size, kernel_size), sigma)
        else:
            return sample

class Normalize(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, sample):
        return (sample / 255. - self.mean) / self.std


class ToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose(2, 0, 1)
        return torch.from_numpy(sample.astype(np.float32))


#----------------------------------------------------------------------------------------------------------------#
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the Image with the probability *p*
    Parameters
    ----------
    p: float
        The probability with which the image is flipped
    Returns
    -------
    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
            img_center = np.array(img.shape[:2])[::-1]/2
            img_center = np.hstack((img_center, img_center))
            if random.random() < self.p:
                img = img[:, ::-1, :]
                bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])

                box_w = abs(bboxes[:, 0] - bboxes[:, 2])

                bboxes[:, 0] -= box_w
                bboxes[:, 2] += box_w

            return img, bboxes

class HorizontalFlip(object):

    """Randomly horizontally flips the Image with the probability *p*
    Parameters
    ----------
    p: float
        The probability with which the image is flipped
    Returns
    -------
    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """
    def __init__(self):
        pass

    def __call__(self, img, bboxes):
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))

        img = img[:, ::-1, :]
        bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])

        box_w = abs(bboxes[:, 0] - bboxes[:, 2])

        bboxes[:, 0] -= box_w
        bboxes[:, 2] += box_w

        return img, bboxes

class RandomScale(object):
    """Randomly scales an image    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    Parameters
    ----------
    scale: float or tuple(float)
        if **float**, the image is scaled by a factor drawn 
        randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**,
        the `scale` is drawn randomly from values specified by the 
        tuple   
    Returns
    -------
    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box   
    """
    def __init__(self, scale = 0.2, diff = False):
        self.scale = scale
        
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)
        
        self.diff = diff 

    def __call__(self, img, bboxes):   
        #Chose a random digit to scale by 
        img_shape = img.shape
        
        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x
            
        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y
        
        img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)
        
        bboxes[:,:4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
        
        canvas = np.zeros(img_shape, dtype = np.uint8)
        
        y_lim = int(min(resize_scale_y,1)*img_shape[0])
        x_lim = int(min(resize_scale_x,1)*img_shape[1])
        
        canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]
        
        img = canvas
        bboxes = clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]], 0.25)
    
        return img, bboxes

class Scale(object):
    """Scales the image    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color
    Parameters
    ----------
    scale_x: float
        The factor by which the image is scaled horizontally
        
    scale_y: float
        The factor by which the image is scaled vertically
    Returns
    -------
    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box  
    """
    def __init__(self, scale_x = 0.2, scale_y = 0.2):
        self.scale_x = scale_x
        self.scale_y = scale_y
    
    def __call__(self, img, bboxes): 
        #Chose a random digit to scale by 
        img_shape = img.shape
        
        resize_scale_x = 1 + self.scale_x
        resize_scale_y = 1 + self.scale_y
        
        img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)
        
        bboxes[:,:4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
        
        canvas = np.zeros(img_shape, dtype = np.uint8)
        
        y_lim = int(min(resize_scale_y,1)*img_shape[0])
        x_lim = int(min(resize_scale_x,1)*img_shape[1])
        
        canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]
        
        img = canvas
        bboxes = clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]], 0.25)

        return img, bboxes  

# 平移
class RandomTranslate(object):
    """Randomly Translates the image    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn 
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the 
        tuple   
    Returns
    -------
    numpy.ndaaray
        Translated image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box 
    """
    def __init__(self, translate = 0.2, diff = False):
        self.translate = translate
        
        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"  
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1
        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)   
        self.diff = diff

    def __call__(self, img, bboxes):        
        #Chose a random digit to scale by 
        img_shape = img.shape
        #translate the image
        #percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self.translate)

        translate_factor_y = random.uniform(*self.translate)
        if not self.diff:
            translate_factor_y = translate_factor_x
       
        canvas = np.zeros(img_shape).astype(np.uint8)
    
        corner_x = int(translate_factor_x*img.shape[1])
        corner_y = int(translate_factor_y*img.shape[0])
        
        #change the origin to the top-left corner of the translated box
        orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]
           
        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas
        
        bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]
        
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)
        
        return img, bboxes
    
class Translate(object):
    """Randomly Translates the image    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn 
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the 
        tuple    
    Returns
    -------
    numpy.ndaaray
        Translated image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """

    def __init__(self, translate_x = 0.2, translate_y = 0.2, diff = False):
        self.translate_x = translate_x
        self.translate_y = translate_y

        assert self.translate_x > 0 and self.translate_x < 1
        assert self.translate_y > 0 and self.translate_y < 1
 
    def __call__(self, img, bboxes):        
        #Chose a random digit to scale by 
        img_shape = img.shape
        #translate the image
        #percentage of the dimension of the image to translate
        translate_factor_x = self.translate_x
        translate_factor_y = self.translate_y
         
        canvas = np.zeros(img_shape).astype(np.uint8)
        #get the top-left corner co-ordinates of the shifted box 
        corner_x = int(translate_factor_x*img.shape[1])
        corner_y = int(translate_factor_y*img.shape[0])
        
        #change the origin to the top-left corner of the translated box
        orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]

        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas
        
        bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]
        
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)

        return img, bboxes
    
class RandomRotate(object):

    def __init__(self,angle=5,scale=1.):
        
        self.angle = angle
        self.scale = scale 

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"  
            
        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, bboxes):
        '''
        参考:https://blog.csdn.net/zhy9495/article/details/86703407
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        #----------------------------------------- 旋转图像 ----------------------------
        angle = random.uniform(*self.angle)
        #print(angle) 
        
        scale = random.uniform(0.7, 1)
       
        w = img.shape[1]
        h = img.shape[0]

        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        # 仿射变换
        channel_mean=np.mean(img,axis=(0,1))
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4,borderValue=(channel_mean[0],channel_mean[1],channel_mean[2]))

        #---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx+rw
            ry_max = ry+rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])
        
        return rot_img, np.array(rot_bboxes)

    
class Rotate(object):
    """Rotates an image    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    Parameters
    ----------
    angle: float
        The angle by which the image is to be rotated  
    Returns
    -------
    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """
    def __init__(self, angle):
        self.angle = angle
        
    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
              
        """
        angle = self.angle
        print(self.angle)
        
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
        
        corners = get_corners(bboxes)
        
        corners = np.hstack((corners, bboxes[:,4:]))

        img = rotate_im(img, angle)
        
        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
          
        new_bbox = get_enclosing_box(corners)
        
        scale_factor_x = img.shape[1] / w
        
        scale_factor_y = img.shape[0] / h
        
        img = cv2.resize(img, (w,h))
        
        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
    
        bboxes  = new_bbox

        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
        
        return img, bboxes
        
class RandomShear(object):
    """Randomly shears an image in horizontal direction   
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    Parameters
    ----------
    shear_factor: float or tuple(float)
        if **float**, the image is sheared horizontally by a factor drawn 
        randomly from a range (-`shear_factor`, `shear_factor`). If **tuple**,
        the `shear_factor` is drawn randomly from values specified by the 
        tuple
    Returns
    -------
    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box 
    """
    def __init__(self, shear_factor = 0.2):
        self.shear_factor = shear_factor
        
        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for scaling factor"   
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)
        
        shear_factor = random.uniform(*self.shear_factor)
        
    def __call__(self, img, bboxes):
    
        shear_factor = random.uniform(*self.shear_factor)
    
        w,h = img.shape[1], img.shape[0]
    
        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)
    
        M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
    
        nW =  img.shape[1] + abs(shear_factor*img.shape[0])
    
        bboxes[:,[0,2]] += ((bboxes[:,[1,3]]) * abs(shear_factor) ).astype(int) 
    
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
    
        if shear_factor < 0:
        	img, bboxes = HorizontalFlip()(img, bboxes)
    
        img = cv2.resize(img, (w,h))
    
        scale_factor_x = nW / w
    
        bboxes[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1] 
    
        return img, bboxes
        
class Shear(object):
    """Shears an image in horizontal direction   
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    Parameters
    ----------
    shear_factor: float
        Factor by which the image is sheared in the x-direction  
    Returns
    -------
    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box  
    """
    def __init__(self, shear_factor = 0.2):
        self.shear_factor = shear_factor     
    
    def __call__(self, img, bboxes):
        
        shear_factor = self.shear_factor
        if shear_factor < 0:
            img, bboxes = HorizontalFlip()(img, bboxes)

        M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
                
        nW =  img.shape[1] + abs(shear_factor*img.shape[0])
        
        bboxes[:,[0,2]] += ((bboxes[:,[1,3]])*abs(shear_factor)).astype(int) 
        
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
        
        if shear_factor < 0:
             img, bboxes = HorizontalFlip()(img, bboxes)
             
        return img, bboxes
    
class Resize(object):
    """Resize the image in accordance to `image_letter_box` function in darknet 
    
    The aspect ratio is maintained. The longer side is resized to the input 
    size of the network, while the remaining space on the shorter side is filled 
    with black color. **This should be the last transform**
    Parameters
    ----------
    inp_dim : tuple(int)
        tuple containing the size to which the image will be resized.    
    Returns
    -------
    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box 
    """
    def __init__(self, inp_dim):
        self.inp_dim = inp_dim
        
    def __call__(self, img, bboxes):
        w,h = img.shape[1], img.shape[0]
        img = letterbox_image(img, self.inp_dim)

        scale = min(self.inp_dim/h, self.inp_dim/w)
        bboxes[:,:4] *= (scale)
    
        new_w = scale*w
        new_h = scale*h
        inp_dim = self.inp_dim   
    
        del_h = (inp_dim - new_h)/2
        del_w = (inp_dim - new_w)/2
    
        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)
    
        bboxes[:,:4] += add_matrix
    
        img = img.astype(np.uint8)
    
        return img, bboxes 

class RandomHSV(object):
    """HSV Transform to vary hue saturation and brightness
    Hue has a range of 0-179
    Saturation and Brightness have a range of 0-255. 
    Chose the amount you want to change thhe above quantities accordingly. 
    Parameters
    ----------
    hue : None or int or tuple (int)
        If None, the hue of the image is left unchanged. If int, 
        a random int is uniformly sampled from (-hue, hue) and added to the 
        hue of the image. If tuple, the int is sampled from the range 
        specified by the tuple.    
    saturation : None or int or tuple(int)
        If None, the saturation of the image is left unchanged. If int, 
        a random int is uniformly sampled from (-saturation, saturation) 
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.    
    brightness : None or int or tuple(int)
        If None, the brightness of the image is left unchanged. If int, 
        a random int is uniformly sampled from (-brightness, brightness) 
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.   
    Returns
    -------
    numpy.ndaaray
        Transformed image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box 
    """
    def __init__(self, hue = None, saturation = None, brightness = None):
        if hue:
            self.hue = hue 
        else:
            self.hue = 0
            
        if saturation:
            self.saturation = saturation 
        else:
            self.saturation = 0
            
        if brightness:
            self.brightness = brightness
        else:
            self.brightness = 0

        if type(self.hue) != tuple:
            self.hue = (-self.hue, self.hue)
            
        if type(self.saturation) != tuple:
            self.saturation = (-self.saturation, self.saturation)
        
        if type(brightness) != tuple:
            self.brightness = (-self.brightness, self.brightness)
    
    def __call__(self, img, bboxes):

        hue = random.randint(*self.hue)
        saturation = random.randint(*self.saturation)
        brightness = random.randint(*self.brightness)
        
        img = img.astype(int)
        
        a = np.array([hue, saturation, brightness]).astype(int)
        img += np.reshape(a, (1,1,3))
        
        img = np.clip(img, 0, 255)
        img[:,:,0] = np.clip(img[:,:,0],0, 179)
        
        img = img.astype(np.uint8)
  
        return img, bboxes
    
class Sequence(object):

    """Initialise Sequence object
    Apply a Sequence of transformations to the images/boxes.
    Parameters
    ----------
    augemnetations : list 
        List containing Transformation Objects in Sequence they are to be 
        applied
    probs : int or list 
        If **int**, the probability with which each of the transformation will 
        be applied. If **list**, the length must be equal to *augmentations*. 
        Each element of this list is the probability with which each 
        corresponding transformation is applied
    Returns
    -------
    Sequence
        Sequence Object  
    """
    def __init__(self, augmentations, probs = 1):
  
        self.augmentations = augmentations
        self.probs = probs
        
    def __call__(self, images, bboxes):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i] 
            else:
                prob = self.probs
                
            if random.random() < prob:
                images, bboxes = augmentation(images, bboxes)
        return images, bboxes