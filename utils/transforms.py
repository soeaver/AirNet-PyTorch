import cv2
import math
import numbers
import random
import collections
import numpy as np
from PIL import Image


def bgr2rgb(im):
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return rgb_im


def rgb2bgr(im):
    bgr_im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return bgr_im


def normalize(im, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), rgb=False):
    if rgb:
        r, g, b = cv2.split(im)
    else:
        b, g, r = cv2.split(im)
    norm_im = cv2.merge([(b - mean[0]) / std[0], (g - mean[1]) / std[1], (r - mean[2]) / std[2]])
    return norm_im


def scale(im, short_size=256, max_size=1e5, interp=cv2.INTER_LINEAR):
    """ support gray im; interp: cv2.INTER_LINEAR (default) or cv2.INTER_NEAREST; """
    im_size_min = np.min(im.shape[0:2])
    im_size_max = np.max(im.shape[0:2])
    scale_ratio = float(short_size) / float(im_size_min)
    if np.round(scale_ratio * im_size_max) > float(max_size):
        scale_ratio = float(max_size) / float(im_size_max)

    scale_im = cv2.resize(im, None, None, fx=scale_ratio, fy=scale_ratio, interpolation=interp)

    return scale_im, scale_ratio


def scale_by_max(im, long_size=512, interp=cv2.INTER_LINEAR):
    """ support gray im; interp: cv2.INTER_LINEAR (default) or cv2.INTER_NEAREST; """
    im_size_max = np.max(im.shape[0:2])
    scale_ratio = float(long_size) / float(im_size_max)

    scale_im = cv2.resize(im, None, None, fx=scale_ratio, fy=scale_ratio, interpolation=interp)

    return scale_im, scale_ratio


def scale_by_target(im, target_size=(512, 256), interp=cv2.INTER_LINEAR):
    """ target_size=(h, w), support gray im; interp: cv2.INTER_LINEAR (default) or cv2.INTER_NEAREST; """
    min_factor = min(float(target_size[0]) / float(im.shape[0]),
                     float(target_size[1]) / float(im.shape[1]))

    scale_im = cv2.resize(im, None, None, fx=min_factor, fy=min_factor, interpolation=interp)

    return scale_im, min_factor


def rotate(im, degree=0, borderValue=(0, 0, 0), interp=cv2.INTER_LINEAR):
    """ support gray im; interp: cv2.INTER_LINEAR (default) or cv2.INTER_NEAREST; """
    h, w = im.shape[:2]
    rotate_mat = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)
    rotation = cv2.warpAffine(im, rotate_mat, (w, h), flags=interp,
                              borderValue=cv2.cv.Scalar(borderValue[0], borderValue[1], borderValue[2]))

    return rotation


def HSV_adjust(im, color=1.0, contrast=1.0, brightness=1.0):
    _HSV = np.dot(cv2.cvtColor(im, cv2.COLOR_BGR2HSV).reshape((-1, 3)),
                  np.array([[color, 0, 0], [0, contrast, 0], [0, 0, brightness]]))

    _HSV_H = np.where(_HSV < 255, _HSV, 255)
    hsv = cv2.cvtColor(np.uint8(_HSV_H.reshape((-1, im.shape[1], 3))), cv2.COLOR_HSV2BGR)

    return hsv


def salt_pepper(im, SNR=1.0):
    """ SNR: better >= 0.9; """
    noise_num = int((1 - SNR) * im.shape[0] * im.shape[1])
    noise_im = im.copy()
    for i in xrange(noise_num):
        rand_x = np.random.random_integers(0, im.shape[0] - 1)
        rand_y = np.random.random_integers(0, im.shape[1] - 1)

        if np.random.random_integers(0, 1) == 0:
            noise_im[rand_x, rand_y] = 0
        else:
            noise_im[rand_x, rand_y] = 255

    return noise_im


def padding_im(im, target_size=(512, 512), borderType=cv2.BORDER_CONSTANT, mode=0):
    """ support gray im; target_size=(h, w); mode=0 left-top, mode=1 center; """
    if mode not in (0, 1):
        raise Exception("mode need to be one of 0 or 1, 0 for left-top mode, 1 for center mode.")

    pad_h_top = max(int((target_size[0] - im.shape[0]) * 0.5), 0) * mode
    pad_h_bottom = max(target_size[0] - im.shape[0], 0) - pad_h_top
    pad_w_left = max(int((target_size[1] - im.shape[1]) * 0.5), 0) * mode
    pad_w_right = max(target_size[1] - im.shape[1], 0) - pad_w_left

    if borderType == cv2.BORDER_CONSTANT:
        pad_im = cv2.copyMakeBorder(im, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, cv2.BORDER_CONSTANT)
    else:
        pad_im = cv2.copyMakeBorder(im, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, borderType)

    return pad_im


def extend_bbox(im, bbox, margin=(0.5, 0.5, 0.5, 0.5)):
    box_w = int(bbox[2] - bbox[0])
    box_h = int(bbox[3] - bbox[1])

    new_x1 = max(1, bbox[0] - margin[0] * box_w)
    new_y1 = max(1, bbox[1] - margin[1] * box_h)
    new_x2 = min(im.shape[1] - 1, bbox[2] + margin[2] * box_w)
    new_y2 = min(im.shape[0] - 1, bbox[3] + margin[3] * box_h)

    return np.asarray([new_x1, new_y1, new_x2, new_y2])


def bbox_crop(im, bbox):
    return im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]


def center_crop(im, crop_size=224):  # single crop
    im_size_min = min(im.shape[:2])
    if im_size_min < crop_size:
        return
    yy = int((im.shape[0] - crop_size) / 2)
    xx = int((im.shape[1] - crop_size) / 2)
    crop_im = im[yy: yy + crop_size, xx: xx + crop_size]

    return crop_im


def over_sample(im, crop_size=224):  # 5 crops of image
    im_size_min = min(im.shape[:2])
    if im_size_min < crop_size:
        return
    yy = int((im.shape[0] - crop_size) / 2)
    xx = int((im.shape[1] - crop_size) / 2)
    sample_list = [im[:crop_size, :crop_size], im[-crop_size:, -crop_size:], im[:crop_size, -crop_size:],
                   im[-crop_size:, :crop_size], im[yy: yy + crop_size, xx: xx + crop_size]]

    return sample_list


def mirror_crop(im, crop_size=224):  # 10 crops
    crop_list = []
    mirror = im[:, ::-1]
    crop_list.extend(over_sample(im, crop_size=crop_size))
    crop_list.extend(over_sample(mirror, crop_size=crop_size))

    return crop_list


def multiscale_mirrorcrop(im, scales=(256, 288, 320, 352)):  # 120(4*3*10) crops
    crop_list = []
    im_size_min = np.min(im.shape[0:2])
    for i in scales:
        resize_im = cv2.resize(im, (im.shape[1] * i / im_size_min, im.shape[0] * i / im_size_min))
        yy = int((resize_im.shape[0] - i) / 2)
        xx = int((resize_im.shape[1] - i) / 2)
        for j in xrange(3):
            left_center_right = resize_im[yy * j: yy * j + i, xx * j: xx * j + i]
            mirror = left_center_right[:, ::-1]
            crop_list.extend(over_sample(left_center_right))
            crop_list.extend(over_sample(mirror))

    return crop_list


def multi_scale(im, scales=(480, 576, 688, 864, 1200), max_sizes=(800, 1000, 1200, 1500, 1800), image_flip=False):
    im_size_min = np.min(im.shape[0:2])
    im_size_max = np.max(im.shape[0:2])

    scale_ims = []
    scale_ratios = []
    for i in xrange(len(scales)):
        scale_ratio = float(scales[i]) / float(im_size_min)
        if np.round(scale_ratio * im_size_max) > float(max_sizes[i]):
            scale_ratio = float(max_sizes[i]) / float(im_size_max)
        resize_im = cv2.resize(im, None, None, fx=scale_ratio, fy=scale_ratio,
                               interpolation=cv2.INTER_LINEAR)
        scale_ims.append(resize_im)
        scale_ratios.append(scale_ratio)
        if image_flip:
            scale_ims.append(cv2.resize(im[:, ::-1], None, None, fx=scale_ratio, fy=scale_ratio,
                                        interpolation=cv2.INTER_LINEAR))
            scale_ratios.append(-scale_ratio)

    return scale_ims, scale_ratios


def multi_scale_by_max(im, scales=(480, 576, 688, 864, 1200), image_flip=False):
    im_size_max = np.max(im.shape[0:2])

    scale_ims = []
    scale_ratios = []
    for i in xrange(len(scales)):
        scale_ratio = float(scales[i]) / float(im_size_max)

        resize_im = cv2.resize(im, None, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_LINEAR)
        scale_ims.append(resize_im)
        scale_ratios.append(scale_ratio)
        if image_flip:
            scale_ims.append(cv2.resize(im[:, ::-1], None, None, fx=scale_ratio, fy=scale_ratio,
                                        interpolation=cv2.INTER_LINEAR))
            scale_ratios.append(-scale_ratio)

    return scale_ims, scale_ratios


def pil_resize(im, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = im.size
        if (w <= h and w == size) or (h <= w and h == size):
            return im
        if w < h:
            ow = size
            oh = int(size * h / w)
            return im.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return im.resize((ow, oh), interpolation)
    else:
        return im.resize(size[::-1], interpolation) 

