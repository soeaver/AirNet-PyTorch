import os
import numpy as np
import cv2
import datetime
import argparse

import torch
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn

import models as customized_models
from PIL import Image
from utils import measure_model, weight_filler
from utils import transforms as T

# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))
customized_models_names = sorted(name for name in customized_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))
for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]
model_names = default_model_names + customized_models_names
print(model_names)

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluat the imagenet validation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_id', type=str, default='1', help='gpu id for evaluation')
parser.add_argument('--data_root', type=str, default='/home/user/Database/ILSVRC2012/Data/CLS-LOC/val/',
                    help='Path to imagenet validation path')
parser.add_argument('--val_file', type=str, default='ILSVRC2012_val.txt',
                    help='val_file')
parser.add_argument('--arch', type=str,
                    default='air50_1x64d',
                    help='model arch')
parser.add_argument('--model_weights', type=str,
                    default='./ckpts/air50_1x64d.pth',
                    help='model weights')

parser.add_argument('--ground_truth', type=bool, default=True, help='whether provide gt labels')
parser.add_argument('--class_num', type=int, default=1000, help='predict classes number')
parser.add_argument('--skip_num', type=int, default=0, help='skip_num for evaluation')
parser.add_argument('--base_size', type=int, default=256, help='short size of images')
parser.add_argument('--crop_size', type=int, default=224, help='crop size of images')
parser.add_argument('--crop_type', type=str, default='center', choices=['center', 'multi'],
                    help='crop type of evaluation')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of multi-crop test')
parser.add_argument('--top_k', type=int, nargs='+', default=[1, 5], help='top_k')
parser.add_argument('--save_score_vec', type=bool, default=False, help='whether save the score map')

args = parser.parse_args()

# ------------------ MEAN & STD ---------------------
PIXEL_MEANS = np.array([0.485, 0.456, 0.406])
PIXEL_STDS = np.array([0.229, 0.224, 0.225])
# ---------------------------------------------------

# Set GPU id, CUDA and cudnn
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
USE_CUDA = torch.cuda.is_available()
cudnn.benchmark = True

# Create & Load model
MODEL = models.__dict__[args.arch]()
# Calculate FLOPs & Param
n_flops, n_convops, n_params = measure_model(MODEL, args.crop_size, args.crop_size)
print('==> FLOPs: {:.4f}M, Conv_FLOPs: {:.4f}M, Params: {:.4f}M'.
      format(n_flops / 1e6, n_convops / 1e6, n_params / 1e6))
del MODEL

# Load Weights
MODEL = models.__dict__[args.arch]()
checkpoint = torch.load(args.model_weights)
weight_dict = checkpoint
model_dict = MODEL.state_dict()
updated_dict, match_layers, mismatch_layers = weight_filler(weight_dict, model_dict)
model_dict.update(updated_dict)
MODEL.load_state_dict(model_dict)

# Switch to evaluate mode
MODEL.cuda().eval()
print(MODEL)

# Create log & dict
LOG_PTH = './log{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
SET_DICT = dict()
f = open(args.val_file, 'r')
img_order = 0
for _ in f:
    img_dict = dict()
    img_dict['path'] = os.path.join(args.data_root + _.strip().split(' ')[0])
    img_dict['evaluated'] = False
    img_dict['score_vec'] = []
    if args.ground_truth:
        img_dict['gt'] = int(_.strip().split(' ')[1])
    else:
        img_dict['gt'] = -1
    SET_DICT[img_order] = img_dict
    img_order += 1
f.close()


def eval_batch():
    eval_len = len(SET_DICT)
    accuracy = np.zeros(len(args.top_k))
    start_time = datetime.datetime.now()

    for i in xrange(eval_len - args.skip_num):
        im = cv2.imread(SET_DICT[i + args.skip_num]['path'])
        im = T.bgr2rgb(im)
        scale_im = T.pil_resize(Image.fromarray(im), args.base_size)
        normalized_im = T.normalize(np.asarray(scale_im) / 255.0, mean=PIXEL_MEANS, std=PIXEL_STDS)
        crop_ims = []
        if args.crop_type == 'center':  # for single crop
            crop_ims.append(T.center_crop(normalized_im, crop_size=args.crop_size))
        elif args.crop_type == 'multi':  # for 10 crops
            crop_ims.extend(T.mirror_crop(normalized_im, crop_size=args.crop_size))
        else:
            crop_ims.append(normalized_im)

        score_vec = np.zeros(args.class_num, dtype=np.float32)
        iter_num = int(len(crop_ims) / args.batch_size)
        timer_pt1 = datetime.datetime.now()
        for j in xrange(iter_num):
            input_data = np.asarray(crop_ims, dtype=np.float32)[j * args.batch_size:(j + 1) * args.batch_size]
            input_data = input_data.transpose(0, 3, 1, 2)
            input_data = torch.autograd.Variable(torch.from_numpy(input_data).cuda(), volatile=True)
            outputs = MODEL(input_data)
            scores = outputs.data.cpu().numpy()
            score_vec += np.sum(scores, axis=0)
        score_index = (-score_vec / len(crop_ims)).argsort()
        timer_pt2 = datetime.datetime.now()

        SET_DICT[i + args.skip_num]['evaluated'] = True
        SET_DICT[i + args.skip_num]['score_vec'] = score_vec / len(crop_ims)

        print 'Testing image: {}/{} {} {}/{} {}s' \
            .format(str(i + 1), str(eval_len - args.skip_num), str(SET_DICT[i + args.skip_num]['path'].split('/')[-1]),
                    str(score_index[0]), str(SET_DICT[i + args.skip_num]['gt']),
                    str((timer_pt2 - timer_pt1).microseconds / 1e6 + (timer_pt2 - timer_pt1).seconds)),

        for j in xrange(len(args.top_k)):
            if SET_DICT[i + args.skip_num]['gt'] in score_index[:args.top_k[j]]:
                accuracy[j] += 1
            tmp_acc = float(accuracy[j]) / float(i + 1)
            if args.top_k[j] == 1:
                print '\ttop_' + str(args.top_k[j]) + ':' + str(tmp_acc),
            else:
                print 'top_' + str(args.top_k[j]) + ':' + str(tmp_acc)
    end_time = datetime.datetime.now()

    w = open(LOG_PTH, 'w')
    s1 = 'Evaluation process ends at: {}. \nTime cost is: {}. '.format(str(end_time), str(end_time - start_time))
    s2 = '\nThe model is: {}. \nThe val file is: {}. \n{} images has been tested, crop_type is: {}, base_size is: {}, ' \
         'crop_size is: {}.'.format(args.model_weights, args.val_file, str(eval_len - args.skip_num),
                                    args.crop_type, str(args.base_size), str(args.crop_size))
    s3 = '\nThe PIXEL_MEANS is: ({}, {}, {}), PIXEL_STDS is : ({}, {}, {}).' \
        .format(str(PIXEL_MEANS[0]), str(PIXEL_MEANS[1]), str(PIXEL_MEANS[2]), str(PIXEL_STDS[0]), str(PIXEL_STDS[1]),
                str(PIXEL_STDS[2]))
    s4 = ''
    for i in xrange(len(args.top_k)):
        _acc = float(accuracy[i]) / float(eval_len - args.skip_num)
        s4 += '\nAccuracy of top_{} is: {}; correct num is {}.'.format(str(args.top_k[i]), str(_acc),
                                                                       str(int(accuracy[i])))
    print s1, s2, s3, s4
    w.write(s1 + s2 + s3 + s4)
    w.close()

    if args.save_score_vec:
        w = open(LOG_PTH.replace('.txt', 'scorevec.txt'), 'w')
        for i in xrange(eval_len - args.skip_num):
            w.write(SET_DICT[i + args.skip_num]['score_vec'])
        w.close()
    print('DONE!')


if __name__ == '__main__':
    eval_batch()
