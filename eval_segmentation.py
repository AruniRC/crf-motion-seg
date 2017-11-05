#!/home/arunirc/dense-crf/bin/python


'''
    Motion Segmentation Evaluation
    ----------------------------------------
    - Specify data locations and settings below.
    - 
    - 
'''

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio
from skimage import color
from skimage.io import imread, imsave
import os
from os import listdir, makedirs
from os.path import isfile, join, isdir
import argparse

import traceback
import warnings
import sys


GT_DATA = '/data/arunirc/Research/dense-crf-data/training_subset/'
CRF_SEG_DATA = '/data/arunirc/Research/dense-crf-data/FBMS-train-subset-01/'
RAW_SEG_DATA = '/data2/arunirc/Research/dense-crf/data/our/FBMS/Trainingset/'
OUT_DIR = '/data/arunirc/Research/dense-crf-data/eval-FBMS-train-subset-01'


def parse_input_opts():
    parser = argparse.ArgumentParser(description='Evaluate CRF segmentations')
    parser.add_argument('-g', '--gt_data', help='Ground truth data', \
                            default=GT_DATA)
    parser.add_argument('-c', '--crf_data', help='CRF segmentations', \
                            default=CRF_SEG_DATA)
    parser.add_argument('-r', '--raw_data', help='Raw segmentaitons', \
                            default=RAW_SEG_DATA)
    parser.add_argument('-o', '--out_dir', help='Specify output directory', \
                            default=OUT_DIR)
    parser.add_argument('-v', '--viz', help='Save visualized original and CRF segmentations as images', \
                            default=False, action='store_true')
    opts = parser.parse_args()
    return opts


def image_to_label(gt_img):
    '''
        Convert RGB ground truth image to label image, 
        with labels starting from zero.
        gt_label = image_to_label(gt_img)
    '''
    u_val = np.unique(gt_img)
    gt_label = np.zeros(gt_img.shape, dtype=np.uint8)
    for i, v in enumerate(u_val):
        idx = np.where(gt_img==v)
        gt_label[idx[0], idx[1]] = i
    return gt_label


def label_to_image(im):
    # rescale pixel values
    low, high = np.min(im), np.max(im)
    im1 = 255.0 * (im - low) / (high - low)
    return im1.astype('uint8')


def fast_hist(a, b, n):
    '''
        Computes a histogram over bins for an image segmentation.a
        Similar to the confusion matrix (for tfg-bg only)
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], \
                       minlength=n**2).reshape(n, n)


def get_iou(gt_label, res_label):    
    seg_hist = fast_hist(gt_label.flatten(), res_label.flatten(), \
                         np.size(np.unique(gt_label)))
    # per-class IU
    iu = 1.0 * np.diag(seg_hist) / (seg_hist.sum(1) + seg_hist.sum(0) - np.diag(seg_hist))
    return iu


def iou_from_hist(hist):
    '''
        Return Intersection-over-Union (IoU) metric given a histogram (confusion matrix)
    '''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def pr_from_hist(hist):
    '''
        Return precision, recall and f-measure given confusion matrix (hist)
    '''
    prec = hist[1,1] / (hist[1,1] + hist[0,1])
    rec = hist[1,1] / (hist[1,1] + hist[1,0])
    f_m = 2 * (prec*rec) / (prec + rec)
    return prec, rec, f_m




def eval_seg(GT_DATA, CRF_SEG_DATA, RAW_SEG_DATA, OUT_DIR):

    n_cl = 2
    hist_raw = np.zeros((n_cl, n_cl))
    hist_crf = np.zeros((n_cl, n_cl))

    for d in sorted(listdir(GT_DATA)):
        count = 0
        for fn in \
            [x for x in sorted(listdir(join(GT_DATA, d, 'GroundTruth'))) \
                                                     if x.endswith('.png')]:

            count = count + 1

            # ground truth labels
            gt_img = imread(join(GT_DATA, d, 'GroundTruth', fn))
            gt_label = image_to_label(gt_img)

            frame_num = str.split(fn, '_')[0]

            # CRF predicted labels
            seg_file = join(CRF_SEG_DATA, d, frame_num.zfill(5)+'.mat')
            if not isfile(seg_file):
                continue
            mat_data = sio.loadmat(seg_file)
            crf_res = mat_data['objectProb']
            if np.ndim(crf_res) == 3:
                crf_res_label =  np.argmax(crf_res, axis=2)
            else:
                crf_res_label = np.zeros(crf_res.shape)
                    

            # original predicted labels
            seg_file = join(RAW_SEG_DATA, d, 'objectProb', frame_num.zfill(5)+'.mat')         
            mat_data = sio.loadmat(seg_file)
            raw_res = mat_data['objectProb']
            if np.ndim(raw_res) == 3:
                raw_res_label =  np.argmax(raw_res, axis=2)
            else:
                raw_res_label = np.zeros(raw_res.shape)
                

            # visualize labelings
            if opts.viz:
                tiled_img = np.concatenate((label_to_image(gt_label), \
                                            label_to_image(raw_res_label), \
                                            label_to_image(crf_res_label)), axis=1)
                if not os.path.isdir(join(OUT_DIR, d)):
                    os.makedirs(join(OUT_DIR, d))
                imsave(join(OUT_DIR, d, frame_num.zfill(5)+'_gt_raw_crf.png'), tiled_img)


            # simplified performance metric -- foreground IOU
            gt_label = gt_label.astype(int)
            raw_res_label = raw_res_label.astype(int)
            crf_res_label = crf_res_label.astype(int)
            
            # label as 0 for background and 1 for (any) foreground class
            gt_label_binary = gt_label
            gt_label_binary[np.where(gt_label>0)] = 1
            raw_label_binary = raw_res_label
            raw_label_binary[np.where(raw_res_label>0)] = 1
            crf_label_binary = crf_res_label
            crf_label_binary[np.where(crf_res_label>0)] = 1
            
            # get IoU
            iou_raw = get_iou(gt_label_binary, raw_label_binary)
            iou_crf = get_iou(gt_label_binary, crf_label_binary)
            np.savetxt(join(OUT_DIR,d,frame_num.zfill(5)+'_result_iou_fg_raw.txt'), \
                       iou_raw, delimiter=',')
            np.savetxt(join(OUT_DIR,d,frame_num.zfill(5)+'_result_iou_fg_crf.txt'), \
                       iou_crf, delimiter=',')

            hist_raw += fast_hist(gt_label_binary.flatten(), \
                                  raw_label_binary.flatten(), \
                                  n_cl)
            hist_crf += fast_hist(gt_label_binary.flatten(), \
                                  crf_label_binary.flatten(), \
                                  n_cl) 

    
    # save over-all results
    iu_raw = np.diag(hist_raw) / (hist_raw.sum(1) + hist_raw.sum(0) - np.diag(hist_raw))
    iu_crf = np.diag(hist_crf) / (hist_crf.sum(1) + hist_crf.sum(0) - np.diag(hist_crf))

    # over-all precision-recall results
    prec, rec, f_m =  pr_from_hist(hist_crf)
    np.savetxt(join(OUT_DIR,'result_pr_fg_crf.txt'), [prec, rec, f_m], delimiter=',')


    print 'raw'
    print iu_raw
    print 'CRF'
    print iu_crf

    np.savetxt(join(OUT_DIR,'result_iou_fg_raw.txt'), iu_raw, delimiter=',')
    np.savetxt(join(OUT_DIR,'result_iou_fg_crf.txt'), iu_crf, delimiter=',')




# entry point
if __name__ == '__main__':
    opts = parse_input_opts()
    eval_seg(opts.gt_data, opts.crf_data, opts.raw_data, opts.out_dir)