

from __future__ import division

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio
from skimage import color
from skimage.io import imread, imsave

from os import listdir, makedirs
from os.path import isfile, join, isdir
import argparse

import sys
import os
import subprocess
import shutil

# from apply_crf import *


# Change settings here
# Uncomment Run1 or Run 2 ... for different resolutions of the grid-search

IMAGE_DATA = '/data/arunirc/Research/dense-crf-data/training_subset/'
SEG_DATA = '/data/arunirc/Research/dense-crf-data/our-modifiedObjPrior/FBMS/Trainingset'
OUT_DIR = '/data/arunirc/Research/dense-crf-data/cross-val-crf-modifiedObjPrior/'
MODE = 'run'   # 'run' or 'eval' or 'pick'
METRIC = 'pr'  # 'iou' or 'pr'



# Run 1
# bilateral (colorspace)
# RUN_NUM = 1
# range_W=[3, 5, 10]
# range_XY_STD=[40, 50, 60, 70, 80, 90, 100]
# range_RGB_STD=[3, 5, 7, 9, 10]

# # Run 2
# # bilateral (colorspace)
RUN_NUM = 2
range_W = [10, 15, 20]
range_XY_STD = [10, 20, 30, 40]
range_RGB_STD = [1, 2, 3, 4, 5, 6]

# range_W=[5]
# range_XY_STD=[40]
# range_RGB_STD=[3]

# gaussian (positional)
POS_W = 3
POS_X_STD = 3

MAX_ITER = 5





def grid_runner():
    '''
        Run CRF segmentations using a grid-search over CRF settings
    '''

    if not os.path.isdir(OUT_DIR):
            os.makedirs(OUT_DIR)

    for w in range_W:
        Bi_W=w
        for x in range_XY_STD:
            Bi_XY_STD=x
            for r in range_RGB_STD:
                Bi_R_STD = r



                out_dir_name = join( OUT_DIR, 'w-'+str(w) + '_x-'+str(x) + '_r-'+str(r) )

                # # if already computed in a prior run -- skip
                if os.path.isdir(out_dir_name):
                    print 'Skipping %s. Already exists.' % out_dir_name
                    continue

                cmd = 'python apply_crf.py ' \
                        + '-i ' + IMAGE_DATA + ' ' \
                        + '-s ' + SEG_DATA + ' ' \
                        + '-o ' + out_dir_name + ' ' \
                        + '-d ' + 'fbms ' \
                        + '-cgw ' + str(POS_W) + ' ' \
                        + '-cgx ' + str(POS_X_STD) + ' ' \
                        + '-cbw ' + str(w) + ' ' \
                        + '-cbx ' + str(x) + ' ' \
                        + '-cbc ' + str(r) + ' ' \
                        + '-mi ' + str(MAX_ITER) + ' -z &'
                print cmd
                subprocess.call(cmd, shell=True)
    print 'done'



def grid_evaluater():
    '''
       Calculate an evaluation metric over all pre-computed segmentation results
       (run after `grid_runner` ) 
    '''

    print 'Running evaluations'

    GT_DATA = IMAGE_DATA
    RAW_SEG_DATA = SEG_DATA

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    for w in range_W:
        Bi_W=w
        for x in range_XY_STD:
            Bi_XY_STD=x
            for r in range_RGB_STD:
                Bi_R_STD = r

                out_dir_name = join( OUT_DIR, 'w-'+str(w) + '_x-'+str(x) + '_r-'+str(r) )
                CRF_SEG_DATA = out_dir_name

                cmd = 'python eval_segmentation.py ' \
                        + '-g ' + GT_DATA + ' ' \
                        + '-c ' + CRF_SEG_DATA + ' ' \
                        + '-r ' + RAW_SEG_DATA + ' ' \
                        + '-o ' + out_dir_name + ' &'
                print cmd
                subprocess.call(cmd, shell=True)

    print 'Done'



def grid_picker():
    '''
        Pick the best settings from grid search evaluation results
        (run after `grid_evaluater`)
    '''

    print 'Pick best settings'

    GT_DATA = IMAGE_DATA
    RAW_SEG_DATA = SEG_DATA

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    best_w = 0
    best_x = 0
    best_r = 0
    best_val = 0

    grid_val = np.zeros    

    for w in range_W:
        Bi_W = w
        for x in range_XY_STD:
            Bi_XY_STD = x
            for r in range_RGB_STD:
                Bi_R_STD = r

                out_dir_name = join( OUT_DIR, 'w-'+str(w) + '_x-'+str(x) + '_r-'+str(r) )
                CRF_SEG_DATA = out_dir_name

                # select the evaluation metric
                if METRIC == 'iou':
                    iou_crf = np.loadtxt( join(out_dir_name,'result_iou_fg_crf.txt'), delimiter=',' )
                    # print '%d  %d  %d ' % (w, x, r)
                    val = np.mean(iou_crf)
                    # print val
                elif METRIC == 'pr':
                    prf_crf = np.loadtxt( join(out_dir_name,'result_pr_fg_crf.txt'), delimiter=',' )
                    val = prf_crf[2]

                if val > best_val:
                    best_val = val
                    best_w = w
                    best_x = x
                    best_r = r

    if METRIC == 'iou':
        print 'Best IOU: %f' % best_val
        print 'Settings: w=%f, x=%f, r=%f' % (best_w, best_x, best_r)
        np.savetxt(join(OUT_DIR,'crf_best_iou_' + str(RUN_NUM) +'.txt'), \
                   [best_val, best_w, best_x, best_r], delimiter=',')

    elif METRIC == 'pr':
        print 'Best f-measure: %f' % best_val
        print 'Settings: w=%f, x=%f, r=%f' % (best_w, best_x, best_r)
        np.savetxt(join(OUT_DIR,'crf_best_pr_' + str(RUN_NUM) +'.txt'), \
                   [best_val, best_w, best_x, best_r], delimiter=',')




if __name__ == '__main__':
    if MODE == 'run':
        grid_runner()
    elif MODE == 'eval':
        grid_evaluater()
    elif MODE == 'pick':
        grid_picker() 

                



