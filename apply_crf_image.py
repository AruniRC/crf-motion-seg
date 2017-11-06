#!/home/arunirc/dense-crf/bin/python

'''
    Dense CRF Motion Segmentation for SINGLE FRAME
    -----------------------------------------------
    - Specify data locations and settings below.

    - Usage:
        python apply_crf_image.py -i INPUT_IMAGE_FILE -s INPUT_SEG_MAT_FILE -o OUTPUT_MAT_FILE

    - Optimal CRF flags cross-validated on FBMS Training set:
        (append these to the usual call to the script shown above)
        '... -cbw 15 -cbx 40 -cbc 5'

    - Requirements:
        * install Anaconda
        * Create a conda environment:
            conda create --prefix ~/dense-crf-debug python=2.7
        * Enter that environment:
            source activate ~/dense-crf-debug
        * Install a bunch of things:
            pip install pydensecrf
            conda install numpy
            conda install scipy
            conda install scikit-image
        * Run MATLAB and call this script from inside the conda env.

'''

from __future__ import division

IMAGE_PATH = './samples/bear01/bear01_0002.jpg'
SEG_PATH = './samples/bear01/00002.mat'
OUT_PATH = './samples/bear01/crf_d1_00002.mat'


import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

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

import sys


def parse_input_opts():
    parser = argparse.ArgumentParser(description='Run dense-crf on single frame motion segmentations')
    parser.add_argument('-i', '--image', help='Specify path to RGB image', \
                            default=IMAGE_PATH)
    parser.add_argument('-s', '--seg', help='Specify path to raw segmentations (HxWxK .mat file)', \
                            default=SEG_PATH)
    parser.add_argument('-o', '--out', help='Specify output path for CRF segmentations', \
                            default=OUT_PATH)
    parser.add_argument('-v', '--viz', help='Save visualized original and CRF segmentations as images', \
                            default=False, action='store_true')

    parser.add_argument('-cgw', '--crf_gaussian_weight', help='CRF weight for pairwise Gaussian term', \
                            default=3)
    parser.add_argument('-cgx', '--gaussian_sx', help='x_stdev for pairwise Gaussian term', \
                            default=3)
    parser.add_argument('-cbw', '--crf_bilateral_weight', help='CRF weight for pairwise bilateral term', \
                            default=5)
    parser.add_argument('-cbx', '--bilateral_sx', help='x_stdev for pairwise bilateral term', \
                            default=50)
    parser.add_argument('-cbc', '--bilateral_color', help='color stdev for pairwise bilateral term', \
                            default=10)
    parser.add_argument('-mi', '--max_iter', help='Max iters for CRF', \
                            default=5)

    opts = parser.parse_args()
    return opts



def preprocess_label_scores(res):
    '''
        labels, label_map, n_labels = preprocess_label_scores(res)
            Pre-processes the objectness scores from Pia's code.
            Only the labels with non-zero presence are retained.
            So the `res` HxWxK tensor becomes HxWxC, where C<=K.
            Then it is re-arranged to CxHxW.
    '''

    # handle NaN in `res` - set to 0
    res_is_nan = np.isnan(res)
    if np.sum(res_is_nan.astype(int)) > 0:
        res[res_is_nan] = 0

    res_sum = np.squeeze(np.sum(np.sum(res, axis=0), axis=0))
    label_map = np.array(np.where(res_sum > 0)[0])
    labels = np.squeeze(res[:,:,label_map]) # keep non-zero presence labels
    # assert np.abs(np.sum(labels) - res.shape[0]*res.shape[1]) < np.finfo(np.float32).eps # check probability
    labels = np.transpose(labels, (2,0,1)) # C x H x W
    n_labels = len(label_map)
    return labels, label_map, n_labels


def get_crf_seg(img, labels, n_labels, opts):
    '''
        crf_out_final = get_crf_seg(img, labels, n_labels)
    '''
    crf = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)
    U = unary_from_softmax(labels)
    crf.setUnaryEnergy(U)

    feats = create_pairwise_gaussian(sdims=(opts.gaussian_sx, opts.gaussian_sx), shape=img.shape[:2])
    crf.addPairwiseEnergy(feats, compat=opts.crf_gaussian_weight,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

    feats = create_pairwise_bilateral(sdims=(opts.bilateral_sx, opts.bilateral_sx), \
                                      schan=(opts.bilateral_color, opts.bilateral_color, opts.bilateral_color),
                                      img=img, chdim=2)
    crf.addPairwiseEnergy(feats, compat=opts.crf_bilateral_weight,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = crf.inference(5)

    return Q



def apply_crf_seg_img(opts):

    # Read Motion Segmentation result (MAT file)
    seg_file = opts.seg
    
    if not isfile(seg_file):
        print 'Input motion segmentation file not found.'
        return

    mat_data = sio.loadmat(seg_file)
    res = mat_data['objectProb']

    # check `res` dims
    if res.ndim != 3:
        # if no foreground object, then save the original and exit
        sio.savemat(opts.out, dict(objectProb=res))
        return

    # Read RGB image
    if not isfile(opts.image):
        print 'Input RGB image file not found.'
        return
    img = imread(opts.image)


    # Create unary terms
    labels, label_map, n_labels = preprocess_label_scores(res)
    

    # Dense CRF inference
    Q = get_crf_seg(img, labels, n_labels, opts)
    

    # CRF scores into Pia's format
    probQ = np.array(Q) 
    crf_prob = probQ.reshape((probQ.shape[0], img.shape[0] ,img.shape[1]))
    crf_prob = crf_prob.transpose((1,2,0))
    crf_out_final = np.zeros(res.shape)
    crf_out_final[:,:,label_map] = crf_prob

    
    # Save as matlab MATLAB-style .mat file
    sio.savemat(opts.out, dict(objectProb=crf_out_final))
    

    # Optional: save labelings as image
    if opts.viz:
        # CRF MAP labels
        MAP = np.argmax(Q, axis=0) 
        crf_map_img = MAP.reshape((img.shape[0], img.shape[1]))
        crf_label_img = label_map[crf_map_img] # CRF labels --> instance labels
        
        # side-by-side visualizations as RGB overlays
        crf_rgb = color.label2rgb(crf_label_img, img, bg_label=0)
        res_label =  np.argmax(res, axis=2)
        res_rgb = color.label2rgb(res_label, img, bg_label=0)
        tiled_img = np.concatenate((res_rgb, \
                        np.zeros([res_rgb.shape[0],10,3]), \
                        crf_rgb), axis=1)

        # save tiled image
        out_dir, _ = os.path.split(opts.out)
        _, fn = os.path.split(opts.image)
        imsave(join(out_dir, 'viz_raw_crf_'+fn), tiled_img)


# entry point
if __name__ == '__main__':

    opts = parse_input_opts()
    apply_crf_seg_img(opts)
    