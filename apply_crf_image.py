#!/home/arunirc/dense-crf/bin/python

'''
    Dense CRF Motion Segmentation for SINGLE FRAME
    -----------------------------------------------
    - Specify data locations and settings below. 
    - Alternatively, you can call this script from the cmd line and pass the args:
        > python apply_crf.py -i IMAGE_DATASET -o SEG_DATASET -o data/crf-output
    - Optional: modify path to Python interpreter in the first line of this script.
'''
IMAGE_PATH = '/data2/arunirc/Research/FlowNet2/flownet2-docker/data/complexBackground/complexBackground-multilabel/'
SEG_PATH = '/data/pbideau/motionSegmentation/Aruni_CRF/complex-background-multi-labels/'
IMAGE_EXT = ['.jpg', '.png']
OUT_PATH = 'data/crf-output/complex-bg'


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

import traceback
import warnings
import sys


def parse_input_opts():
    parser = argparse.ArgumentParser(description='Run dense-crf on single frame motion segmentations')
    parser.add_argument('-i', '--image', help='Specify path to RGB image', \
                            default=IMAGE_DATA)
    parser.add_argument('-s', '--seg', help='Specify path to raw segmentations (HxWxK .mat file)', \
                            default=SEG_DATA)
    parser.add_argument('-o', '--out', help='Specify output path for CRF segmentations', \
                            default=OUT_DIR)
    parser.add_argument('-v', '--viz', help='Save visualized original and CRF segmentations as images', \
                            default=False, action='store_true')
    parser.add_argument('-cgw', '--crf_gaussian_weight', help='CRF weight for pairwise Gaussian term', \
                            default=3)
    parser.add_argument('-cbw', '--crf_bilateral_weight', help='CRF weight for pairwise bilateral term', \
                            default=5)
    parser.add_argument('-bsx', '--bilateral_sx', help='x_stdev for pairwise bilateral term', \
                            default=50)
    parser.add_argument('-bsy', '--bilateral_sy', help='y_stdev for pairwise bilateral term', \
                            default=50)
    parser.add_argument('-bsc', '--bilateral_color', help='CRF weight for pairwise bilateral term', \
                            default=10)
    opts = parser.parse_args()
    opts.image_exts = IMAGE_EXT
    opts.crf_gaussian_weight = int(opts.crf_gaussian_weight)
    opts.crf_bilateral_weight = int(opts.crf_bilateral_weight)
    opts.bilateral_sx = int(opts.bilateral_sx)
    opts.bilateral_sy = int(opts.bilateral_sy)
    opts.bilateral_color = int(opts.bilateral_color)
    return opts


# DEBUG
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    # diagnose warning tracebacks
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback # // Uncomment to traceback warnings



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
    # from IPython.core.debugger import Tracer; Tracer()()
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
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    crf.addPairwiseEnergy(feats, compat=opts.crf_gaussian_weight,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats = create_pairwise_bilateral(sdims=(opts.bilateral_sx, opts.bilateral_sy), \
                                      schan=(opts.bilateral_color, opts.bilateral_color, opts.bilateral_color),
                                  img=img, chdim=2)
    crf.addPairwiseEnergy(feats, compat=opts.crf_bilateral_weight,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = crf.inference(5)

    return Q




def apply_crf_seg(opts):

    
                
                # Read Motion Segmentation result (MAT file)
                #   -- assuming that you consistently zero-pad your output seg files
                seg_file = join(seg_dir, frame_num.zfill(5) + '.mat')
                
                if not isfile(seg_file):
                    print 'Input motion segmentation file not found.'
                    return

                mat_data = sio.loadmat(seg_file)
                res = mat_data['objectProb']

                # check `res` dims
                if res.ndim != 3:
                    from IPython.Debugger import Tracer; Tracer()()
                    # only background class predicted: save as-is w/o CRF
                    sio.savemat(join(opts.out_dir, d, frame_num.zfill(5) + '.mat'), \
                            dict(objectProb=res))
                    continue 
                
                # read RGB image
                img = imread(join(vid_dir, frame))

                # DEBUG
                # from IPython.core.debugger import Tracer; Tracer()()

                # input probabilities (unary terms)
                labels, label_map, n_labels = preprocess_label_scores(res)
                
                # Dense CRF inference
                Q = get_crf_seg(img, labels, n_labels, opts)
                
                probQ = np.array(Q) # CRF scores into Pia's format
                crf_prob = probQ.reshape((probQ.shape[0], img.shape[0] ,img.shape[1]))
                crf_prob = crf_prob.transpose((1,2,0))
                crf_out_final = np.zeros(res.shape)
                crf_out_final[:,:,label_map] = crf_prob

                
                # save as matlab MATLAB-style .mat file
                sio.savemat(join(opts.out_dir, d, frame_num.zfill(5) + '.mat'), \
                            dict(objectProb=crf_out_final))
                
                if opts.viz:
                    # CRF MAP labels
                    MAP = np.argmax(Q, axis=0) 
                    crf_map_img = MAP.reshape((img.shape[0], img.shape[1]))
                    crf_label_img = label_map[crf_map_img] # CRF labels --> instance labels
                    
                    # side-by-side visualizations as RGB overlays
                    crf_rgb = color.label2rgb(crf_label_img, img)
                    res_label =  np.argmax(res, axis=2)
                    res_rgb = color.label2rgb(res_label, img)
                    tiled_img = np.concatenate((res_rgb, \
                                    np.zeros([res_rgb.shape[0],10,3]), \
                                    crf_rgb), axis=1)
                    imsave(join(opts.out_dir, d, 'viz', frame_num+'_raw_crf.png'), \
                                    tiled_img)                


# entry point
if __name__ == '__main__':

    opts = parse_input_opts()
    apply_crf_seg(opts)
    