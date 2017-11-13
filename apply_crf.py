#!/home/arunirc/dense-crf/bin/python


'''
    Dense CRF Motion Segmentation refinement
    ----------------------------------------
    - Specify data locations and settings below. 
    - Alternatively, you can call this script from the cmd line and pass the args:
        > python apply_crf.py -i IMAGE_DATASET -o SEG_DATASET -o data/crf-output
    - Optional: modify path to Python interpreter in the first line of this script.

    - This code skips over frames if it finds that it has already been saved on disk.
    - To re-start this code from scratch, you need to completely remove the CRF output folder. 
'''

from __future__ import division

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


IMAGE_DATA = '/data2/arunirc/Research/FlowNet2/flownet2-docker/data/complexBackground/complexBackground-multilabel/'
SEG_DATA = '/data/pbideau/motionSegmentation/Aruni_CRF/complex-background-multi-labels/'
IMAGE_EXT = ['.jpg', '.png']
OUT_DIR = 'data/crf-output/complex-bg'


def parse_input_opts():
    parser = argparse.ArgumentParser(description='Visualize flow')
    parser.add_argument('-i', '--image_data', help='Specify folder containing RGB dataset', \
                            default=IMAGE_DATA)
    parser.add_argument('-s', '--seg_data', help='Specify folder containing original segmentations', \
                            default=SEG_DATA)
    parser.add_argument('-o', '--out_dir', help='Specify output folder for CRF segmentaitons', \
                            default=OUT_DIR)
    parser.add_argument('-d', '--dataset', help='Specify dataset: davis, camo, complex, fbms', \
                            default='davis')
    parser.add_argument('-v', '--viz', help='Save visualized original and CRF segmentations as images', \
                            default=False, action='store_true')

    parser.add_argument('-z', '--slim', help='Compute results only for frames that have a ground-truth annotation', \
                            default=False, action='store_true')
    
    parser.add_argument('-cgw', '--crf_gaussian_weight', help='CRF weight for pairwise Gaussian term', \
                            default=3, type=float)
    parser.add_argument('-cgx', '--gaussian_sx', help='x_stdev for pairwise Gaussian term', \
                            default=3, type=float)
    parser.add_argument('-cbw', '--crf_bilateral_weight', help='CRF weight for pairwise bilateral term', \
                            default=5, type=float)
    parser.add_argument('-cbx', '--bilateral_sx', help='x_stdev for pairwise bilateral term', \
                            default=50, type=float)
    parser.add_argument('-cbc', '--bilateral_color', help='color stdev for pairwise bilateral term', \
                            default=10, type=float)
    parser.add_argument('-mi', '--max_iter', help='Max iters for CRF', \
                            default=5, type=int)

    opts = parser.parse_args()
    opts.image_exts = IMAGE_EXT
    
    # opts.crf_gaussian_weight = float(opts.crf_gaussian_weight)
    # opts.gaussian_sx = float(opts.gaussian_sx)

    # opts.crf_bilateral_weight = float(opts.crf_bilateral_weight)
    # opts.bilateral_sx = float(opts.bilateral_sx)
    # opts.bilateral_color = float(opts.bilateral_color)
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

    # pairwise positional (gaussian) terms
    feats = create_pairwise_gaussian(sdims=(opts.gaussian_sx, opts.gaussian_sx), shape=img.shape[:2])
    crf.addPairwiseEnergy(feats, compat=opts.crf_gaussian_weight,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

    # pairwise bilateral (color + position) terms
    feats = create_pairwise_bilateral(sdims=(opts.bilateral_sx, opts.bilateral_sx), \
                                      schan=(opts.bilateral_color, opts.bilateral_color, opts.bilateral_color),
                                      img=img, chdim=2)
    crf.addPairwiseEnergy(feats, compat=opts.crf_bilateral_weight,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

    # run inference
    Q = crf.inference(5)

    return Q



# ------------------------------------------------------------------------------
def apply_crf_seg(opts):
# ------------------------------------------------------------------------------

    for d in sorted(listdir(opts.image_data)):

        # FBMS videos have inconsistent numbering for frames
        MARPLE_FLAG = False 
        TENNIS_FLAG = False
        
        vid_dir = join(opts.image_data, d)
        if not isdir(vid_dir):
            continue
        print d


        # ----------------------------------------------------------------------
        # Dataset specific hackery
        # ----------------------------------------------------------------------
        if opts.dataset == 'davis':
            seg_dir = join(opts.seg_data, d, 'GroundTruth/objectProb/')

        elif opts.dataset == 'complex':
            seg_dir = join(opts.seg_data, d, 'objectProb/')

        elif opts.dataset == 'camo':
            vid_dir = join(vid_dir, 'frames/')
            seg_dir = join(opts.seg_data, d, 'objectProb/')

        elif opts.dataset == 'fbms':
            seg_dir = join(opts.seg_data, d, 'objectProb/')
        


        # ----------------------------------------------------------------------
        # Corner case: some videos in original dataset may be skipped when 
        #   evaluating the segmentations. We skip these too.
        # ---------------------------------------------------------------------- 
        if not isdir(seg_dir):
            continue
        
        if not os.path.isdir(join(opts.out_dir, d)):
            os.makedirs(join(opts.out_dir, d))
            
        if not os.path.isdir(join(opts.out_dir, d, 'viz')):
            os.makedirs(join(opts.out_dir, d, 'viz'))

        # HACK: ugly hack to handle marple4's weird frame numbering
        if d=='marple4':
            MARPLE_FLAG = True
        if d=='tennis':
            TENNIS_FLAG = True
        
        

        count = 0
        for frame in sorted(listdir(vid_dir)):
            
            if frame.endswith(tuple(opts.image_exts)):

                # --------------------------------------------------------------
                #   Initial checks
                # --------------------------------------------------------------
                count = count + 1

                if MARPLE_FLAG or TENNIS_FLAG:
                    frame_num = str(count)
                else:
                    frame_name = str.split(frame, '.')[0]
                    frame_num = str.split(frame_name, '_')[1]

                # check if output is already computed -- skip
                if isfile(join(opts.out_dir, d, frame_num.zfill(5) + '.mat')):
                    continue

                # if no corresponding ground truth image -- skip
                if opts.slim:
                    f_num = int(frame_num)
                    gt_file = join(opts.image_data, d, 'GroundTruth', str(f_num).zfill(3)+'_gt.png')
                    if not isfile(gt_file):
                        continue
                


                # --------------------------------------------------------------
                # Read Motion Segmentation result (MAT file)
                #   -- assuming that you consistently zero-pad your output seg files
                # --------------------------------------------------------------
                seg_file = join(seg_dir, frame_num.zfill(5) + '.mat')
                print seg_file
                
                if not isfile(seg_file):
                    if count>1:
                        continue  # there is one less seg-file than frames

                mat_data = sio.loadmat(seg_file)
                res = mat_data['objectProb']

                # check `res` dims
                if res.ndim != 3:
                    # only background class -- skip!
                    continue
                    # only background class predicted in raw segmentation: 
                    #   -- stack `(1-res)` onto 3rd axis
                    # res = np.stack((res,1-res), axis=2)  
                
                
                
                # --------------------------------------------------------------
                # Read RGB image
                # --------------------------------------------------------------
                img = imread(join(vid_dir, frame))
                

                # DEBUG
                # from IPython.core.debugger import Tracer; Tracer()()


                # --------------------------------------------------------------
                # Dense CRF inference
                # --------------------------------------------------------------
                # input probabilities (unary terms)
                labels, label_map, n_labels = preprocess_label_scores(res)
                
                Q = get_crf_seg(img, labels, n_labels, opts)
                
                probQ = np.array(Q) # CRF scores into Pia's format
                crf_prob = probQ.reshape((probQ.shape[0], img.shape[0] ,img.shape[1]))
                crf_prob = crf_prob.transpose((1,2,0))
                crf_out_final = np.zeros(res.shape)
                crf_out_final[:,:,label_map] = crf_prob
                

                
                # --------------------------------------------------------------
                # Outputs and visualization
                # --------------------------------------------------------------
                # save as matlab MATLAB-style .mat file
                sio.savemat(join(opts.out_dir, d, frame_num.zfill(5) + '.mat'), \
                            dict(objectProb=crf_out_final))
                
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
                    imsave(join(opts.out_dir, d, 'viz', frame_num+'_raw_crf.png'), \
                                    tiled_img)                


# entry point
if __name__ == '__main__':
    opts = parse_input_opts()
    apply_crf_seg(opts)
    