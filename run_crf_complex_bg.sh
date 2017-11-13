# 

# Run CRF on Complex Background dataset

IMAGE_DATA='/data2/arunirc/Research/FlowNet2/flownet2-docker/data/complexBackground/complexBackground-multilabel/'
SEG_DATA='/data/arunirc/Research/dense-crf-data/our-modifiedObjPrior/complex-background-multi-labels/'
OUT_DIR='data/crf-output-modifiedObjPrior-v3-improved/complex-bg'

# default CRF params
# ./apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'complex' -v

# # crf params
# W=15
# X=40
# R=5

# ./apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'complex' -v -cbw $W -cbx $X -cbc $R

# crf params - improved for modifiedObjPrior
W=10
X=40
R=7

./apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'complex' -v -cbw $W -cbx $X -cbc $R