# 

# Run CRF on Complex Background dataset

IMAGE_DATA='/data2/arunirc/Research/FlowNet2/flownet2-docker/data/complexBackground/complexBackground-multilabel/'
SEG_DATA='data/our/complex-background-multi-labels/'
OUT_DIR='data/crf-output/complex-bg'
./apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'complex' -v