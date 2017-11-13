# Run CRF on FBMS Training set data

IMAGE_DATA='/data2/arunirc/Research/FlowNet2/flownet2-docker/data/FBMS/Testset/'
SEG_DATA='/data/arunirc/Research/dense-crf-data/our-modifiedObjPrior/FBMS/Testset/'
OUT_DIR='data/crf-output-modifiedObjPrior-v3-improved/FBMS-test'

# defaults
python apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'fbms'

# # crf params
# W=15
# X=40
# R=5

# python apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'fbms' -cbw $W -cbx $X -cbc $R


# crf params
W=10
X=40
R=7

python apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'fbms' -cbw $W -cbx $X -cbc $R