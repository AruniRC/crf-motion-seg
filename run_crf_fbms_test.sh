# Run CRF on FBMS Training set data

IMAGE_DATA='/data2/arunirc/Research/FlowNet2/flownet2-docker/data/FBMS/Testset/'
SEG_DATA='data/our/FBMS/Testset/'
OUT_DIR='data/crf-output-v2/FBMS-test'

# crf params
W=15
X=40
R=5

python apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'fbms' -cbw $W -cbx $X -cbc $R