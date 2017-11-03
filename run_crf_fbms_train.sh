# Run CRF on FBMS Training set data

IMAGE_DATA='/data2/arunirc/Research/FlowNet2/flownet2-docker/data/FBMS/Trainingset/'
SEG_DATA='data/our/FBMS/Trainingset/'
OUT_DIR='data/crf-output/FBMS-train'
python -m pdb apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'fbms'