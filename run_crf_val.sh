# Run CRF on FBMS Training set data

IMAGE_DATA='/data/arunirc/Research/dense-crf-data/training_subset/'
SEG_DATA='data/our/FBMS/Trainingset/'
OUT_DIR='/data/arunirc/Research/dense-crf-data/FBMS-train-subset-01'
python -m pdb apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'fbms'