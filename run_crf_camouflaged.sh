# Run CRF on Camouflaged dataset

IMAGE_DATA='/data2/arunirc/Research/FlowNet2/flownet2-docker/data/CamAnimal/CamouflagedAnimalDataset/'
SEG_DATA='/data/arunirc/Research/dense-crf-data/our-modifiedObjPrior/camouflaged-animals/'
OUT_DIR='data/crf-output-modifiedObjPrior-v3-improved/camouflaged-animals'

# default crf params
python apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'camo'

# # crf params
# W=15
# X=40
# R=5

# python apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'camo' -cbw $W -cbx $X -cbc $R


# crf params
W=10
X=40
R=7

python apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'camo' -cbw $W -cbx $X -cbc $R