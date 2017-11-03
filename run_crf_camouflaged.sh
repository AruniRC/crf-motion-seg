# Run CRF on Camouflaged dataset

IMAGE_DATA='/data2/arunirc/Research/FlowNet2/flownet2-docker/data/CamAnimal/CamouflagedAnimalDataset/'
SEG_DATA='data/our/camouflaged-animals/'
OUT_DIR='data/crf-output/camouflaged-animals'
python -m pdb apply_crf.py -i $IMAGE_DATA -s $SEG_DATA -o $OUT_DIR -d 'camo' -v