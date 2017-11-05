# crf-motion-seg


### Setup Dense CRF codebase

We use the [py-dense-crf](https://github.com/lucasb-eyer/pydensecrf) implementation inside a `conda` environment. 

Inside the env:
    
    pip install pydensecrf
    pip install -U setuptools
    pip install -U cython
    pip install numpy matplotlib scipy skimage ipython jupyter


### Apply CRF

The `apply_crf.py` file contains the Python code to apply dense CRF 
on motion segmentation datasets.

The `run_*.sh` shell scripts run the CRF segmentation on various datasets.
The settings and paths should be modified inside these scripts before calling. 

The `apply_crf_image.py` can be called from within a MATLAB script or from the terminal to do CRF segmentation for a single frame. It takes in one image file, one MAT file of raw segmentations and outputs the CRF segmentations in a MAT file at a specified location.

### Setup with MATLAB calls on swarm2 cluster 

The main purpose of `apply_crf_image.py`  is to be able to call it from within a MATLAB loop -- with some file I/O overheads. The MATLAB instance needs to be run inside a conda environment. It can be a bit tricky to set this up on a SLURM cluster.

To setup on **swarm2**:

1) install _miniconda_ (from your home directory on swarm2):

    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
    bash Miniconda2-latest-Linux-x86_64.sh

2) setup the conda environment:

    conda create --prefix ~/dense-crf-conda python=2.7
    source activate /home/arunirc/dense-crf-conda
    pip install pydensecrf     

The last bit takes a long time ... without any output messages!

    conda install numpy
    conda install scipy
    conda install scikit-image


3) then check if this works on a single input:

    wget https://github.com/AruniRC/crf-motion-seg/archive/master.zip
    unzip master.zip
    cd crf-motion-seg-master/

    srun python apply_crf_image.py -i ./samples/bear01/bear01_0002.jpg -s ./samples/bear01/00002.mat -o ./samples/test_seg_2.mat -v

You should see `./samples/viz_raw_crf_bear01_0002.jpg` -- the segmentation visualization image.

Basically, just run your SLURM scripts from inside the conda environment.

