# crf-motion-seg


### Setup Dense CRF

We use the [py-dense-crf](https://github.com/lucasb-eyer/pydensecrf) implementation inside a `conda` environment. 

Inside the env:
    
    pip install pydensecrf
    pip install -U setuptools
    pip install -U cython
    pip install numpy matplotlib scipy skimage ipython jupyter


Add Hungarian algorithm solver for evaluation.


### Apply CRF

The `apply_crf.py` file contains the Python code to apply dense CRF 
on motion segmentation datasets.

The `run_*.sh` shell scripts run the CRF segmentation on various datasets.
The settings and paths should be modified inside these scripts before calling. 

