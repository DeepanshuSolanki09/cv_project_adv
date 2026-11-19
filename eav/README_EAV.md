# EAV pipeline (NOT EXECUTED in this project)

The selected dataset (EAV, Lee et al., Sci. Data 11:1026, 2024, DOI 10.1038/s41597-024-03838-4)
is licensed CC BY 4.0 but its files are restricted. Request access with the form linked on
https://doi.org/10.5281/zenodo.13799131 . The code in this folder follows the official layout
(https://github.com/nubcico/EAV) but could not be run because no data were available.

1. `python eav/inventory.py --root /path/to/EAV`  -> results/tables/eav_inventory.csv (actual counts)
2. Feature extraction: `eav/features.py` (EEG band power, speech MFCC); video: `models/video_branch.py`
   (visual only); face: frames from the same video -> `models/branches.py:FaceVGG`.
3. Participant-independent 6-fold CV over the 42 participants (7 test participants per fold,
   5 validation participants inside each training part). The official baseline split is
   WITHIN-subject (first samples of each class for training); it is not used here.
4. Train branches, compute q_m on validation participants, fuse with `fuzzy/fis.py`, evaluate
   with the same functions as `evaluation/fusion_experiments.py`, using REAL paired trials
   (speaking trials: all four branches; listening trials: speech mask a=0).
