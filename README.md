# Fuzzy Logic Based Multimodal Emotion Recognition (EEG, Face, Speech, Video)

## Status (read first)
| Category | Content |
|---|---|
| A. Completed | Dataset selection (EAV); four branches implemented; availability mask, reliability weights; Mamdani, Sugeno-0, constrained Sugeno-1, Tsukamoto; executed engine tests (16/16 pass); missing-modality weight table; preliminary independent-branch experiments (EEG: Feeling Emotions, Face: FER2013, Speech: RAVDESS audio) |
| B. Partial | Video branch (visual only; verified on synthetic frames, not trained); EAV pipeline code (`eav/`, not run) |
| C. NOT EXECUTED | Every EAV experiment (inventory, training, multimodal fusion, fuzzy comparison, missing-modality performance) |
| D. Next steps | Request EAV access (form on https://doi.org/10.5281/zenodo.13799131), then run `eav/` pipeline |

No result in this folder is evidence of multimodal recognition. `results/tables/appendix_fusion_stress_test_pseudo_paired.csv` uses artificial cross-dataset pairings and is a code stress test only.

## Install
```
pip install -r requirements.txt
```

## Executed commands
```
python tests/test_fusion.py                 # engine checks -> results/tables/fusion_engine_tests.csv
python preprocessing/speech_features.py     # preliminary RAVDESS MFCC maps (needs RAVDESS audio)
python training/train_unimodal.py eeg       # seed 42; also: eeg_all, speech, face
python evaluation/fusion_experiments.py     # appendix stress test only
python visualizations/make_figures.py
```
Data used for the preliminary experiments: EEG Brainwave Feeling Emotions (emotions.csv), FER2013 (35,887 images), RAVDESS speech audio (1,440 files). Paths are set at the top of each script (`/home/claude/data/...`); change them to your location.

## Selected dataset: EAV (NOT EXECUTED)
See `eav/README_EAV.md`. Lee et al., Scientific Data 11:1026 (2024), DOI 10.1038/s41597-024-03838-4. Official code: https://github.com/nubcico/EAV.

## Layout
- `fuzzy/fis.py` - masks, weights, UNAVAILABLE state, Mamdani / Sugeno (0, constrained 1) / Tsukamoto
- `models/branches.py` - EEG-FNN, Face VGG, Speech CNN, ResNet-18 extractor; `models/video_branch.py` - visual-only video branch
- `eav/` - EAV inventory and feature extraction (not run)
- `results/tables/*.csv`, `results/graphs/*.pdf` - generated outputs
- `report/final_report.pdf`
