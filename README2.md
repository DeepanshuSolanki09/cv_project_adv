# Comprehensive Technical Guide & Extended Documentation (README 2)
## Fuzzy Logic-Based Multimodal Emotion Recognition (EEG, Face, Speech, Video)

This document provides a comprehensive technical reference and developer guide for the fuzzy logic-based multimodal emotion recognition (MER) framework implemented in this repository.

---

## Table of Contents
1. [Overview & Objectives](#1-overview--objectives)
2. [Theoretical Framework & Mathematics](#2-theoretical-framework--mathematics)
   - [Quality-Based Reliability Weighting](#quality-based-reliability-weighting)
   - [Availability Masking & Graceful Degradation](#availability-masking--graceful-degradation)
   - [Crisp Feature Formulation (Support & Consistency)](#crisp-feature-formulation-support--consistency)
   - [Membership Functions & 9-Rule Base](#membership-functions--9-rule-base)
   - [Fuzzy Inference Systems (FIS)](#fuzzy-inference-systems-fis)
3. [Unimodal Neural & Statistical Branches](#3-unimodal-neural--statistical-branches)
   - [EEG Branch (`EEGFNN`)](#eeg-branch-eegfnn)
   - [Facial Expression Branch (`FaceVGG`)](#facial-expression-branch-facevgg)
   - [Speech Branch (`SpeechCNN`)](#speech-branch-speechcnn)
   - [Video Branch (Visual Spatiotemporal)](#video-branch-visual-spatiotemporal)
4. [Dataset Specifications & Status](#4-dataset-specifications--status)
   - [Preliminary Datasets (Executed)](#preliminary-datasets-executed)
   - [Target EAV Dataset (Staged)](#target-eav-dataset-staged)
5. [Step-by-Step Setup & Execution](#5-step-by-step-setup--execution)
6. [Benchmarking, Tests & Results](#6-benchmarking-tests--results)
7. [Repository Map & File Index](#7-repository-map--file-index)

---

## 1. Overview & Objectives

Multimodal Emotion Recognition (MER) combines complementary physiological and behavioral signals to classify affective states. Traditional deep fusion approaches (e.g., feature concatenation or joint embeddings) suffer from three critical weaknesses in real-world deployments:
1. **Black-Box Decision Making**: Lack of interpretability regarding which modality drove the emotional classification.
2. **Missing Modality Vulnerability**: Complete failure or dramatic degradation when a sensor fails (e.g., disconnected EEG electrode, face occluded, microphone muted).
3. **Overconfidence Under Inter-Modality Disagreement**: Inability to explicitly model conflict when modalities indicate contradictory emotions (e.g., smiling face but stressed physiological EEG).

This framework resolves these issues through **decision-level fuzzy fusion**, combining:
- A uniform 3-class emotional taxonomy: **Negative (0)**, **Neutral (1)**, and **Positive (2)**.
- Modality reliability estimation based on chance-corrected validation macro-F1 scores.
- Dynamic missing-modality handling via boolean availability masking.
- Four distinct fuzzy inference paradigms sharing identical input membership and rule bases: **Mamdani**, **Zero-Order Sugeno (Sugeno-0)**, **Constrained First-Order Sugeno (Sugeno-1)**, and **Tsukamoto**.

```
                ┌────────────────────────────────────────────────────────┐
                │             Unimodal Classifiers (4 Branches)          │
                │     EEG (MLP), Face (VGG), Speech (CNN), Video (SVM)   │
                └───────────────────────────┬────────────────────────────┘
                                            │
                                            ▼ Posterior Probabilities P_{m, c} ∈ [0, 1]
┌────────────────────────┐                  │
│ Modality Qualities q_m │──────────┐       │
└────────────────────────┘          ▼       ▼
┌────────────────────────┐     ┌────────────────────────────────────────┐
│ Availability Mask A    │────▶│ Dynamic Reliability Weights W (N, M)   │
└────────────────────────┘     └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │ Crisp Input Extraction:                │
                               │  - Weighted Support S_c ∈ [0, 1]       │
                               │  - Inter-Modality Consistency G_c      │
                               └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │ Fuzzification & Rule Evaluation (9 R)  │
                               │ Triangular MFs (Low, Med, High)        │
                               │ Firing Strength α_k = min(μ_S, μ_G)    │
                               └───────────────────┬────────────────────┘
                                                   │
                        ┌──────────────────────────┴──────────────────────────┐
                        ▼                          ▼                          ▼
               [ Mamdani Engine ]         [ Sugeno-0 / Sugeno-1 ]    [ Tsukamoto Engine ]
               Centroid Defuzz.           Linear/Constant Output     Monotonic Ramp Inv.
                        │                          │                          │
                        └──────────────────────────┬──────────────────────────┘
                                                   │
                                                   ▼
                                  Belief Scores B_c ∈ [0, 1]
                                  Predicted Class ŷ ∈ {0, 1, 2}
                                  (or UNAVAILABLE = -1)
```

---

## 2. Theoretical Framework & Mathematics

All fuzzy logic routines reside in `fuzzy/fis.py`.

### Quality-Based Reliability Weighting
To prevent poorly performing or untrained modalities from corrupting the final decision, each modality $m \in \{\text{EEG}, \text{Face}, \text{Speech}, \text{Video}\}$ is assigned an intrinsic quality metric $q_m$ based on its validation macro-F1 score:

$$q_m = \max\left(\frac{\text{Macro-F1}_m - \frac{1}{C}}{1 - \frac{1}{C}},\, \epsilon\right)$$

- $C = 3$ (number of classes).
- A modality performing at random chance ($1/3 \approx 0.333$) receives $q_m \le \epsilon = 10^{-3}$.
- A model with no trained weights (e.g., Video in the preliminary experiments) is assigned $q_m = 0.0$.

### Availability Masking & Graceful Degradation
Let $a_{n, m} \in \{0, 1\}$ denote the availability of modality $m$ for sample $n$. The effective fusion weights $w_{n, m}$ are computed as:

$$w_{n, m} = \frac{a_{n, m} q_m}{\sum_{k=1}^M a_{n, k} q_k}$$

- If $\sum_{k=1}^M a_{n, k} q_k = 0$ (all sensors missing or untrained), the system outputs `UNAVAILABLE` (`-1`) with `valid = False`, preventing any zero-division exception.
- When valid, $\sum_{m=1}^M w_{n, m} = 1$.

### Crisp Feature Formulation (Support & Consistency)
For each sample $n$ and emotion class $c \in \{0, 1, 2\}$, two crisp inputs are computed from the unimodal class probabilities $p_{m, c}$:

1. **Weighted Support ($S_c$)**:
   $$S_c = \sum_{m=1}^M w_m p_{m, c} \quad \in [0, 1]$$
   Measures the overall confidence across all available modalities for class $c$.

2. **Weighted Dispersion ($D_c$)**:
   $$D_c = \sqrt{\sum_{m=1}^M w_m (p_{m, c} - S_c)^2} \quad \in [0, 0.5]$$
   Quantifies variance or disagreement between sensors regarding class $c$.

3. **Modality Consistency ($G_c$)**:
   $$G_c = \text{clip}(1.0 - 2.0 \cdot D_c,\, 0.0,\, 1.0) \quad \in [0, 1]$$
   - $G_c = 1.0$: Absolute consensus across all modalities.
   - $G_c = 0.0$: Maximum possible disagreement.

### Membership Functions & 9-Rule Base
Both $S_c$ and $G_c$ are fuzzified into three linguistic variables (`Low`, `Medium`, `High`) using triangular membership functions $\mu(x; a, b, c)$:

$$\mu(x; a, b, c) = \max\left(0, \min\left(\frac{x - a}{b - a}, \frac{c - x}{c - b}\right)\right)$$

Default boundary parameters:
- **Support ($S$)**: Low = `(0.0, 0.0, 0.5)`, Medium = `(0.15, 0.5, 0.85)`, High = `(0.5, 1.0, 1.0)`
- **Consistency ($G$)**: Low = `(0.0, 0.0, 0.6)`, Medium = `(0.3, 0.65, 1.0)`, High = `(0.7, 1.0, 1.0)`

The rule base maps the combinations of $S$ and $G$ to belief levels:

| Rule | Antecedent: Support ($S$) | Antecedent: Consistency ($G$) | Consequent: Output Level | Anchor Value |
| :---: | :---: | :---: | :---: | :---: |
| **R1** | Low | Low | Very Low (VL) | 0.00 |
| **R2** | Low | Medium | Very Low (VL) | 0.00 |
| **R3** | Low | High | Very Low (VL) | 0.00 |
| **R4** | Medium | Low | Low (L) | 0.25 |
| **R5** | Medium | Medium | Medium (M) | 0.50 |
| **R6** | Medium | High | Medium (M) | 0.50 |
| **R7** | High | Low | Medium (M) | 0.50 |
| **R8** | High | Medium | High (H) | 0.75 |
| **R9** | High | High | Very High (VH) | 1.00 |

Rule firing strength is evaluated with the Zadeh t-norm (min):
$$\alpha_k = \min(\mu_{S, k}(S_c), \mu_{G, k}(G_c))$$

### Fuzzy Inference Systems (FIS)

1. **Mamdani**:
   - Output membership functions are triangular sets centered at anchor values.
   - Implication: $\mu_{\text{out}, k}(z) = \min(\alpha_k, \mu_{\text{term}, k}(z))$.
   - Aggregation: $\mu_{\text{agg}}(z) = \max_k \mu_{\text{out}, k}(z)$.
   - Defuzzification: Centroid $\displaystyle B_c = \frac{\int z \mu_{\text{agg}}(z) dz}{\int \mu_{\text{agg}}(z) dz}$ evaluated over a 501-point discrete grid in chunks to optimize memory.

2. **Sugeno-0 (Zero-Order)**:
   - Output consequents are constant anchors $z_k \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$.
   - Defuzzification: $\displaystyle B_c = \frac{\sum_k \alpha_k z_k}{\sum_k \alpha_k}$.

3. **Constrained Sugeno-1 (First-Order)**:
   - Consequents are linear equations: $z_k = p_k S_c + q_k G_c + r_k$.
   - **Optimization with Box Constraints**: Parameters $(p_k, q_k, r_k)$ are fitted via SLSQP quadratic programming on validation data with strict inequality constraints:
     $$0 \le z_k(s, g) \le 1 \quad \forall (s, g) \in \{(0,0), (1,0), (0,1), (1,1)\}$$
     Because $z_k$ is linear, enforcing bounds at the four vertices of the unit square guarantees $z_k \in [0, 1]$ over the entire domain $[0, 1]^2$, completely preventing negative belief scores.

4. **Tsukamoto**:
   - Consequent sets are strictly monotonic ramps:
     - Levels `VL`, `L`: Decreasing ramps.
     - Levels `H`, `VH`: Increasing ramps.
     - Level `M`: Narrow ramp spanning $[0.4, 0.5]$.
   - Each rule directly inverts its firing strength: $z_k = \mu_k^{-1}(\alpha_k) = \text{start}_k + \alpha_k (\text{end}_k - \text{start}_k)$.
   - Defuzzification: Weighted average $\displaystyle B_c = \frac{\sum_k \alpha_k z_k}{\sum_k \alpha_k}$.

---

## 3. Unimodal Neural & Statistical Branches

All model architectures are defined in `models/branches.py` and `models/video_branch.py`.

### EEG Branch (`EEGFNN`)
- **Input**: 750 Fast Fourier Transform (FFT) features extracted from EEG brainwave recordings.
- **Architecture**:
  - `Linear(750, 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)`
  - `Linear(256, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)`
  - `Linear(128, 3)` (Logits)
- **Parameters**: 226,051 trainable weights.
- **Normalization**: Per-window standard scaling fitted strictly on training splits.

### Facial Expression Branch (`FaceVGG`)
- **Input**: $48 \times 48$ grayscale face images.
- **Architecture**:
  - Stage 1: `[Conv2d(1, 32, 3) - BatchNorm2d - ReLU]` $\times 2$ $\rightarrow$ `MaxPool2d(2)`
  - Stage 2: `[Conv2d(32, 64, 3) - BatchNorm2d - ReLU]` $\times 2$ $\rightarrow$ `MaxPool2d(2)`
  - Stage 3: `[Conv2d(64, 128, 3) - BatchNorm2d - ReLU]` $\times 2$ $\rightarrow$ `DropBlock2D(block=3, keep=0.9)` $\rightarrow$ `MaxPool2d(2)`
  - Dense Head: `Flatten` $\rightarrow$ `Linear(128 * 6 * 6, 256)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(256, 3)`
- **DropBlock Regularization**: Rather than removing isolated random units (which allows neighboring correlated spatial pixels to compensate), DropBlock drops contiguous $3 \times 3$ feature regions, preventing the CNN from memorizing local artifacts.

### Speech Branch (`SpeechCNN`)
- **Input**: $40 \times 128$ normalized log-Mel Spectrogram (MFCC) feature maps.
- **Audio Preprocessing** (`preprocessing/speech_features.py`):
  - Sample rate: 22,050 Hz, duration: 3.0s, window: Hamming, FFT size: 1024, hop length: 512, Mel filters: 64, MFCC coefficients: 40.
  - Normalization: Per-utterance **Cepstral Mean and Variance Normalization (CMVN)** ensuring zero information leakage across data splits.
- **Architecture**:
  - `Conv-BN-ReLU(1, 32)` $\rightarrow$ `MaxPool2d(2)`
  - `Conv-BN-ReLU(32, 64)` $\rightarrow$ `MaxPool2d(2)`
  - `Conv-BN-ReLU(64, 128)` $\rightarrow$ `MaxPool2d(2)` $\rightarrow$ `AdaptiveAvgPool2d(1)`
  - Dense Head: `Linear(128, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Linear(128, 3)`
- **Augmentation**: Random time shifts ($\pm 12$ steps) + SpecAugment-style zero-masking on time bands.

### Video Branch (Visual Spatiotemporal)
- **Design Philosophy**: Visual-only to avoid double-counting acoustic evidence processed by the speech branch.
- **Pipeline**:
  1. Uniform extraction of $T = 16$ frames from video.
  2. Face detection and bounding box crop via OpenCV Haar Cascade (`models/assets/haarcascade_frontalface_default.xml`).
  3. Pretrained ResNet-18 backbone (frozen) produces 512-dimensional embeddings $f_t$ per frame.
  4. Temporal Pooling: Concatenation of:
     - Appearance Mean: $\frac{1}{T}\sum_t f_t$ (512-d)
     - Temporal Variability: $\text{std}_t(f_t)$ (512-d)
     - Dynamics/Motion: $\frac{1}{T-1}\sum_{t=1}^{T-1} |f_t - f_{t-1}|$ (512-d)
     - Total descriptor: 1536-d vector.
  5. Classifier: Scikit-learn RBF Support Vector Classifier (`SVC(C=10.0, kernel='rbf', probability=True)`) with Platt scaling.

---

## 4. Dataset Specifications & Status

### Preliminary Datasets (Executed)
Because no single public dataset with synchronized, uncorrupted recordings for all four modalities was immediately unencumbered, preliminary model development utilized three well-established benchmarks:

| Modality | Dataset | Instances | Split Strategy | Test Accuracy | Macro-F1 | Validation Macro-F1 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **EEG** | Bird "Feeling Emotions" (`emotions.csv`) | 2,132 | Stratified 60/20/20 | 94.85% | 0.9485 | 0.9530 |
| **Face** | FER2013 (3-class mapping) | 35,887 | Official Train / Val / Test | 72.55% | 0.7167 | 0.7046 |
| **Speech** | RAVDESS Audio (3-class mapping) | 1,440 | Actor-Independent 4-Fold CV | 64.66% | 0.5873 | 0.6359 |
| **Video** | None (Synthetic unit test only) | — | Weight masked ($w = 0$) | — | — | 0.0000 |

### Target EAV Dataset (Staged)
The production target for this framework is the **EAV Dataset** (Lee et al., *Scientific Data* 11:1026, 2024; DOI: 10.1038/s41597-024-03838-4):
- 42 participants performing speaking and listening emotional tasks across 5 categories (`Neutral`, `Anger`, `Happiness`, `Sadness`, `Calmness`).
- Synchronized raw 30-channel EEG, vocal audio, and high-definition facial video.
- All ingestion scripts are staged under `eav/`:
  - `eav/inventory.py`: Scans data directory and builds trial rosters.
  - `eav/features.py`: Computes 0.3–50 Hz 4th-order Butterworth bandpass filtered log band power across 5 bands (delta, theta, alpha, beta, gamma) for EEG, and 3-second segmented MFCCs for speech.

---

## 5. Step-by-Step Setup & Execution

### Prerequisites & Installation
Ensure Python 3.9+ and PyTorch are available:

```bash
git clone https://github.com/anujbis2006/cv_project_adv.git
cd cv_project_adv
pip install -r requirements.txt
```

### 1. Verify Fuzzy Engine & Constraints
Run the 16 unit tests verifying boundary values, zero divisions, Sugeno-1 constraints, and single-modality equivalence:

```bash
python tests/test_fusion.py
```
*Expected Output: `16/16 passed`, writing `results/tables/fusion_engine_tests.csv`.*

### 2. Audio Preprocessing (RAVDESS)
Extract 40 $\times$ 128 MFCC spectrogram representations:

```bash
python preprocessing/speech_features.py
```

### 3. Unimodal Model Training
Train individual branches with early stopping and automatic checkpoint recovery:

```bash
# Train EEG branch
python training/train_unimodal.py eeg

# Train Face branch (FER2013)
python training/train_unimodal.py face

# Train Speech branch (RAVDESS 4-fold CV)
python training/train_unimodal.py speech
```

### 4. Multimodal Fusion Evaluation
Execute the fusion experiment suite comparing Mean, Weighted-Mean, Mamdani, Sugeno-0, Sugeno-1, and Tsukamoto under full and missing-modality scenarios:

```bash
python evaluation/fusion_experiments.py
```

### 5. Generate Figures & Visualizations
Recreate all diagnostic plots and training curves:

```bash
python visualizations/make_figures.py
```

---

## 6. Benchmarking, Tests & Results

### Unit Test Suite Results (`fusion_engine_tests.csv`)
All 16 engine assertions pass cleanly:
1. `weights are zero for every unavailable modality` $\rightarrow$ **PASS**
2. `weights are zero for a modality without a trained model (q=0)` $\rightarrow$ **PASS**
3. `weights sum to 1 whenever a trained modality is available` $\rightarrow$ **PASS**
4. `rows with no trained modality are flagged unavailable (no division by zero)` $\rightarrow$ **PASS**
5. `all-unavailable input returns zero weights and valid=False` $\rightarrow$ **PASS**
6. `Mamdani: label UNAVAILABLE exactly where no modality is usable` $\rightarrow$ **PASS**
7. `Mamdani: all beliefs within [0,1]` $\rightarrow$ **PASS**
8. `Sugeno-0: label UNAVAILABLE exactly where no modality is usable` $\rightarrow$ **PASS**
9. `Sugeno-0: all beliefs within [0,1]` $\rightarrow$ **PASS**
10. `Tsukamoto: label UNAVAILABLE exactly where no modality is usable` $\rightarrow$ **PASS**
11. `Tsukamoto: all beliefs within [0,1]` $\rightarrow$ **PASS**
12. `Sugeno-1 constrained fit converged` $\rightarrow$ **PASS**
13. `Sugeno-1: all beliefs within [0,1] (no negative scores)` $\rightarrow$ **PASS**
14. `Sugeno-1: every rule consequent within [0,1] at all corners of the input domain` $\rightarrow$ **PASS**
15. `single modality: weighted mean equals that branch's argmax` $\rightarrow$ **PASS**
16. `chance-level model receives (near) zero quality` $\rightarrow$ **PASS**

### Missing-Modality Dynamic Weight Redistribution
Derived from `results/tables/missing_modality_weights.csv`:

| Available Modalities | Binary Mask | $w_{\text{EEG}}$ | $w_{\text{Face}}$ | $w_{\text{Speech}}$ | $w_{\text{Video}}$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Active** | `1110` | 0.479 | 0.287 | 0.234 | 0.000 | OK |
| **EEG + Face** | `1100` | 0.625 | 0.375 | 0.000 | 0.000 | OK |
| **Face + Speech** | `0110` | 0.000 | 0.551 | 0.449 | 0.000 | OK |
| **EEG Only** | `1000` | 1.000 | 0.000 | 0.000 | 0.000 | OK |
| **Face Only** | `0100` | 0.000 | 1.000 | 0.000 | 0.000 | OK |
| **Speech Only** | `0010` | 0.000 | 0.000 | 1.000 | 0.000 | OK |
| **Video Only** | `0001` | 0.000 | 0.000 | 0.000 | 0.000 | UNAVAILABLE |

---

## 7. Repository Map & File Index

```
cv_project_adv/
├── .gitignore
├── README.md                      # High-level project status & execution guide
├── README2.md                     # This file (in-depth theoretical & technical manual)
├── requirements.txt               # Python package dependencies
│
├── fuzzy/
│   └── fis.py                     # Mamdani, Sugeno-0/1, Tsukamoto, weights & masks
│
├── models/
│   ├── branches.py                # EEGFNN, FaceVGG, SpeechCNN, ResNet-18 extractor
│   ├── video_branch.py            # Visual temporal face pipeline & SVM
│   └── assets/
│       └── haarcascade_frontalface_default.xml # OpenCV Haar Cascade weights
│
├── preprocessing/
│   └── speech_features.py         # RAVDESS audio MFCC extraction
│
├── training/
│   └── train_unimodal.py          # Unified unimodal training harness
│
├── evaluation/
│   └── fusion_experiments.py      # Multimodal fusion evaluation suite
│
├── tests/
│   └── test_fusion.py             # 16 executable verification unit tests
│
├── eav/
│   ├── README_EAV.md              # EAV dataset documentation
│   ├── inventory.py               # EAV file and participant scanner
│   └── features.py                # EAV Butterworth EEG bandpower & MFCCs
│
├── visualizations/
│   └── make_figures.py            # Plotting script for results & curves
│
├── report/
│   └── final_report.pdf           # Comprehensive project research report
│
└── results/
    ├── graphs/                    # Generated PDF visualization plots
    ├── tables/                    # CSVs for test checks, weights, & metrics
    └── preliminary_outputs/       # Saved predictions, metrics, & histories
```
