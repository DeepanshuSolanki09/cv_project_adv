"""EAV per-modality feature extraction (STATUS: NOT RUN - no data; interfaces match the
official layout).  One feature/probability vector per 20 s trial so that all branches
refer to the SAME trial index (participant, trial)."""
import numpy as np
from scipy import signal

# Three-class mapping used in the report: Neg = anger, sadness; Neu = neutral;
# Pos = happiness, calmness  (calmness -> Neutral is the sensitivity variant)
MAP3 = {"Neutral": 1, "Anger": 0, "Sadness": 0, "Happiness": 2, "Calmness": 2}
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]

def eeg_bandpower(trial, fs=500, fs_target=100, win_s=2.0):
    """trial: (T=10000, C=30) raw EEG -> (n_windows, 150) log band power.
    Band-pass 0.3-50 Hz (4th-order zero-phase Butterworth), downsample to 100 Hz,
    2 s Hann windows, log power in five bands per channel (Eq. bandpower in the report)."""
    b, a = signal.butter(4, [0.3, 50], btype="band", fs=fs)
    x = signal.filtfilt(b, a, trial, axis=0)
    x = signal.resample_poly(x, 1, fs // fs_target, axis=0)
    n = int(win_s * fs_target); feats = []
    for s in range(0, x.shape[0] - n + 1, n):
        f, P = signal.periodogram(x[s:s + n], fs=fs_target, window="hann", axis=0)
        feats.append(np.concatenate([np.log(P[(f >= lo) & (f < hi)].sum(0) + 1e-12) for lo, hi in BANDS]))
    return np.stack(feats)

def speech_mfcc_segments(wav_path, seg_s=3.0, sr=22050):
    """Participant speech -> list of 40x128 MFCC maps (3 s segments), same settings as the
    preliminary speech branch (preprocessing/speech_features.py)."""
    import librosa
    y, _ = librosa.load(wav_path, sr=sr); y, _ = librosa.effects.trim(y, top_db=30)
    y = y / (np.abs(y).max() + 1e-9); n = int(seg_s * sr); out = []
    for s in range(0, max(len(y) - n, 0) + 1, n):
        seg = np.pad(y[s:s + n], (0, max(0, n - len(y[s:s + n]))))
        S = librosa.feature.melspectrogram(y=seg, sr=sr, n_fft=1024, hop_length=512, window="hamming", n_mels=64)
        M = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)[:, :128]
        out.append(((M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)).astype(np.float32))
    return out
