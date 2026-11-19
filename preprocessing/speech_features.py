"""RAVDESS speech -> 40x128 MFCC maps (3-class mapping, surprise excluded)."""
import glob, os, numpy as np, librosa
SR, DUR, NFFT, HOP, NMEL, NMFCC, T = 22050, 3.0, 1024, 512, 64, 40, 128
MAP = {1: 1, 2: 1, 3: 2, 4: 0, 5: 0, 6: 0, 7: 0}   # neutral,calm->Neu; happy->Pos; sad,angry,fearful,disgust->Neg

def mfcc_map(path):
    y, _ = librosa.load(path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y / (np.max(np.abs(y)) + 1e-9)
    n = int(SR * DUR)
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=NFFT, hop_length=HOP, window="hamming", n_mels=NMEL)
    M = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=NMFCC)
    return M[:, :T].astype(np.float32)

if __name__ == "__main__":
    root = "/home/claude/data/ravdess"
    files = sorted({os.path.basename(p): p for p in glob.glob(f"{root}/**/03-01-*.wav", recursive=True)}.values())
    X, y, actor, ids = [], [], [], []
    for p in files:
        parts = os.path.basename(p)[:-4].split("-")
        emo, act = int(parts[2]), int(parts[6])
        if emo not in MAP: continue
        X.append(mfcc_map(p)); y.append(MAP[emo]); actor.append(act); ids.append(os.path.basename(p))
    X = np.stack(X)
    np.savez_compressed("/home/claude/data/ravdess_speech.npz", X=X, y=np.array(y), actor=np.array(actor), ids=np.array(ids))
    print(X.shape, np.bincount(y), len(set(actor)))
