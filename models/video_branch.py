"""Video branch (VISUAL ONLY): 16 frames -> face crop -> frozen ResNet-18 (512-d per
frame) -> temporal summary: mean, std and mean absolute frame-to-frame difference over
frames (3 x 512 = 1536-d) -> StandardScaler -> RBF-SVM with Platt probabilities.
Audio/MFCC features are deliberately NOT used here: speech is handled by the Speech
branch, so using the soundtrack again would count the same evidence twice.

STATUS: implemented and unit-tested on synthetic frames only.  No video
recordings could be obtained in the execution environment, so this branch was
NOT trained or evaluated; its fusion weight is 0 in every executed experiment.
"""
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sample_frames(path, n=16):
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 1, 0), n).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f[:, :, ::-1])  # BGR -> RGB
    cap.release()
    return frames


def crop_face(rgb, size=224):
    import cv2
    import os
    det = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), "assets",
                                             "haarcascade_frontalface_default.xml"))
    if det.empty():
        raise RuntimeError("Haar cascade not found")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = det.detectMultiScale(gray, 1.1, 5)
    if len(faces):
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        rgb = rgb[y:y + h, x:x + w]
    return cv2.resize(rgb, (size, size))


def frames_to_tensor(frames):
    x = np.stack([crop_face(f) for f in frames]).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(0, 3, 1, 2)


@torch.no_grad()
def visual_descriptor(backbone, frames_tensor):
    """Per-frame ResNet-18 features f_t (T x 512) summarised over time:
    mean (appearance), std (variability) and mean |f_t - f_{t-1}| (motion/dynamics)."""
    feats = backbone(frames_tensor)                                   # (T, 512)
    motion = (feats[1:] - feats[:-1]).abs().mean(0)
    return torch.cat([feats.mean(0), feats.std(0), motion]).numpy()  # (1536,)


def make_svm(C=10.0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(StandardScaler(), SVC(C=C, kernel="rbf", gamma="scale", probability=True,
                                               class_weight="balanced", random_state=42))
