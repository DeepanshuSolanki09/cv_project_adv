"""Train the EEG, Face and Speech branches and save probabilities + histories.

Usage: python training/train_unimodal.py {eeg|eeg_all|face|speech}
Outputs go to outputs/<branch>/ : history.json, probs_{val,test}.npz, metrics.json
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.branches import EEGFNN, FaceVGG, SpeechCNN, n_params

ROOT = "/home/claude"
OUT = f"{ROOT}/mer/outputs"
SEED = 42
torch.set_num_threads(1)


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)


def class_weights(y, C=3):
    cnt = np.bincount(y, minlength=C).astype(float)
    return torch.tensor(cnt.sum() / (C * cnt), dtype=torch.float32)


@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    out = [torch.softmax(model(torch.from_numpy(X[i:i + bs])), 1).numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out)


def fit(model, Xtr, ytr, Xva, yva, epochs, lr, bs, wd=1e-4, patience=8, augment=None, tag=""):
    """Training loop with early stopping on validation macro-F1.  State is checkpointed
    after every epoch (outputs/ckpt/<tag>.pt) so an interrupted run resumes exactly."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr))
    hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_macro_f1"]}
    best, best_state, bad, start = -1, None, 0, 0
    yva_t = torch.from_numpy(yva)
    ck = f"{OUT}/ckpt/{tag}.pt"; os.makedirs(f"{OUT}/ckpt", exist_ok=True)
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        hist, best, best_state, bad, start = st["hist"], st["best"], st["best_state"], st["bad"], st["epoch"]
        np.random.set_state(st["np_rng"]); torch.set_rng_state(st["torch_rng"])
        print(f"[{tag}] resumed at epoch {start}", flush=True)
        if st.get("done"):
            start = epochs
    for ep in range(start, epochs):
        model.train(); t0 = time.time()
        perm = np.random.permutation(len(Xtr)); tl, tc = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx]
            if augment is not None:
                xb = augment(xb)
            xb = torch.from_numpy(xb); yb = torch.from_numpy(ytr[idx])
            opt.zero_grad(); logits = model(xb); loss = lossf(logits, yb); loss.backward(); opt.step()
            tl += loss.item() * len(idx); tc += (logits.argmax(1) == yb).sum().item()
        pva = predict(model, Xva)
        vloss = nn.functional.nll_loss(torch.log(torch.from_numpy(pva) + 1e-9), yva_t,
                                       weight=class_weights(ytr)).item()
        vf1 = f1_score(yva, pva.argmax(1), average="macro")
        hist["train_loss"].append(tl / len(Xtr)); hist["train_acc"].append(tc / len(Xtr))
        hist["val_loss"].append(vloss); hist["val_acc"].append(accuracy_score(yva, pva.argmax(1)))
        hist["val_macro_f1"].append(vf1)
        sched.step(vf1)
        print(f"[{tag}] ep {ep+1:02d} loss {tl/len(Xtr):.3f}/{vloss:.3f} acc {tc/len(Xtr):.3f}/"
              f"{hist['val_acc'][-1]:.3f} vF1 {vf1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if vf1 > best:
            best, bad = vf1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        done = bad >= patience or ep + 1 == epochs
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "hist": hist,
                    "best": best, "best_state": best_state, "bad": bad, "epoch": ep + 1, "done": done,
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        if done:
            break
    model.load_state_dict(best_state)
    hist["best_epoch"] = int(np.argmax(hist["val_macro_f1"]) + 1)
    return model, hist


def save(name, hist, extra, **arrays):
    d = f"{OUT}/{name}"; os.makedirs(d, exist_ok=True)
    json.dump(hist, open(f"{d}/history.json", "w"), indent=1)
    json.dump(extra, open(f"{d}/meta.json", "w"), indent=1)
    np.savez_compressed(f"{d}/probs.npz", **arrays)


def run_eeg(all_features=False):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    d = pd.read_csv(f"{ROOT}/data/eeg/emotions.csv")
    y = d["label"].map({"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}).values.astype(np.int64)
    cols = [c for c in d.columns if c != "label"] if all_features else [f"fft_{i}_a" for i in range(750)]
    X = d[cols].values.astype(np.float32)
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    itr, iva = train_test_split(itr, test_size=0.25, stratify=y[itr], random_state=SEED)
    sc = StandardScaler().fit(X[itr])                      # fitted on training windows only
    Xs = np.clip(sc.transform(X), -10, 10).astype(np.float32)
    seed_all(SEED)
    m = EEGFNN(n_in=X.shape[1])
    m, h = fit(m, Xs[itr], y[itr], Xs[iva], y[iva], epochs=100, lr=1e-3, bs=64, patience=15, tag="eeg")
    name = "eeg_all" if all_features else "eeg"
    save(name, h, {"n_params": n_params(m), "n_features": X.shape[1], "split": [len(itr), len(iva), len(ite)]},
         p_val=predict(m, Xs[iva]), y_val=y[iva], p_test=predict(m, Xs[ite]), y_test=y[ite])


def run_face():
    tr = np.load(f"{ROOT}/data/fer/fer_Training.npz"); va = np.load(f"{ROOT}/data/fer/fer_PublicTest.npz")
    te = np.load(f"{ROOT}/data/fer/fer_PrivateTest.npz")
    f = lambda X: (X[:, None].astype(np.float32) / 255.0)
    Xtr, Xva, Xte = f(tr["X"]), f(va["X"]), f(te["X"])

    def augment(xb):                                       # horizontal flip + small shift
        xb = xb.copy()
        flip = np.random.rand(len(xb)) < 0.5
        xb[flip] = xb[flip, :, :, ::-1]
        dx, dy = np.random.randint(-3, 4, 2)
        return np.roll(np.roll(xb, dx, 3), dy, 2)

    seed_all(SEED)
    m = FaceVGG(width=32)
    m, h = fit(m, Xtr, tr["y"].astype(np.int64), Xva, va["y"].astype(np.int64), epochs=int(os.environ.get("FACE_EPOCHS", 12)),
               lr=1e-3, bs=128, patience=3, augment=augment, tag="face")
    torch.save(m.state_dict(), f"{OUT}/face_vgg.pt")
    save("face", h, {"n_params": n_params(m), "width": 32},
         p_val=predict(m, Xva), y_val=va["y"], p_test=predict(m, Xte), y_test=te["y"])


def run_speech(n_folds=4):
    d = np.load(f"{ROOT}/data/ravdess_speech.npz")
    X, y, actor = d["X"], d["y"].astype(np.int64), d["actor"]
    actors = np.arange(1, 25)
    rng = np.random.default_rng(SEED); order = rng.permutation(actors)
    folds = np.array_split(order, n_folds)                  # 6 test actors per fold
    p_test = np.zeros((len(y), 3), np.float32); fold_of = np.zeros(len(y), int)
    pv, yv, hists, fold_actors = [], [], [], []
    for k in range(n_folds):
        te_a = folds[k]; va_a = folds[(k + 1) % n_folds][:4]
        tr_a = np.setdiff1d(actors, np.concatenate([te_a, va_a]))
        itr, iva, ite = [np.where(np.isin(actor, a))[0] for a in (tr_a, va_a, te_a)]
        # per-utterance cepstral mean & variance normalisation (CMVN): every clip is
        # normalised with its OWN statistics, so no information crosses splits
        Z = ((X - X.mean(2, keepdims=True)) / (X.std(2, keepdims=True) + 1e-6))[:, None].astype(np.float32)

        def augment(xb):                                   # random time shift + SpecAugment-style mask
            xb = np.roll(xb, np.random.randint(-12, 13), axis=3).copy()
            t0 = np.random.randint(0, 112); xb[:, :, :, t0:t0 + np.random.randint(0, 16)] = 0
            return xb

        fold_file = f"{OUT}/ckpt/speech_fold{k}.npz"
        if os.path.exists(fold_file):
            fz = np.load(fold_file, allow_pickle=True)
            p_test[ite] = fz["pt"]; fold_of[ite] = k; pv.append(fz["pv"]); yv.append(y[iva])
            hists.append(json.loads(str(fz["h"]))); fold_actors.append({"train": tr_a.tolist(), "val": va_a.tolist(), "test": te_a.tolist()})
            continue
        seed_all(SEED + k)
        m = SpeechCNN()
        m, h = fit(m, Z[itr], y[itr], Z[iva], y[iva], epochs=60, lr=5e-4, bs=32, patience=12,
                   augment=augment, tag=f"speech-f{k}")
        p_test[ite] = predict(m, Z[ite]); fold_of[ite] = k
        np.savez(fold_file, pt=p_test[ite], pv=predict(m, Z[iva]), h=json.dumps(h))
        pv.append(predict(m, Z[iva])); yv.append(y[iva]); hists.append(h)
        fold_actors.append({"train": tr_a.tolist(), "val": va_a.tolist(), "test": te_a.tolist()})
    save("speech", {"folds": hists}, {"n_params": n_params(SpeechCNN()), "folds": fold_actors},
         p_val=np.concatenate(pv), y_val=np.concatenate(yv), p_test=p_test, y_test=y, fold=fold_of,
         actor=actor)


if __name__ == "__main__":
    which = sys.argv[1]
    {"eeg": lambda: run_eeg(False), "eeg_all": lambda: run_eeg(True), "face": run_face, "speech": run_speech}[which]()
