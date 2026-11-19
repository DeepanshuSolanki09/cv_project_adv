"""Generate all result figures from outputs/*.  Every figure here is computed from
saved experiment artefacts; nothing is typed in by hand."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fuzzy.fis import Mamdani, Tsukamoto, trimf, with_support_peak, RULES, ANCHOR

OUT = "/home/claude/mer/outputs"; FIG = "/home/claude/report/figures"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
CL = ["Negative", "Neutral", "Positive"]
COL = {"EEG": "#1b9e77", "Face": "#7570b3", "Speech": "#d95f02", "Video": "#999999"}
MCOL = {"Mean": "#bbbbbb", "W-Mean": "#666666", "Mamdani": "#1f78b4", "Sugeno-0": "#33a02c",
        "Sugeno-1": "#b2df8a", "Tsukamoto": "#e31a1c"}
METHODS = ["Mean", "W-Mean", "Mamdani", "Sugeno-0", "Sugeno-1", "Tsukamoto"]


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf"); plt.close(fig)


def class_distributions():
    d = np.load("/home/claude/data/ravdess_speech.npz")
    import pandas as pd
    e = pd.read_csv("/home/claude/data/eeg/emotions.csv", usecols=["label"])["label"].map(
        {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}).values
    fer = sum(np.bincount(np.load(f"/home/claude/data/fer/fer_{s}.npz")["y"], minlength=3)
              for s in ["Training", "PublicTest", "PrivateTest"])
    fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.5))
    for ax, (t, cnt, c) in zip(axs, [("EEG: Feeling Emotions", np.bincount(e), COL["EEG"]),
                                      ("Face: FER2013 (3-class)", fer, COL["Face"]),
                                      ("Speech: RAVDESS (3-class)", np.bincount(d["y"]), COL["Speech"])]):
        b = ax.bar(CL, cnt, color=c); ax.bar_label(b, fontsize=8); ax.set_title(t, fontsize=9)
        ax.set_ylim(0, max(cnt) * 1.18)
    axs[0].set_ylabel("samples")
    save(fig, "class_distribution")


def curves():
    fig, axs = plt.subplots(2, 3, figsize=(8.8, 4.6))
    for j, (b, t) in enumerate([("eeg", "EEG-FNN"), ("face", "Face CNN"), ("speech", "Speech CNN (fold 1)")]):
        h = json.load(open(f"{OUT}/{b}/history.json"))
        if "folds" in h: h = h["folds"][0]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        axs[0, j].plot(ep, h["train_loss"], label="train"); axs[0, j].plot(ep, h["val_loss"], label="validation")
        axs[1, j].plot(ep, h["train_acc"], label="train"); axs[1, j].plot(ep, h["val_acc"], label="validation")
        for i in (0, 1):
            axs[i, j].axvline(h["best_epoch"], ls=":", c="k", lw=0.8)
        axs[0, j].set_title(t, fontsize=9); axs[1, j].set_xlabel("epoch")
    axs[0, 0].set_ylabel("weighted CE loss"); axs[1, 0].set_ylabel("accuracy"); axs[0, 0].legend(fontsize=7)
    save(fig, "training_curves")


def speech_folds():
    h = json.load(open(f"{OUT}/speech/history.json"))["folds"]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for k, hk in enumerate(h):
        ax.plot(np.arange(1, len(hk["val_macro_f1"]) + 1), hk["val_macro_f1"], label=f"fold {k+1}")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation macro-F1"); ax.legend(fontsize=7)
    save(fig, "speech_fold_curves")


def unimodal(R):
    u = R["unimodal"]; mods = list(u)
    fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.5), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
    keys = ["acc", "prec", "rec", "macro_f1", "weighted_f1", "auc"]
    lab = ["Acc.", "Prec.", "Rec.", "Macro-F1", "W-F1", "AUC"]
    x = np.arange(len(keys)); wd = 0.26
    for i, m in enumerate(mods):
        axs[0].bar(x + (i - 1) * wd, [u[m][k] for k in keys], wd, label=m, color=COL[m])
    axs[0].set_xticks(x, lab, fontsize=7); axs[0].set_ylim(0, 1.05); axs[0].legend(fontsize=7, ncol=3, loc="lower center")
    axs[0].set_title("Test metrics per branch", fontsize=9)
    for ax, m in zip(axs[1:], mods):
        cm = np.array(u[m]["cm"], float); cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{cmn[a,b]:.2f}\n({int(cm[a,b])})", ha="center", va="center", fontsize=6.5,
                        color="white" if cmn[a, b] > 0.55 else "black")
        ax.set_xticks(range(3), ["Neg", "Neu", "Pos"], fontsize=7); ax.set_yticks(range(3), ["Neg", "Neu", "Pos"], fontsize=7)
        ax.set_title(f"{m} (n={u[m]['n_test']})", fontsize=9); ax.set_xlabel("predicted", fontsize=7)
    axs[1].set_ylabel("true", fontsize=7)
    save(fig, "unimodal_results")


def unimodal_roc():
    fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.6))
    for ax, (b, m) in zip(axs, [("eeg", "EEG"), ("face", "Face"), ("speech", "Speech")]):
        d = np.load(f"{OUT}/{b}/probs.npz"); p, y = d["p_test"], d["y_test"].astype(int)
        for c in range(3):
            fpr, tpr, _ = roc_curve(y == c, p[:, c]); ax.plot(fpr, tpr, label=f"{CL[c]} ({auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.7); ax.set_title(m, fontsize=9); ax.set_xlabel("FPR"); ax.legend(fontsize=6.5)
    axs[0].set_ylabel("TPR")
    save(fig, "unimodal_roc")


def mfs(R):
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.3))
    x = np.linspace(0, 1, 401)
    pk = R["tuning"]["Mamdani"]["best_peak"]; p = with_support_peak(pk)
    for k, v in p["S"].items(): axs[0].plot(x, trimf(x, *v), label=k)
    axs[0].set_title(f"Input: weighted support $S_c$ (peak {pk})", fontsize=9)
    for k, v in p["G"].items(): axs[1].plot(x, trimf(x, *v), label=k)
    axs[1].set_title("Input: consistency $G_c$", fontsize=9)
    for k, v in Mamdani.OUT.items(): axs[2].plot(x, trimf(x, *v), label=k)
    axs[2].set_title("Mamdani output sets (belief $B_c$)", fontsize=9)
    for a in axs: a.legend(fontsize=6.5); a.set_ylim(0, 1.08)
    save(fig, "membership_functions")


def tsukamoto_sets():
    t = Tsukamoto(); fig, ax = plt.subplots(figsize=(4.2, 2.4))
    seen = set()
    for k, (_, _, o) in enumerate(RULES):
        if o in seen: continue
        seen.add(o); a = np.linspace(0, 1, 50); z = t.start[k] + a * (t.end[k] - t.start[k])
        ax.plot(z, a, label=f"{o} ({t.start[k]:.2f}→{t.end[k]:.2f})")
    ax.set_xlabel("output z"); ax.set_ylabel("membership"); ax.set_title("Tsukamoto monotonic output sets", fontsize=9)
    ax.legend(fontsize=6.5)
    save(fig, "tsukamoto_sets")


def weights(R):
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    mods = ["EEG", "Face", "Speech", "Video"]
    b = ax.bar(mods, R["w"], color=[COL[m] for m in mods]); ax.bar_label(b, fmt="%.3f", fontsize=8)
    ax.set_ylabel("reliability weight $w_m$"); ax.set_ylim(0, max(R["w"]) * 1.25)
    ax.set_title("Weights from validation macro-F1 (Video masked)", fontsize=9)
    save(fig, "reliability_weights")


def fusion_main(R):
    s = R["summary"]["EEG+Face+Speech"]
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 2.8))
    for ax, k, t in [(axs[0], "macro_f1", "Macro-F1"), (axs[1], "auc", "Macro ROC-AUC")]:
        m = [s[x][k][0] for x in METHODS]; e = [s[x][k][1] for x in METHODS]
        b = ax.bar(METHODS, m, yerr=e, capsize=2, color=[MCOL[x] for x in METHODS])
        ax.bar_label(b, fmt="%.3f", fontsize=7, padding=2)
        lo = min(m) - 0.05; ax.set_ylim(lo, min(1.0, max(m) + 0.03)); ax.set_title(t + " (EEG+Face+Speech)", fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=20)
    best = R["unimodal"]
    axs[0].axhline(max(best[m]["macro_f1"] for m in best), ls="--", c="k", lw=0.8)
    save(fig, "fusion_comparison")


def fusion_subsets(R):
    order = ["EEG", "Face", "Speech", "EEG+Face", "EEG+Speech", "Face+Speech", "EEG+Face+Speech", "Random missing (30%)"]
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    x = np.arange(len(order)); wd = 0.13
    for i, m in enumerate(METHODS):
        ax.bar(x + (i - 2.5) * wd, [R["summary"][o][m]["macro_f1"][0] for o in order], wd, label=m, color=MCOL[m])
    ax.set_xticks(x, [o.replace("+", "+\n") if len(o) > 12 else o for o in order], fontsize=7)
    ax.set_ylim(0.3, 1.0); ax.set_ylabel("macro-F1 (mean of 20 pairings)"); ax.legend(fontsize=7, ncol=6, loc="upper left")
    save(fig, "fusion_subsets")


def confusions(R):
    fig, axs = plt.subplots(1, 4, figsize=(9, 2.3))
    for ax, m in zip(axs, ["Mamdani", "Sugeno-0", "Sugeno-1", "Tsukamoto"]):
        cm = np.array(R["cms"][m], float); cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{cmn[a,b]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[a, b] > 0.55 else "black")
        ax.set_xticks(range(3), ["Neg", "Neu", "Pos"], fontsize=7); ax.set_yticks(range(3), ["Neg", "Neu", "Pos"], fontsize=7)
        ax.set_title(m, fontsize=9)
    save(fig, "fusion_confusions")


def timing(R):
    t = R["time_us_per_sample"]; fig, ax = plt.subplots(figsize=(4.4, 2.4))
    b = ax.bar(METHODS, [t[m][0] for m in METHODS], color=[MCOL[m] for m in METHODS])
    ax.bar_label(b, fmt="%.1f", fontsize=7); ax.set_yscale("log"); ax.set_ylabel("µs per sample (log)")
    ax.tick_params(axis="x", labelsize=7, rotation=20); ax.set_title("Fusion-stage inference time (1 CPU core)", fontsize=9)
    save(fig, "inference_time")


def tuning(R):
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    for m in ["Mamdani", "Sugeno-0", "Sugeno-1", "Tsukamoto"]:
        ax.plot(R["tuning"][m]["grid"], R["tuning"][m]["val_f1"], marker="o", ms=3, label=m, color=MCOL[m])
    ax.set_xlabel("peak of 'Medium support' MF"); ax.set_ylabel("validation macro-F1"); ax.legend(fontsize=7)
    save(fig, "tuning")


def example(R):
    e = R["example"]; P = np.array(e["P"])
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 2.5))
    x = np.arange(3); wd = 0.25
    for i, m in enumerate(["EEG", "Face", "Speech"]):
        axs[0].bar(x + (i - 1) * wd, P[i], wd, label=f"{m} (w={e['w'][i]:.2f})", color=COL[m])
    axs[0].set_xticks(x, CL); axs[0].legend(fontsize=7); axs[0].set_title(f"Branch probabilities (true: {CL[e['true']]})", fontsize=9)
    ms = ["W-Mean", "Mamdani", "Sugeno-0", "Sugeno-1", "Tsukamoto"]; wd = 0.16
    for i, m in enumerate(ms):
        axs[1].bar(x + (i - 2) * wd, e["outputs"][m], wd, label=m, color=MCOL[m])
    axs[1].set_xticks(x, CL); axs[1].legend(fontsize=6.5); axs[1].set_title("Fused class scores", fontsize=9)
    save(fig, "prediction_example")


if __name__ == "__main__":
    R = json.load(open(f"{OUT}/fusion_results.json"))
    for f in [class_distributions, curves, speech_folds, unimodal_roc, tsukamoto_sets]: f()
    for f in [unimodal, mfs, weights, fusion_main, fusion_subsets, confusions, timing, tuning, example]: f(R)
    print("figures written to", FIG)
