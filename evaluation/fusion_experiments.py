"""Decision-level fusion experiments on REAL branch outputs.

IMPORTANT - what this experiment is and is not
---------------------------------------------
The three executed branches were trained on three different datasets (EEG:
Bird 'Feeling Emotions'; Face: FER2013; Speech: RAVDESS).  These are NOT
recordings of the same person or event.  To exercise the fusion layer on real
classifier outputs we use *class-conditional pseudo-pairing*: a pseudo-trial
of class c combines one held-out EEG prediction, one held-out face prediction
and one held-out speech prediction whose TRUE label is c.  This makes the
modality errors conditionally independent by construction, which is optimistic
for any fusion rule.  Results therefore measure how the fusion RULES combine
real, imperfectly calibrated classifier outputs; they do NOT measure multimodal
emotion-recognition accuracy on people.  The video branch has no data and is
masked (weight 0) everywhere.
"""
import itertools, json, os, sys, time
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fuzzy.fis import (Mamdani, Sugeno, Tsukamoto, chance_corrected_quality, masked_weights,
                       weighted_mean_predict, with_support_peak, fuzzy_inputs, firing_strengths, RULE_TEXT)

OUT = "/home/claude/mer/outputs"
MODS = ["EEG", "Face", "Speech", "Video"]
BR = {"EEG": "eeg", "Face": "face", "Speech": "speech"}
N_REP = 20


def load():
    pools, val_f1 = {}, {}
    for m, b in BR.items():
        d = np.load(f"{OUT}/{b}/probs.npz")
        pools[m] = {s: (d[f"p_{s}"], d[f"y_{s}"].astype(int)) for s in ("val", "test")}
        val_f1[m] = f1_score(d["y_val"], d["p_val"].argmax(1), average="macro")
    return pools, val_f1


def pseudo_trials(pools, split, rng, n_c=None):
    """Return P (N,4,3), y (N,).  Same n_c per class for every modality."""
    if n_c is None:
        n_c = min(np.bincount(pools[m][split][1], minlength=3).min() for m in BR)
    P, Y = [], []
    for c in range(3):
        block = np.full((n_c, 4, 3), np.nan)
        for j, m in enumerate(MODS[:3]):
            p, y = pools[m][split]
            idx = rng.choice(np.where(y == c)[0], n_c, replace=False)
            block[:, j] = p[idx]
        P.append(block); Y.append(np.full(n_c, c))
    return np.concatenate(P), np.concatenate(Y), n_c


def metrics(y, pred, score=None):
    r = {"acc": accuracy_score(y, pred),
         "prec": precision_score(y, pred, average="macro", zero_division=0),
         "rec": recall_score(y, pred, average="macro", zero_division=0),
         "macro_f1": f1_score(y, pred, average="macro"),
         "weighted_f1": f1_score(y, pred, average="weighted")}
    if score is not None:
        s = np.clip(score, 1e-9, None); s = s / s.sum(1, keepdims=True)
        try:
            r["auc"] = roc_auc_score(y, s, multi_class="ovr", average="macro")
        except ValueError:
            r["auc"] = np.nan
    return r


def make_methods(params, sug1_theta):
    s1 = Sugeno(params["Sugeno-1"], order=1); s1.theta = sug1_theta
    return {"Mamdani": Mamdani(params["Mamdani"]), "Sugeno-0": Sugeno(params["Sugeno-0"], order=0),
            "Sugeno-1": s1, "Tsukamoto": Tsukamoto(params["Tsukamoto"])}


def run_methods(P, avail, q, methods, single_best_idx):
    W = masked_weights(q, avail)
    res = {}
    t0 = time.perf_counter(); pr, sc = weighted_mean_predict(P, W); res["W-Mean"] = (pr, sc, time.perf_counter() - t0)
    Wu = masked_weights(np.where(q > 0, 1.0, 0.0), avail)
    t0 = time.perf_counter(); pr, sc = weighted_mean_predict(P, Wu); res["Mean"] = (pr, sc, time.perf_counter() - t0)
    for name, f in methods.items():
        t0 = time.perf_counter(); pr, B = f.predict(P, W); res[name] = (pr, B, time.perf_counter() - t0)
    return res


def main():
    pools, val_f1 = load()
    q = np.array([chance_corrected_quality([val_f1[m]])[0] if m in val_f1 else 0.0 for m in MODS])
    w_all = q / q.sum()
    print("val macro-F1", val_f1, "\nq", q.round(4), "w", w_all.round(4))
    full = np.array([1, 1, 1, 0], bool)

    # ---------------- tuning on validation pseudo-trials ----------------
    rng = np.random.default_rng(1000)
    val_sets = [pseudo_trials(pools, "val", rng) for _ in range(5)]
    grid = np.round(np.arange(0.30, 0.66, 0.05), 2)
    params, tune_log = {}, {}
    sug1_theta = None
    for name in ["Mamdani", "Sugeno-0", "Sugeno-1", "Tsukamoto"]:
        scores = []
        for pk in grid:
            pr = with_support_peak(pk); f1s = []
            if name == "Sugeno-1":
                Pv = np.concatenate([v[0] for v in val_sets[:3]]); yv = np.concatenate([v[1] for v in val_sets[:3]])
                Av = np.tile(full, (len(Pv), 1))
                f = Sugeno(pr, order=1).fit(Pv, masked_weights(q, Av), yv)
                evals = val_sets[3:]            # fit on 3 resamples, score on 2 others
            else:
                f = {"Mamdani": Mamdani, "Sugeno-0": lambda p: Sugeno(p, 0), "Tsukamoto": Tsukamoto}[name](pr)
                evals = val_sets
            for Pv, yv, _ in evals:
                Av = np.tile(full, (len(Pv), 1))
                f1s.append(f1_score(yv, f.predict(Pv, masked_weights(q, Av))[0], average="macro"))
            scores.append(np.mean(f1s))
        best = grid[int(np.argmax(scores))]
        params[name] = with_support_peak(best); tune_log[name] = {"grid": grid.tolist(), "val_f1": scores, "best_peak": float(best)}
        if name == "Sugeno-1":
            Pv = np.concatenate([v[0] for v in val_sets]); yv = np.concatenate([v[1] for v in val_sets])
            sug1_theta = Sugeno(params[name], order=1).fit(Pv, masked_weights(q, np.tile(full, (len(Pv), 1))), yv).theta
        print(name, "best support peak", best, "val F1", round(max(scores), 4))
    methods = make_methods(params, sug1_theta)

    # ---------------- test scenarios ----------------
    subsets = [s for r in (1, 2, 3) for s in itertools.combinations(range(3), r)]
    scen = {"+".join(MODS[i] for i in s): s for s in subsets}
    results = {k: {} for k in list(scen) + ["Random missing (30%)"]}
    cms, timing, example = {}, {}, None
    rng = np.random.default_rng(2026)
    for rep in range(N_REP):
        P, y, n_c = pseudo_trials(pools, "test", rng)
        for sname, s in scen.items():
            avail = np.zeros((len(y), 4), bool); avail[:, list(s)] = True
            res = run_methods(P, avail, q, methods, None)
            for meth, (pr, sc, dt) in res.items():
                results[sname].setdefault(meth, []).append(metrics(y, pr, sc))
                if sname == "EEG+Face+Speech":
                    timing.setdefault(meth, []).append(dt / len(y) * 1e6)
                    if rep == 0:
                        cms[meth] = confusion_matrix(y, pr, labels=[0, 1, 2]).tolist()
            if sname == "EEG+Face+Speech" and rep == 0:
                example = (P, y, avail, res)
        # random missing: each of the three modalities dropped with prob 0.3, at least one kept
        avail = rng.random((len(y), 4)) > 0.3; avail[:, 3] = False
        empty = ~avail[:, :3].any(1); avail[empty, rng.integers(0, 3, empty.sum())] = True
        for meth, (pr, sc, dt) in run_methods(P, avail, q, methods, None).items():
            results["Random missing (30%)"].setdefault(meth, []).append(metrics(y, pr, sc))

    # unimodal test metrics on their own full test sets (real, not pseudo-paired)
    uni = {}
    for m in BR:
        p, yy = pools[m]["test"]
        uni[m] = metrics(yy, p.argmax(1), p); uni[m]["val_macro_f1"] = val_f1[m]
        uni[m]["cm"] = confusion_matrix(yy, p.argmax(1), labels=[0, 1, 2]).tolist(); uni[m]["n_test"] = int(len(yy))

    summ = {s: {meth: {k: (float(np.nanmean([r[k] for r in rs])), float(np.nanstd([r[k] for r in rs])))
                       for k in rs[0]} for meth, rs in d.items()} for s, d in results.items()}
    # paired differences vs weighted mean (macro-F1), all-available scenario
    paired = {}
    for s in results:
        base = np.array([r["macro_f1"] for r in results[s]["W-Mean"]])
        paired[s] = {m: (float(np.mean(np.array([r["macro_f1"] for r in results[s][m]]) - base)),
                         float(np.std(np.array([r["macro_f1"] for r in results[s][m]]) - base)))
                     for m in ["Mamdani", "Sugeno-0", "Sugeno-1", "Tsukamoto"]}

    # worked example from the first test repetition: a sample where the modalities disagree
    P, y, avail, res = example
    W = masked_weights(q, avail)
    argm = np.nanargmax(np.nan_to_num(P[:, :3], nan=-1), axis=2)
    disagree = np.where((argm[:, 0] != argm[:, 1]) | (argm[:, 1] != argm[:, 2]))[0]
    i = int(disagree[(res["Mamdani"][0][disagree] == y[disagree])][0])
    S, G = fuzzy_inputs(P[i:i + 1], W[i:i + 1])
    al = firing_strengths(S, G, params["Mamdani"])
    ex = {"index": i, "true": int(y[i]), "P": np.nan_to_num(P[i, :3]).round(4).tolist(), "w": W[i].round(4).tolist(),
          "S": S[0].round(4).tolist(), "G": G[0].round(4).tolist(),
          "alpha": al[0].round(4).tolist(),
          "outputs": {m: np.asarray(res[m][1][i]).round(4).tolist() for m in res},
          "preds": {m: int(res[m][0][i]) for m in res}}

    json.dump({"val_f1": val_f1, "q": q.tolist(), "w": w_all.tolist(), "tuning": tune_log,
               "sugeno1_theta": sug1_theta.tolist(), "n_per_class": int(n_c), "n_rep": N_REP,
               "summary": summ, "paired_vs_wmean": paired, "unimodal": uni, "cms": cms,
               "time_us_per_sample": {m: (float(np.mean(t)), float(np.std(t))) for m, t in timing.items()},
               "example": ex, "rules": RULE_TEXT}, open(f"{OUT}/fusion_results.json", "w"), indent=1)
    for s in ["EEG+Face+Speech", "Face+Speech", "Random missing (30%)"]:
        print(s, {m: round(v["macro_f1"][0], 4) for m, v in summ[s].items()})


if __name__ == "__main__":
    main()
