"""Executable checks of the fusion engine (run: python tests/test_fusion.py)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fuzzy.fis import (masked_weights, Mamdani, Sugeno, Tsukamoto, weighted_mean_predict, UNAVAILABLE,
                       chance_corrected_quality)

rng = np.random.default_rng(0)
results = []

def check(name, cond):
    results.append({"test": name, "passed": bool(cond)}); print(("PASS " if cond else "FAIL ") + name)

def rand_probs(n, m=4):
    x = rng.random((n, m, 3)); return x / x.sum(2, keepdims=True)

q = np.array([0.93, 0.56, 0.45, 0.0])                      # Video untrained -> q = 0
N = 5000
P = rand_probs(N)
A = rng.random((N, 4)) > 0.4
W, valid = masked_weights(q, A, return_valid=True)
check("weights are zero for every unavailable modality", np.all(W[~A] == 0))
check("weights are zero for a modality without a trained model (q=0)", np.all(W[:, 3] == 0))
check("weights sum to 1 whenever a trained modality is available", np.allclose(W[valid].sum(1), 1))
check("rows with no trained modality are flagged unavailable (no division by zero)",
      np.all(valid == (A[:, :3].any(1))))
Anone = np.zeros((3, 4), bool); Anone[1, 3] = True          # only the untrained Video "available"
Wn, vn = masked_weights(q, Anone, return_valid=True)
check("all-unavailable input returns zero weights and valid=False", np.all(Wn == 0) and not vn.any())
for f in [Mamdani(), Sugeno(order=0), Tsukamoto()]:
    y, B = f.predict(P, W)
    check(f"{f.name}: label UNAVAILABLE exactly where no modality is usable", np.all((y == UNAVAILABLE) == ~valid))
    check(f"{f.name}: all beliefs within [0,1]", np.nanmin(B) >= 0 and np.nanmax(B) <= 1)
# Sugeno-1 with constrained consequents: fit on random labelled data, then test on new random data
yv = rng.integers(0, 3, N)
Pv = rand_probs(N); Pv[np.arange(N), :, yv] += 0.8; Pv /= Pv.sum(2, keepdims=True)
Wv = masked_weights(q, np.ones((N, 4), bool))
s1 = Sugeno(order=1).fit(Pv, Wv, yv)
y1, B1 = s1.predict(P, W)
check("Sugeno-1 constrained fit converged", s1.fit_info["success"])
check("Sugeno-1: all beliefs within [0,1] (no negative scores)", np.nanmin(B1) >= -1e-9 and np.nanmax(B1) <= 1 + 1e-9)
th = s1.theta; corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
zc = th[:, :1].T * 0  # placeholder
zvals = np.array([[t[0] * cs + t[1] * cg + t[2] for cs, cg in corners] for t in th])
check("Sugeno-1: every rule consequent within [0,1] at all corners of the input domain",
      zvals.min() >= -1e-7 and zvals.max() <= 1 + 1e-7)
# single available modality -> W-Mean reduces to that modality
A1 = np.zeros((N, 4), bool); A1[:, 0] = True
y_wm, _ = weighted_mean_predict(P, masked_weights(q, A1))
check("single modality: weighted mean equals that branch's argmax", np.all(y_wm == P[:, 0].argmax(1)))
check("chance-level model receives (near) zero quality", chance_corrected_quality([1/3])[0] <= 1e-3)
os.makedirs("results/tables", exist_ok=True)
import csv
with open("results/tables/fusion_engine_tests.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, ["test", "passed"]); w.writeheader(); w.writerows(results)
print(f"{sum(r['passed'] for r in results)}/{len(results)} passed")
