"""Decision-level fuzzy fusion for multimodal emotion recognition.

All three inference systems share EXACTLY the same front end:
  * the same per-class crisp inputs (weighted support S_c and consistency G_c),
  * the same input membership functions,
  * the same 9-rule antecedent base and the same AND operator (min).
They differ only in how rule consequents are represented and combined:
  Mamdani   : fuzzy output sets, min-implication, max-aggregation, centroid.
  Sugeno    : crisp consequents (zero-order constants or first-order linear
              functions of the inputs), firing-strength weighted average.
  Tsukamoto : monotonic output sets; each rule gives z_k = mu_k^{-1}(alpha_k),
              final output is the firing-strength weighted average of z_k.

Modality order: 0=EEG, 1=Face image, 2=Speech, 3=Video.  Class order: 0=Negative,
1=Neutral, 2=Positive.
"""
from __future__ import annotations
import numpy as np

MODALITIES = ["EEG", "Face", "Speech", "Video"]
CLASSES = ["Negative", "Neutral", "Positive"]

# --------------------------------------------------------------------------
# Reliability weights
# --------------------------------------------------------------------------

def chance_corrected_quality(val_macro_f1, n_classes=3, eps=1e-3):
    """q_m = max((F1_m - 1/C) / (1 - 1/C), eps): 0 = chance level, 1 = perfect."""
    f = np.asarray(val_macro_f1, dtype=float)
    return np.maximum((f - 1.0 / n_classes) / (1.0 - 1.0 / n_classes), eps)


UNAVAILABLE = -1  # label returned when no trained modality is available for a sample


def masked_weights(q, avail, return_valid=False):
    """w_{n,m} = a_{n,m} q_m / sum_k a_{n,k} q_k  (sums to 1 over available modalities).

    q     : (M,) modality qualities; q_m = 0 for a modality without a trained model
    avail : (N, M) boolean availability mask a_{n,m}
    Rows in which no modality with q_m > 0 is available get all-zero weights and
    valid = False (an 'unavailable' state) instead of a division by zero.
    """
    a = np.atleast_2d(np.asarray(avail, dtype=float))
    raw = a * np.asarray(q, dtype=float)[None, :]
    s = raw.sum(axis=1, keepdims=True)
    valid = s[:, 0] > 0
    W = np.divide(raw, s, out=np.zeros_like(raw), where=s > 0)
    return (W, valid) if return_valid else W


# --------------------------------------------------------------------------
# Crisp inputs of the fuzzy system (computed per sample and per class)
# --------------------------------------------------------------------------

def fuzzy_inputs(P, W):
    """P: (N, M, C) class probabilities (any value where unavailable), W: (N, M) weights.

    S_c = sum_m w_m p_{m,c}                         weighted support  in [0, 1]
    D_c = sqrt(sum_m w_m (p_{m,c} - S_c)^2)         weighted dispersion in [0, 0.5]
    G_c = 1 - 2 D_c                                 consistency in [0, 1]
    Unavailable modalities have w_m = 0 and therefore contribute nothing.
    """
    P = np.nan_to_num(P, nan=0.0)
    S = np.einsum("nm,nmc->nc", W, P)
    D = np.sqrt(np.einsum("nm,nmc->nc", W, (P - S[:, None, :]) ** 2))
    G = np.clip(1.0 - 2.0 * D, 0.0, 1.0)
    return np.clip(S, 0, 1), G


# --------------------------------------------------------------------------
# Membership functions
# --------------------------------------------------------------------------

def trimf(x, a, b, c):
    """Triangular MF; a==b or b==c give left/right shoulders."""
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / max(b - a, 1e-12), (x >= a).astype(float))
    right = np.where(c > b, (c - x) / max(c - b, 1e-12), (x <= c).astype(float))
    return np.clip(np.minimum(left, right), 0.0, 1.0)


DEFAULT_PARAMS = {
    # input sets on [0,1]: (a, b, c)
    "S": {"Low": (0.0, 0.0, 0.5), "Medium": (0.15, 0.5, 0.85), "High": (0.5, 1.0, 1.0)},
    "G": {"Low": (0.0, 0.0, 0.6), "Medium": (0.3, 0.65, 1.0), "High": (0.7, 1.0, 1.0)},
}

# Output levels and their representative ("anchor") values
LEVELS = ["VL", "L", "M", "H", "VH"]
ANCHOR = {"VL": 0.0, "L": 0.25, "M": 0.5, "H": 0.75, "VH": 1.0}

# Rule base: (S label, G label) -> output level
RULES = [
    ("Low", "Low", "VL"), ("Low", "Medium", "VL"), ("Low", "High", "VL"),
    ("Medium", "Low", "L"), ("Medium", "Medium", "M"), ("Medium", "High", "M"),
    ("High", "Low", "M"), ("High", "Medium", "H"), ("High", "High", "VH"),
]
RULE_TEXT = [f"R{i+1}: IF S is {s} AND G is {g} THEN B is {o}" for i, (s, g, o) in enumerate(RULES)]


def with_support_peak(peak, base=DEFAULT_PARAMS):
    """Return a copy of the MF parameters with the 'Medium support' peak moved
    (and Low/High boundaries moved with it).  'peak' is tuned on validation data."""
    p = {k: dict(v) for k, v in base.items()}
    p["S"] = {"Low": (0.0, 0.0, peak), "Medium": (max(peak - 0.35, 0.0), peak, min(peak + 0.35, 1.0)),
              "High": (peak, 1.0, 1.0)}
    return p


def firing_strengths(S, G, params=DEFAULT_PARAMS):
    """alpha: (N, C, R) firing strength of each rule for each class (AND = min)."""
    muS = {k: trimf(S, *v) for k, v in params["S"].items()}
    muG = {k: trimf(G, *v) for k, v in params["G"].items()}
    return np.stack([np.minimum(muS[s], muG[g]) for s, g, _ in RULES], axis=-1)


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------

class FuzzyFusion:
    name = "base"

    def __init__(self, params=None):
        self.params = params or DEFAULT_PARAMS

    def _defuzz(self, alpha, S, G):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def scores(self, P, W):
        """Return B (N, C) crisp class beliefs in [0, 1] and the firing strengths."""
        S, G = fuzzy_inputs(P, W)
        alpha = firing_strengths(S, G, self.params)
        B = self._defuzz(alpha, S, G)
        # if no rule fires for a class (cannot happen with these covering MFs) fall back to S
        none = alpha.sum(-1) <= 1e-12
        B = np.where(none, S, B)
        return B, alpha, S, G

    def predict(self, P, W):
        """Returns labels (UNAVAILABLE = -1 where all weights are zero) and the crisp
        beliefs B (N, C), each in [0, 1]."""
        valid = W.sum(1) > 0
        B, alpha, S, G = self.scores(P, W)
        keyed = B + 1e-6 * S          # deterministic tie-break by the support S
        y = np.where(valid, keyed.argmax(1), UNAVAILABLE)
        B = np.where(valid[:, None], B, np.nan)
        return y, B


class Mamdani(FuzzyFusion):
    name = "Mamdani"
    OUT = {"VL": (0.0, 0.0, 0.25), "L": (0.0, 0.25, 0.5), "M": (0.25, 0.5, 0.75),
           "H": (0.5, 0.75, 1.0), "VH": (0.75, 1.0, 1.0)}

    def __init__(self, params=None, grid=501):
        super().__init__(params)
        self.z = np.linspace(0.0, 1.0, grid)
        self.mu_out = np.stack([trimf(self.z, *self.OUT[o]) for _, _, o in RULES])  # (R, Z)

    def _defuzz(self, alpha, S, G):
        N, C, R = alpha.shape
        out = np.empty((N, C))
        a2 = alpha.reshape(-1, R)
        for i0 in range(0, a2.shape[0], 4096):  # chunked to bound memory
            a = a2[i0:i0 + 4096]
            clipped = np.minimum(a[:, :, None], self.mu_out[None])  # implication (min)
            agg = clipped.max(axis=1)                                  # aggregation (max)
            num = (agg * self.z).sum(1)
            den = agg.sum(1)
            out.reshape(-1)[i0:i0 + 4096] = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        return out


class Sugeno(FuzzyFusion):
    """Zero-order: z_k = constant.  First-order: z_k = p_k*S + q_k*G + r_k, with
    (p_k, q_k, r_k) estimated by least squares on validation data (Takagi-Sugeno
    identification; membership functions stay fixed, so this is NOT ANFIS)."""

    def __init__(self, params=None, order=0, ridge=1e-3):
        super().__init__(params)
        self.order = order
        self.ridge = ridge
        self.name = f"Sugeno-{order}"
        self.const = np.array([ANCHOR[o] for _, _, o in RULES])
        self.theta = None  # (R, 3) for first order

    def _design(self, alpha, S, G):
        wbar = alpha / np.maximum(alpha.sum(-1, keepdims=True), 1e-12)  # normalised strengths
        X = np.concatenate([wbar * S[..., None], wbar * G[..., None], wbar], axis=-1)
        return X.reshape(-1, X.shape[-1])

    def fit(self, P, W, y):
        """Fit first-order consequents z_k = pi_k S + rho_k G + tau_k by least squares
        (target 1 for the true class, 0 otherwise) SUBJECT TO 0 <= z_k <= 1 on the whole
        input domain [0,1]^2.  Because z_k is linear, it is bounded on the unit square
        iff it is bounded at the four corners, giving 4 linear constraints per rule.
        Since B_c is a convex combination of the z_k, this guarantees B_c in [0,1].
        (The unconstrained fit used in the previous version could produce B_c < 0.)"""
        if self.order == 0:
            return self
        from scipy.optimize import minimize
        S, G = fuzzy_inputs(P, W)
        alpha = firing_strengths(S, G, self.params)
        X = self._design(alpha, S, G)
        T = np.eye(alpha.shape[1])[y].reshape(-1).astype(float)
        R = alpha.shape[-1]
        A = X.T @ X + self.ridge * np.eye(3 * R); b = X.T @ T
        corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
        rows = []
        for k in range(R):
            for cs, cg in corners:
                r = np.zeros(3 * R); r[k] = cs; r[R + k] = cg; r[2 * R + k] = 1.0
                rows.append(r)
        Cm = np.array(rows)
        cons = [{"type": "ineq", "fun": lambda t: Cm @ t, "jac": lambda t: Cm},
                {"type": "ineq", "fun": lambda t: 1.0 - Cm @ t, "jac": lambda t: -Cm}]
        x0 = np.concatenate([np.zeros(2 * R), np.array([ANCHOR[o] for _, _, o in RULES])])
        res = minimize(lambda t: 0.5 * t @ A @ t - b @ t, x0, jac=lambda t: A @ t - b,
                       constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-10})
        sol = res.x
        self.fit_info = {"success": bool(res.success), "message": str(res.message)}
        self.theta = np.stack([sol[:R], sol[R:2 * R], sol[2 * R:]], axis=1)
        return self

    def _defuzz(self, alpha, S, G):
        wsum = np.maximum(alpha.sum(-1), 1e-12)
        if self.order == 0:
            z = np.broadcast_to(self.const, alpha.shape)
        else:
            if self.theta is None:
                raise RuntimeError("first-order Sugeno must be fitted first")
            z = (self.theta[:, 0] * S[..., None] + self.theta[:, 1] * G[..., None] + self.theta[:, 2])
        return (alpha * z).sum(-1) / wsum


class Tsukamoto(FuzzyFusion):
    """Each consequent is a monotonic linear ramp between the neutral value 0.5 and
    the level's anchor value v_k, so mu_k^{-1}(alpha) = 0.5 + alpha (v_k - 0.5).
    'Medium' (anchor 0.5) is a narrow increasing ramp from 0.4 to 0.5."""
    name = "Tsukamoto"

    def __init__(self, params=None):
        super().__init__(params)
        lo, hi = [], []
        for _, _, o in RULES:
            v = ANCHOR[o]
            if o == "M":
                lo.append(0.4); hi.append(0.5)
            else:
                lo.append(0.5); hi.append(v)
        self.start = np.array(lo)   # z where mu = 0
        self.end = np.array(hi)     # z where mu = 1

    def inverse(self, alpha):
        return self.start + alpha * (self.end - self.start)

    def _defuzz(self, alpha, S, G):
        z = self.inverse(alpha)
        return (alpha * z).sum(-1) / np.maximum(alpha.sum(-1), 1e-12)


# --------------------------------------------------------------------------
# Non-fuzzy reference rules (same weights and masks)
# --------------------------------------------------------------------------

def weighted_mean_predict(P, W):
    valid = W.sum(1) > 0
    S = np.einsum("nm,nmc->nc", W, np.nan_to_num(P))
    return np.where(valid, S.argmax(1), UNAVAILABLE), np.where(valid[:, None], S, np.nan)


def normalised_scores(B):
    """Class scores that sum to 1 (for ROC-AUC and display); B itself is a belief in [0,1]
    per class, not a calibrated probability."""
    B = np.clip(B, 0, None)
    s = B.sum(1, keepdims=True)
    return np.divide(B, s, out=np.full_like(B, 1 / B.shape[1]), where=s > 0)


def mean_predict(P, avail):
    W = np.asarray(avail, float) / np.asarray(avail, float).sum(1, keepdims=True)
    return weighted_mean_predict(P, W)
