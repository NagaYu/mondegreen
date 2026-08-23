"""ConservativeGate: the component whose job is to say *no*.

The hard phonetic constraint decides what a span is *allowed* to become.  The
gate decides whether to touch the span at all.  Splitting the two is the whole
trick: the constraint bounds the worst case, and the gate buys back precision
inside that bound without ever being able to escape it.

The gate is a calibrated logistic regression over cheap, interpretable span
features.  It is deliberately not a neural network:

* it must run in microseconds per span on a laptop (LOCAL-SPEED);
* its threshold has to be a knob the user can turn to trade correction rate
  against damage rate, and read off a curve (LOW-DAMAGE);
* every feature has to be explainable in the Space's evidence panel.

Untrained, it falls back to :class:`HeuristicGate`, a hand-set scoring rule, so
the system is safe out of the box.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .types import SpanDecision

#: Canonical feature order.  Persisted with the model so a saved gate can never
#: be fed features in the wrong order.
FEATURE_NAMES: Tuple[str, ...] = (
    "norm_distance",
    "margin",
    "exact_reading_match",
    "span_morae",
    "term_morae",
    "mora_ratio",
    "span_chars",
    "n_candidates",
    "unknown_ratio",
    "reading_ambiguity",
    "log_term_weight",
    "span_has_kanji",
    "span_has_katakana",
    "span_all_hiragana",
    "span_is_glossary_surface",
    "span_is_common_word",
    "span_is_proper_noun",
    "boundary_left",
    "boundary_right",
    "lm_delta",
)


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Project a feature dict onto :data:`FEATURE_NAMES` in canonical order.

    Claim: SUPPORT -- feature-order bugs are silent and would corrupt the
    LOW-DAMAGE numbers.
    """
    return np.array([float(features.get(n, 0.0)) for n in FEATURE_NAMES], dtype=np.float64)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

        Claim: LOW-DAMAGE -- an overflowing sigmoid would silently mis-calibrate the
        gate, and calibration is what makes its threshold a safety knob.
        """
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class HeuristicGate:
    """Hand-set fallback used before any training data exists.

    Encodes the three things that reliably predict a safe edit: the span sounds
    almost exactly like the term, the runner-up is far away, and the span is long
    enough that the match cannot be a coincidence.

    Claim: LOW-DAMAGE -- an untrained system must still be conservative, not
    merely permissive.
    """

    name = "heuristic"
    is_trained = False

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialise the gate, falling back to the heuristic until trained.

                Claim: LOW-DAMAGE.
                """
        self.threshold = threshold

    def predict_proba_one(self, features: Dict[str, float]) -> float:
        """Score one span in [0, 1]; higher means "safe to replace".

        Claim: LOW-DAMAGE.
        """
        if features.get("span_is_glossary_surface", 0.0) >= 1.0:
            return 0.0
        d = features.get("norm_distance", 1.0)
        margin = features.get("margin", 0.0)
        morae = features.get("span_morae", 0.0)
        unknown = features.get("unknown_ratio", 0.0)
        ambiguity = features.get("reading_ambiguity", 1.0)
        z = 3.2 - 11.0 * d + 2.4 * margin + 0.35 * min(morae, 8.0)
        z -= 3.0 * unknown
        z -= 0.12 * max(0.0, ambiguity - 1.0)
        z -= 1.4 * max(0.0, 2.0 - morae)          # 1-mora spans are almost never safe
        z += 0.6 * features.get("span_has_katakana", 0.0)
        z += 0.4 * features.get("boundary_left", 0.0) * features.get("boundary_right", 0.0)
        # Rewriting an ordinary word of Japanese is the expensive mistake; rewriting
        # a proper noun is what the glossary is for.
        z -= 2.6 * features.get("span_is_common_word", 0.0)
        z += 0.8 * features.get("span_is_proper_noun", 0.0)
        return float(1.0 / (1.0 + math.exp(-z)))

    def decide(self, features: Dict[str, float]) -> Tuple[float, bool]:
        """Claim: LOW-DAMAGE."""
        p = self.predict_proba_one(features)
        return p, p >= self.threshold

    def to_dict(self) -> Dict[str, object]:
        """Claim: SUPPORT."""
        return {"kind": "heuristic", "threshold": self.threshold}


@dataclass
class GateTrainingReport:
    """Diagnostics from :meth:`ConservativeGate.fit`."""

    n_train: int = 0
    n_positive: int = 0
    accuracy: float = 0.0
    auc: float = 0.0
    log_loss: float = 0.0
    ece: float = 0.0
    platt: Tuple[float, float] = (1.0, 0.0)
    feature_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Claim: SUPPORT."""
        return {
            "n_train": self.n_train,
            "n_positive": self.n_positive,
            "accuracy": self.accuracy,
            "auc": self.auc,
            "log_loss": self.log_loss,
            "ece": self.ece,
            "platt": list(self.platt),
            "feature_weights": self.feature_weights,
        }


class ConservativeGate:
    """Calibrated per-span "correct it / leave it alone" classifier.

    Claim: LOW-DAMAGE -- this is the component that produces metric (3) and the
    correction-rate vs damage-rate curve.
    """

    name = "logistic"

    def __init__(
        self,
        weights: Optional[np.ndarray] = None,
        bias: float = 0.0,
        threshold: float = 0.5,
        platt: Tuple[float, float] = (1.0, 0.0),
        mean: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        """Initialise the gate, falling back to the heuristic until trained.

                Claim: LOW-DAMAGE.
                """
        d = len(FEATURE_NAMES)
        self.weights = np.zeros(d) if weights is None else np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        self.threshold = float(threshold)
        self.platt = (float(platt[0]), float(platt[1]))
        self.mean = np.zeros(d) if mean is None else np.asarray(mean, dtype=np.float64)
        self.scale = np.ones(d) if scale is None else np.asarray(scale, dtype=np.float64)
        self.metadata: Dict[str, object] = dict(metadata or {})
        self.is_trained = bool(np.any(self.weights))
        self._fallback = HeuristicGate(threshold)

    # ------------------------------------------------------------------ score
    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Apply the training-time feature mean and scale.

            Claim: LOW-DAMAGE.
            """
        return (X - self.mean) / np.where(self.scale == 0, 1.0, self.scale)

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Pre-calibration decision values.

        Claim: SUPPORT.
        """
        return self._standardize(X) @ self.weights + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated P(replacing this span moves us toward the gold text).

        Claim: LOW-DAMAGE -- a threshold on an *uncalibrated* score is not a
        usable safety knob, so calibration is part of the contract.
        """
        a, b = self.platt
        return _sigmoid(a * self.raw_scores(np.atleast_2d(X)) + b)

    def predict_proba_one(self, features: Dict[str, float]) -> float:
        """Score a single span from its feature dict.

        Claim: LOW-DAMAGE.
        """
        if features.get("span_is_glossary_surface", 0.0) >= 1.0:
            return 0.0
        if not self.is_trained:
            return self._fallback.predict_proba_one(features)
        return float(self.predict_proba(features_to_vector(features))[0])

    def decide(self, features: Dict[str, float]) -> Tuple[float, bool]:
        """Return ``(probability, accept)`` for one span.

        Claim: LOW-DAMAGE.
        """
        p = self.predict_proba_one(features)
        return p, p >= self.threshold

    # -------------------------------------------------------------------- fit
    def fit(
        self,
        decisions: Sequence[SpanDecision],
        l2: float = 1.0,
        epochs: int = 400,
        lr: float = 0.5,
        class_weight_negative: float = 3.0,
        holdout: float = 0.25,
        seed: int = 0,
    ) -> GateTrainingReport:
        """Train on labelled span decisions, then Platt-calibrate on a held-out split.

        Negatives (spans that should be left alone) get ``class_weight_negative``
        times the weight of positives.  That asymmetry is intentional and is the
        statistical expression of "breaking correct text is worse than missing a
        fix".

        Claim: LOW-DAMAGE.
        """
        if not decisions:
            raise ValueError("no training decisions provided")
        X = np.stack([features_to_vector(d.features) for d in decisions])
        y = np.array([int(d.label) for d in decisions], dtype=np.float64)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y))
        n_hold = max(1, int(len(y) * holdout)) if len(y) > 8 else 0
        hold_idx, tr_idx = idx[:n_hold], idx[n_hold:]
        if len(tr_idx) == 0:
            tr_idx, hold_idx = idx, np.array([], dtype=int)

        Xtr, ytr = X[tr_idx], y[tr_idx]
        self.mean = Xtr.mean(axis=0)
        self.scale = Xtr.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        Ztr = self._standardize(Xtr)

        w = np.zeros(Ztr.shape[1])
        b = 0.0
        sample_w = np.where(ytr > 0.5, 1.0, class_weight_negative)
        sample_w = sample_w / sample_w.mean()
        n = len(ytr)
        for epoch in range(epochs):
            p = _sigmoid(Ztr @ w + b)
            err = (p - ytr) * sample_w
            gw = (Ztr.T @ err) / n + l2 * w / n
            gb = err.mean()
            step = lr / (1.0 + 0.01 * epoch)
            w -= step * gw
            b -= step * gb
        self.weights, self.bias = w, float(b)
        self.is_trained = True

        # --- Platt calibration on the held-out split -------------------------
        if len(hold_idx) >= 8:
            s = self.raw_scores(X[hold_idx])
            yh = y[hold_idx]
            a, c = 1.0, 0.0
            for epoch in range(600):
                p = _sigmoid(a * s + c)
                err = p - yh
                a -= 0.05 * float((err * s).mean())
                c -= 0.05 * float(err.mean())
            self.platt = (float(a), float(c))
        else:
            self.platt = (1.0, 0.0)

        probs = self.predict_proba(X)
        report = GateTrainingReport(
            n_train=int(len(tr_idx)),
            n_positive=int(y.sum()),
            accuracy=float(((probs >= 0.5).astype(float) == y).mean()),
            auc=roc_auc(y, probs),
            log_loss=float(-np.mean(y * np.log(probs + 1e-12) + (1 - y) * np.log(1 - probs + 1e-12))),
            ece=expected_calibration_error(y, probs),
            platt=self.platt,
            feature_weights={n_: float(v) for n_, v in zip(FEATURE_NAMES, self.weights)},
        )
        self.metadata["training"] = report.to_dict()
        return report

    # ----------------------------------------------------------------- persist
    def save(self, path: str) -> None:
        """Persist to JSON (no pickle: the gate ships inside a model repo).

        Claim: SUPPORT.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "version": 1,
                    "kind": "logistic",
                    "features": list(FEATURE_NAMES),
                    "weights": self.weights.tolist(),
                    "bias": self.bias,
                    "threshold": self.threshold,
                    "platt": list(self.platt),
                    "mean": self.mean.tolist(),
                    "scale": self.scale.tolist(),
                    "metadata": self.metadata,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "ConservativeGate":
        """Load a gate, refusing files whose feature order does not match.

        Claim: LOW-DAMAGE -- silently mismatched features would degrade the gate
        into noise while still reporting probabilities.
        """
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        feats = tuple(d.get("features", ()))
        if feats and feats != FEATURE_NAMES:
            raise ValueError(
                f"gate was trained on a different feature set: {feats} != {FEATURE_NAMES}"
            )
        return cls(
            weights=np.array(d["weights"], dtype=np.float64),
            bias=d.get("bias", 0.0),
            threshold=d.get("threshold", 0.5),
            platt=tuple(d.get("platt", (1.0, 0.0))),  # type: ignore[arg-type]
            mean=np.array(d.get("mean", [0.0] * len(FEATURE_NAMES))),
            scale=np.array(d.get("scale", [1.0] * len(FEATURE_NAMES))),
            metadata=d.get("metadata", {}),
        )


# --------------------------------------------------------------------------------------
# Diagnostics and the threshold sweep
# --------------------------------------------------------------------------------------

def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC; 0.5 when one class is absent.

    Claim: SUPPORT.
    """
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    vals = np.concatenate([pos, neg])
    srt = np.sort(vals)
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[np.isin(vals, srt[i])] = avg
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE over equal-width probability bins.

    Claim: LOW-DAMAGE -- a threshold is only a safety knob if the probabilities
    behind it mean what they say.
    """
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    if len(y) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


@dataclass
class SweepPoint:
    """One point on the correction-rate vs damage-rate curve."""

    threshold: float
    correction_rate: float
    damage_rate: float
    edit_damage_rate: float
    accepted: int
    repairs: int
    damages: int
    neutrals: int

    def to_dict(self) -> Dict[str, float]:
        """Claim: SUPPORT."""
        return {
            "threshold": self.threshold,
            "correction_rate": self.correction_rate,
            "damage_rate": self.damage_rate,
            "edit_damage_rate": self.edit_damage_rate,
            "accepted": float(self.accepted),
            "repairs": float(self.repairs),
            "damages": float(self.damages),
            "neutrals": float(self.neutrals),
        }


def sweep_thresholds(
    probs: Sequence[float],
    outcomes: Sequence[str],
    thresholds: Optional[Sequence[float]] = None,
) -> List[SweepPoint]:
    """Trace correction rate against damage rate as the gate threshold moves.

    ``outcomes`` labels each *candidate* edit with what would happen if it were
    applied: ``repair`` / ``damage`` / ``neutral`` / ``no-op``.  The denominators
    are fixed across the sweep:

    * correction rate = accepted repairs / all available repairs
    * damage rate     = accepted damages / all spans that were already correct

    Claim: LOW-DAMAGE -- this function produces the second headline figure, and
    it is how a user picks their own operating point.
    """
    probs = list(probs)
    outcomes = list(outcomes)
    if thresholds is None:
        thresholds = [i / 100.0 for i in range(0, 101)]
    n_repairable = sum(1 for o in outcomes if o == "repair")
    n_protected = sum(1 for o in outcomes if o in ("damage", "no-op"))
    points: List[SweepPoint] = []
    for t in thresholds:
        acc = [o for p, o in zip(probs, outcomes) if p >= t]
        repairs = sum(1 for o in acc if o == "repair")
        damages = sum(1 for o in acc if o == "damage")
        neutrals = sum(1 for o in acc if o not in ("repair", "damage"))
        points.append(
            SweepPoint(
                threshold=float(t),
                correction_rate=(repairs / n_repairable) if n_repairable else 0.0,
                damage_rate=(damages / n_protected) if n_protected else 0.0,
                edit_damage_rate=(damages / len(acc)) if acc else 0.0,
                accepted=len(acc),
                repairs=repairs,
                damages=damages,
                neutrals=neutrals,
            )
        )
    return points


def pick_threshold(
    points: Sequence[SweepPoint],
    max_damage_rate: float = 0.01,
    tolerance: float = 0.02,
) -> float:
    """Buy the largest safety margin that costs almost no correction rate.

    Naively "maximise correction subject to the damage budget" is the wrong rule
    here, and it fails in a specific, measurable way: on a corpus where the hard
    constraint already achieves zero damage at *every* threshold, that rule
    selects threshold 0.0 -- shipping a gate that never refuses anything, and
    therefore no margin at all when the data turns out to be harder than the
    benchmark.  (Measured: with threshold 0.0 the term damage rate on a
    partial-coverage evaluation glossary was 4.8%; at 0.9 it was 1.0%, for 2%
    relative correction rate.)

    So: among points meeting the budget, find the best correction rate, then
    accept any point within ``tolerance`` *relative* of it and take the highest
    threshold among those.  Safety that is nearly free should be taken.

    If the budget cannot be met at all, returns the most conservative point --
    refusing to correct is the right failure mode.

    Claim: LOW-DAMAGE.
    """
    ok = [p for p in points if p.damage_rate <= max_damage_rate]
    if not ok:
        return max((p.threshold for p in points), default=1.0)
    best_rate = max(p.correction_rate for p in ok)
    if best_rate <= 0.0:
        return max(ok, key=lambda p: p.threshold).threshold
    floor = best_rate * (1.0 - tolerance) - 1e-9
    plateau = [p for p in ok if p.correction_rate >= floor]
    return max(plateau, key=lambda p: p.threshold).threshold
