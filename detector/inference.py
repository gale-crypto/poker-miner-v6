"""v6 serving path: shift-invariant features + a zero-parameter member.

Two things are combined here, for two different reasons.

  MODEL      a soft vote over the ported coherent/n-gram schema. That schema is
             the one measured to keep its ranking on live traffic (see
             coherent_features.py); it carries no absolute-currency quantity, so
             nothing in it moves when stakes change.

  LUCK       a zero-parameter heuristic (size-lattice discreteness + street-depth
             regularity + signature concentration). It has NO fitted parameters,
             so there is no training distribution for it to drift away from. It
             will never be the most accurate member offline; its job is to put a
             floor under live ranking when the learned member degrades.

The serving layer is carried over unchanged from v4/v5, because that part is
already proven live: MAX_POS_FRAC=0.05 keeps flagged chunks well inside the
fpr<=0.10 knee at any bot prevalence, and _positive_floor guarantees at least one
chunk over 0.5 so threshold_sanity_quality -- which gates the ENTIRE reward --
can never be zeroed by a mis-placed threshold.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from detector.coherent_features import chunk_features
from detector.luck import build_luck_detector

_ART = Path(__file__).resolve().parents[1] / "detector" / "artifacts"

MAX_POS_FRAC = float(os.environ.get("POKER44_MAX_POS_FRAC", "0.05"))
LUCK_WEIGHT = float(os.environ.get("POKER44_LUCK_WEIGHT", "-1"))  # <0 = use artifact
# Refuse to serve when artifact and code disagree about the schema by more than
# this. A renamed column silently reads as 0.0 otherwise, which looks like a
# working miner emitting quietly worse scores -- the worst failure mode there is.
_MAX_MISSING_FRAC = 0.05

_LUCK = build_luck_detector()


class SoftVote:
    """Probability-space soft vote. Pickled into the artifact, so this class must
    live in a shipped file to unpickle at serve time."""

    def __init__(self, members, cols, weights):
        self.members = list(members)
        self.cols = list(cols)
        self.weights = tuple(float(w) for w in weights)

    def score(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        total = sum(self.weights) or 1.0
        out = np.zeros(X.shape[0], dtype=float)
        for weight, member in zip(self.weights, self.members):
            out += weight * member.predict_proba(X)[:, 1]
        return out / total


def feature_matrix(chunks: List[List[Dict[str, Any]]], cols: List[str]) -> np.ndarray:
    feats = [chunk_features(c) for c in chunks]
    if feats and cols:
        missing = sum(1 for c in cols if c not in feats[0])
        if missing > _MAX_MISSING_FRAC * len(cols):
            raise RuntimeError(
                f"feature schema mismatch: {missing}/{len(cols)} expected columns "
                "are absent. The artifact was trained against a different schema; "
                "retrain rather than serving zeros for the missing columns."
            )
    return np.array([[float(f.get(c, 0.0)) for c in cols] for f in feats], dtype=float)


def luck_scores(chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
    out = []
    for chunk in chunks:
        try:
            out.append(float(_LUCK.score_chunk(chunk)))
        except Exception:
            out.append(0.5)          # a broken heuristic must not break serving
    return np.asarray(out, dtype=float)


def _remap_to_threshold(p: np.ndarray, t: float) -> np.ndarray:
    """Monotone piecewise-linear remap sending decision threshold t -> 0.5."""
    t = float(min(max(t, 1e-6), 1 - 1e-6))
    out = np.where(p >= t, 0.5 + 0.5 * (p - t) / (1 - t), 0.5 * p / t)
    return np.clip(out, 0.0, 1.0)


def _batch_safety_budget(scores: np.ndarray, max_frac: float) -> np.ndarray:
    """Cap the fraction of >=0.5 calls per batch WITHOUT changing the ranking.

    fpr@0.5 approaches fraction/(1-bot_share), so a batch that is mostly human
    turns a generous flag rate straight into threshold_sanity_quality decay.
    """
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0 or max_frac >= 1.0:
        return s
    k = max(1, int(np.floor(max_frac * n)))
    positive = np.flatnonzero(s >= 0.5)
    if positive.size <= k:
        return s
    order = positive[np.argsort(-s[positive], kind="stable")]
    demote = order[k:]
    out = s.copy()
    # Pack the demoted chunks just under 0.5, preserving their relative order.
    ranks = np.argsort(np.argsort(-out[demote], kind="stable"), kind="stable")
    out[demote] = 0.499 - 0.009 * (ranks / max(1, len(demote) - 1))
    return np.clip(out, 0.0, 1.0)


def _positive_floor(scores: np.ndarray, k_min: int = 1) -> np.ndarray:
    """Guarantee at least ``k_min`` chunks sit >= 0.5, order preserved.

    Zero true positives at 0.5 zeroes threshold_sanity_quality and with it the
    ENTIRE reward, not merely its 0.30 share. Lifting the top of the ranking
    across the line costs nothing: AP and recall@FPR are rank metrics.
    """
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0:
        return s
    k = max(1, min(int(k_min), n))
    if int((s >= 0.5).sum()) >= k:
        return s
    out = s.copy()
    for rank, idx in enumerate(np.argsort(-s, kind="stable")[:k]):
        out[idx] = 0.501 + 0.008 * (k - rank) / (k + 1.0)
    return np.clip(out, 0.0, 1.0)


class Detector:
    """Loads the trained artifact and scores validator batches."""

    def __init__(self, art_dir: Path | str = _ART):
        art_dir = Path(art_dir)
        art = joblib.load(art_dir / "model.joblib")
        self.vote: SoftVote = art["vote"]
        self.cols = self.vote.cols
        self.luck_weight = (
            LUCK_WEIGHT if LUCK_WEIGHT >= 0.0 else float(art.get("luck_weight", 0.0))
        )
        self.threshold = float(art.get("deploy_threshold", 0.5))
        with open(art_dir / "meta.json") as fh:
            self.meta = json.load(fh)

    def combined(self, chunks) -> np.ndarray:
        p_model = self.vote.score(feature_matrix(chunks, self.cols))
        if self.luck_weight <= 0.0:
            return p_model
        w = self.luck_weight
        return (1.0 - w) * p_model + w * luck_scores(chunks)

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        s = _remap_to_threshold(self.combined(chunks), self.threshold)
        s = _batch_safety_budget(s, MAX_POS_FRAC)
        s = _positive_floor(s)
        return [0.1 if not chunk else round(float(v), 6)
                for chunk, v in zip(chunks, s)]


_SINGLETON: Detector | None = None


def get_model() -> Detector:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Detector()
    return _SINGLETON
