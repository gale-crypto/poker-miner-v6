"""Sequence-signature + sizing-dispersion luck detector (variant S1-RW).

Scores a miner-visible chunk primarily by how much its hands share a common
*action sequence signature* (the proven signal the top fork already ranks well
on), and secondarily by the *continuous dispersion of voluntary bet sizes*. A
scripted seat replays a small number of decision templates, so its hands both
collapse onto a handful of street/action n-gram signatures AND settle on a few
exact bet sizes; human play spreads across many distinct sequences and jitters
its sizing continuously. The signature-concentration core stays the dominant
term; the size-dispersion term only re-orders chunks the core leaves ambiguous
and closes the detector's known blind spot to seats that jitter their token
string but keep anomalously tight continuous sizing.

PROVENANCE. Ported verbatim from poker44_ml/luck_detector.py in
code-seqsig-rw-detector-1 (MIT), replacing the T4-RW variant this repo shipped
previously. The two upstream siblings are byte-identical apart from this one
file and scored 0.690 (S1-RW) against 0.647 (T4-RW) live, which is as close to a
controlled comparison as the leaderboard offers.

WHY THIS VARIANT TRANSFERS BETTER. T4-RW's size tell counts distinct sizes after
``round(v, 1)``; that lattice means something different at the benchmark's
~240bb stacks than at live's pinned 100bb, so it partly measures the domain
rather than the behaviour. S1-RW replaces it with a coefficient of variation
(std/mean over winsorized voluntary sizes), which is invariant to any global
rescaling of bet sizes -- the same no-absolute-currency property that makes the
leading miner's schema survive the shift. Measured on captured live payloads it
matches that miner's flagged set 5/5 against T4-RW's 4/5.

Here it is one weighted member of the served blend, not the serving backend it
is upstream. It still has zero fitted parameters, so there is no training
distribution for it to drift away from.

Fork fine-tune (RW / winsorized-robust): continuous statistics are winsorized
or Laplace-smoothed before use, hardening every term against the single-hand
outliers that live validator snapshots carry far more often than the released
benchmark does; the concentration core also reads a wider repeat structure.

Contract: ``score_chunk(chunk) -> float in [0, 1]``, higher == more bot-like.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional

PROFILE = "sequence-signature-sd-rw"
VARIANT_TAG = "S1-RW"

_ACTION_CODE = {
    "fold": "F",
    "check": "K",
    "call": "C",
    "bet": "B",
    "raise": "R",
    "allin": "A",
    "all_in": "A",
}
_STREET_CODE = {"preflop": "p", "flop": "f", "turn": "t", "river": "r"}
_VOLUNTARY = {"bet", "raise", "allin", "all_in"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _winsorize(xs: List[float], q: float) -> List[float]:
    """Clip a sample to its [q, 1-q] empirical quantiles (robust-stat guard)."""
    if len(xs) < 3 or q <= 0.0:
        return list(xs)
    srt = sorted(xs)
    k = max(0, min(len(srt) - 1, int(q * (len(srt) - 1))))
    lo, hi = srt[k], srt[len(srt) - 1 - k]
    return [lo if x < lo else hi if x > hi else x for x in xs]


def _size_code(amt: float) -> str:
    if amt <= 0:
        return "0"
    if amt <= 1.0:
        return "s"
    if amt <= 3.0:
        return "m"
    if amt <= 8.0:
        return "l"
    return "x"


class LuckDetector:
    """Signature-concentration + size-dispersion bot detector (variant S1-RW)."""

    PROFILE = PROFILE

    def __init__(
        self,
        *,
        low_anchor: float = 0.29,
        high_anchor: float = 0.88,
        street_weight: float = 0.15,
        disp_weight: float = 0.20,
        cv_ref: float = 0.80,
        floor: float = 0.05,
        trim_q: float = 0.10,
    ) -> None:
        self.low_anchor = low_anchor
        self.high_anchor = high_anchor
        self.street_weight = street_weight
        self.disp_weight = disp_weight
        self.cv_ref = cv_ref
        self.floor = floor
        self.trim_q = trim_q

    @classmethod
    def from_env(cls) -> "LuckDetector":
        return cls(
            low_anchor=_num(os.getenv("LUCK_S_LOW_ANCHOR"), 0.29),
            high_anchor=_num(os.getenv("LUCK_S_HIGH_ANCHOR"), 0.88),
            street_weight=_num(os.getenv("LUCK_S_STREET_WEIGHT"), 0.15),
            disp_weight=_num(os.getenv("LUCK_S_DISP_WEIGHT"), 0.20),
            cv_ref=_num(os.getenv("LUCK_S_CV_REF"), 0.80),
            floor=_num(os.getenv("LUCK_S_FLOOR"), 0.05),
            trim_q=_num(os.getenv("LUCK_S_TRIM_Q"), 0.10),
        )

    def _hand_signature(self, hand: dict) -> str:
        actions = hand.get("actions") or []
        tokens: List[str] = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            st = _STREET_CODE.get(str(a.get("street", "")).lower(), "?")
            ac = _ACTION_CODE.get(str(a.get("action_type", "")).lower(), "?")
            sz = _size_code(_num(a.get("normalized_amount_bb"), _num(a.get("amount"))))
            tokens.append(f"{st}{ac}{sz}")
        street_shape = "".join(
            _STREET_CODE.get(str(s.get("street", "")).lower(), "?")
            for s in (hand.get("streets") or [])
            if isinstance(s, dict)
        )
        return f"{street_shape}#{'.'.join(tokens)}"

    def _street_uniformity(self, hands: List[dict]) -> float:
        shapes = Counter(
            "".join(
                _STREET_CODE.get(str(s.get("street", "")).lower(), "?")
                for s in (h.get("streets") or [])
                if isinstance(s, dict)
            )
            for h in hands
        )
        if not shapes:
            return 0.0
        total = sum(shapes.values())
        return max(shapes.values()) / total

    def _size_dispersion_deficit(self, hands: List[dict]) -> Optional[float]:
        """1 - coefficient-of-variation of voluntary bet/raise sizes (chunk-level).

        Returns None when there are too few voluntary sizes to be meaningful, so
        the caller folds this term's weight back onto the concentration core.
        """
        sizes: List[float] = []
        for h in hands:
            for a in h.get("actions") or []:
                if not isinstance(a, dict):
                    continue
                if str(a.get("action_type", "")).lower() in _VOLUNTARY:
                    v = _num(a.get("normalized_amount_bb"), _num(a.get("amount")))
                    if v > 0:
                        sizes.append(v)
        if len(sizes) < 5:
            return None
        # RW: winsorize before the CV so one outlier size cannot mask tightness.
        sizes = _winsorize(sizes, self.trim_q)
        mean = sum(sizes) / len(sizes)
        if mean <= 0:
            return None
        var = sum((s - mean) ** 2 for s in sizes) / len(sizes)
        cv = math.sqrt(var) / mean
        return _clamp01(1.0 - cv / max(self.cv_ref, 1e-6))

    def score_chunk(self, chunk: List[dict]) -> float:
        hands = [h for h in (chunk or []) if isinstance(h, dict)]
        if not hands:
            return 0.5
        n = len(hands)
        sig_counts = Counter(self._hand_signature(h) for h in hands)

        counts = sorted(sig_counts.values(), reverse=True)
        top_share = counts[0] / n
        top2_share = sum(counts[:2]) / n
        unique_share = len(sig_counts) / n
        repeat_mass = sum(c for c in counts if c >= 2) / n

        # RW concentration mix: part of the top-1 term moves onto a top-2 share
        # so a script alternating two templates is caught as firmly as one.
        concentration = _clamp01(
            0.30 * top_share
            + 0.15 * top2_share
            + 0.35 * repeat_mass
            + 0.20 * (1.0 - unique_share)
        )
        street_uni = self._street_uniformity(hands)

        disp = self._size_dispersion_deficit(hands)
        sw = self.street_weight
        dw = self.disp_weight if disp is not None else 0.0
        cw = max(0.0, 1.0 - sw - dw)
        raw = _clamp01(cw * concentration + sw * street_uni + dw * (disp or 0.0))

        # Piecewise-linear (anchor) calibration: maps [low_anchor, high_anchor]
        # onto ~[0.5, 1.0] so concentrated (replayed) chunks cross 0.5.
        if raw <= self.low_anchor:
            out = self.floor + (0.5 - self.floor) * (raw / max(self.low_anchor, 1e-6))
        elif raw >= self.high_anchor:
            out = 1.0
        else:
            out = 0.5 + 0.5 * (raw - self.low_anchor) / max(self.high_anchor - self.low_anchor, 1e-6)
        return round(_clamp01(out), 6)

    def score_chunks(self, chunks: List[List[dict]]) -> List[float]:
        return [self.score_chunk(list(c or [])) for c in (chunks or [])]

    def debug_components(self, chunks: List[List[dict]]) -> Dict[str, List[float]]:
        top_out, uniq_out, disp_out = [], [], []
        for c in chunks or []:
            hands = [h for h in (c or []) if isinstance(h, dict)]
            if not hands:
                top_out.append(0.0)
                uniq_out.append(1.0)
                disp_out.append(0.0)
                continue
            sig_counts = Counter(self._hand_signature(h) for h in hands)
            top_out.append(max(sig_counts.values()) / len(hands))
            uniq_out.append(len(sig_counts) / len(hands))
            disp_out.append(self._size_dispersion_deficit(hands) or 0.0)
        return {"sig_top_share": top_out, "sig_unique_share": uniq_out, "size_disp_deficit": disp_out}


def build_luck_detector() -> "LuckDetector":
    return LuckDetector.from_env()
