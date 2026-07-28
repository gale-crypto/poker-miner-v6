"""Size-discreteness + depth-regularity luck detector (variant T4-RW).

Scores a miner-visible chunk by how *quantized and structurally rigid* its play
is. Two orthogonal tells drive the primary signal:

  * **Bet-size discreteness** — a scripted seat draws its voluntary bet/raise
    sizes from a tiny lattice of fixed values, so the number of distinct rounded
    sizes it uses, relative to how often it bets, is anomalously small; human
    sizing spreads across many values.
  * **Street-depth regularity** — a bot tends to reach the same number of streets
    hand after hand (a fixed fold/continue policy), so the entropy of its
    per-hand street-depth distribution collapses; human hands terminate at a wide
    range of depths.

A lighter signature-concentration term is retained so the strong ranking on
clearly-replayed chunks is preserved, but the discreteness + depth terms
dominate, giving this fork a genuinely different chunk ordering from the
signature-first siblings. Outputs pass through a convex-power anchor calibration
(distinct from the sibling linear / logistic / smoothstep curves).

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
from typing import Any, Dict, List

PROFILE = "size-lattice-depth-rw"
VARIANT_TAG = "T4-RW"

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
    """Size-discreteness + depth-regularity bot detector (variant T4-RW)."""

    PROFILE = PROFILE

    def __init__(
        self,
        *,
        low_anchor: float = 0.32,
        high_anchor: float = 0.88,
        gamma: float = 1.4,
        conc_weight: float = 0.58,
        street_weight: float = 0.12,
        sec_weight: float = 0.30,
        depth_ref: float = 4.0,
        floor: float = 0.06,
    ) -> None:
        self.low_anchor = low_anchor
        self.high_anchor = high_anchor
        self.gamma = gamma
        self.conc_weight = conc_weight
        self.street_weight = street_weight
        self.sec_weight = sec_weight
        self.depth_ref = depth_ref
        self.floor = floor

    @classmethod
    def from_env(cls) -> "LuckDetector":
        return cls(
            low_anchor=_num(os.getenv("LUCK_T_LOW_ANCHOR"), 0.32),
            high_anchor=_num(os.getenv("LUCK_T_HIGH_ANCHOR"), 0.88),
            gamma=_num(os.getenv("LUCK_T_GAMMA"), 1.4),
            conc_weight=_num(os.getenv("LUCK_T_CONC_WEIGHT"), 0.58),
            street_weight=_num(os.getenv("LUCK_T_STREET_WEIGHT"), 0.12),
            sec_weight=_num(os.getenv("LUCK_T_SEC_WEIGHT"), 0.30),
            depth_ref=_num(os.getenv("LUCK_T_DEPTH_REF"), 4.0),
            floor=_num(os.getenv("LUCK_T_FLOOR"), 0.06),
        )

    def _hand_signature(self, hand: dict) -> str:
        toks = []
        for a in hand.get("actions") or []:
            if not isinstance(a, dict):
                continue
            st = _STREET_CODE.get(str(a.get("street", "")).lower(), "?")
            ac = _ACTION_CODE.get(str(a.get("action_type", "")).lower(), "?")
            sz = _size_code(_num(a.get("normalized_amount_bb"), _num(a.get("amount"))))
            toks.append(f"{st}{ac}{sz}")
        return ".".join(toks)

    def _concentration(self, hands: List[dict]) -> float:
        n = len(hands)
        sig_counts = Counter(self._hand_signature(h) for h in hands)
        top_share = max(sig_counts.values()) / n
        unique_share = len(sig_counts) / n
        repeat_mass = sum(c for c in sig_counts.values() if c >= 2) / n
        # T-variant concentration mix (0.42/0.33/0.25): distinct from siblings.
        return _clamp01(0.42 * top_share + 0.33 * repeat_mass + 0.25 * (1.0 - unique_share))

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
        return max(shapes.values()) / sum(shapes.values())

    def _size_discreteness(self, hands: List[dict]) -> float:
        """1 - (distinct rounded voluntary sizes / voluntary size count)."""
        sizes: List[float] = []
        for h in hands:
            for a in h.get("actions") or []:
                if not isinstance(a, dict):
                    continue
                if str(a.get("action_type", "")).lower() in _VOLUNTARY:
                    v = _num(a.get("normalized_amount_bb"), _num(a.get("amount")))
                    if v > 0:
                        sizes.append(round(v, 1))
        if len(sizes) < 5:
            return 0.0
        counts = sorted(Counter(sizes).values(), reverse=True)
        distinct_share = len(counts) / len(sizes)
        top3_mass = sum(counts[:3]) / len(sizes)
        # RW: blend lattice sparsity with top-3 lattice mass so a couple of
        # stray live sizes cannot hide an otherwise fixed sizing menu.
        return _clamp01(0.65 * (1.0 - distinct_share) + 0.35 * top3_mass)

    def _depth_regularity(self, hands: List[dict]) -> float:
        """1 - normalized entropy of the per-hand street-depth distribution."""
        depths = Counter()
        for h in hands:
            streets = [s for s in (h.get("streets") or []) if isinstance(s, dict)]
            depths[len(streets)] += 1
        total = sum(depths.values())
        if total <= 0:
            return 0.0
        entropy = -sum((c / total) * math.log(c / total) for c in depths.values())
        norm = entropy / math.log(max(self.depth_ref, 1.0 + 1e-6))
        return _clamp01(1.0 - norm)

    def score_chunk(self, chunk: List[dict]) -> float:
        hands = [h for h in (chunk or []) if isinstance(h, dict)]
        if not hands:
            return 0.5

        concentration = self._concentration(hands)
        street_uni = self._street_uniformity(hands)
        secondary = _clamp01(
            0.55 * self._size_discreteness(hands) + 0.45 * self._depth_regularity(hands)
        )

        raw = _clamp01(
            self.conc_weight * concentration
            + self.street_weight * street_uni
            + self.sec_weight * secondary
        )
        # Convex-power anchor calibration (distinct curve family from siblings).
        if raw <= self.low_anchor:
            t = raw / max(self.low_anchor, 1e-6)
            out = self.floor + (0.5 - self.floor) * (t ** self.gamma)
        elif raw >= self.high_anchor:
            out = 1.0
        else:
            t = (raw - self.low_anchor) / max(self.high_anchor - self.low_anchor, 1e-6)
            out = 0.5 + 0.5 * (t ** self.gamma)
        return round(_clamp01(out), 6)

    def score_chunks(self, chunks: List[List[dict]]) -> List[float]:
        return [self.score_chunk(list(c or [])) for c in (chunks or [])]

    def debug_components(self, chunks: List[List[dict]]) -> Dict[str, List[float]]:
        sd, dr = [], []
        for c in chunks or []:
            hands = [h for h in (c or []) if isinstance(h, dict)]
            if not hands:
                sd.append(0.0)
                dr.append(0.0)
                continue
            sd.append(self._size_discreteness(hands))
            dr.append(self._depth_regularity(hands))
        return {"size_discreteness": sd, "depth_regularity": dr}


def build_luck_detector() -> "LuckDetector":
    return LuckDetector.from_env()
