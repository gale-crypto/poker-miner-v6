"""Chunk feature schema ported from poker44-first (MIT, (c) 2026 TakashiAkio).

WHY THIS SCHEMA. On captured live payloads our v4/v5 schema ranks almost
identically to poker44-first on the benchmark (Spearman 0.91-0.93) but diverges
to ~0.69 on live traffic, and they score 0.6478 live against our 0.53-0.58. The
difference is not the learner: this schema contains NO absolute-currency
quantity. Every feature is a share, an entropy, a rate, a run length, an n-gram
frequency, or a quantile of a within-chunk distribution -- all invariant when the
benchmark's ~240bb stacks become live's pinned 100bb. Nothing to normalize after
the fact, nothing to drop.

Ported verbatim apart from the import path, so it can be diffed against upstream.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from detector.coherence import coherent_feature_dict


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    if total <= 0.0 or len(counts) <= 1:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log(p + 1e-12)
    return _safe_div(ent, math.log(len(counts)))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _mean(values: list[float]) -> float:
    return _safe_div(sum(values), len(values))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(max(0.0, _mean([(v - m) * (v - m) for v in values])))


def _max_run_share(values: list[Any]) -> float:
    if not values:
        return 0.0
    longest = 1
    cur = 1
    for prev, cur_value in zip(values, values[1:]):
        if prev == cur_value:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return _safe_div(longest, len(values))


def _amount_bucket(value: float) -> str:
    if value <= 0.0:
        return "z"
    if value <= 0.5:
        return "xs"
    if value <= 1.0:
        return "s"
    if value <= 2.0:
        return "m"
    if value <= 5.0:
        return "l"
    return "xl"


def _hand_features(hand: dict[str, Any]) -> dict[str, float]:
    metadata = hand.get("metadata") or {}
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    actions = hand.get("actions") or []

    max_seats = max(1, _safe_int(metadata.get("max_seats"), 6))
    hero_seat = _safe_int(metadata.get("hero_seat"), 0)
    button_seat = _safe_int(metadata.get("button_seat"), 0)
    player_count = float(len(players))
    street_count = float(len(streets))
    action_count = float(len(actions))

    action_types: list[str] = []
    actor_seats: list[int] = []
    street_names: list[str] = []
    amount_bb: list[float] = []
    pot_before: list[float] = []
    pot_after: list[float] = []
    stack_bb: list[float] = []
    raise_to_present = 0
    call_to_present = 0

    for player in players:
        if not isinstance(player, dict):
            continue
        stack_bb.append(_safe_div(_safe_float(player.get("starting_stack"), 0.0), 0.02))

    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("action_type") or "").lower().strip()
        actor = _safe_int(action.get("actor_seat"), 0)
        street = str(action.get("street") or "").lower().strip()
        amt = _safe_float(action.get("normalized_amount_bb"), 0.0)
        pb = _safe_div(_safe_float(action.get("pot_before"), 0.0), 0.02)
        pa = _safe_div(_safe_float(action.get("pot_after"), 0.0), 0.02)

        action_types.append(action_type)
        if actor > 0:
            actor_seats.append(actor)
        street_names.append(street)
        amount_bb.append(max(0.0, amt))
        pot_before.append(max(0.0, pb))
        pot_after.append(max(0.0, pa))
        raise_to_present += int(action.get("raise_to") is not None)
        call_to_present += int(action.get("call_to") is not None)

    counts = Counter(action_types)
    meaningful = max(
        counts.get("call", 0)
        + counts.get("check", 0)
        + counts.get("bet", 0)
        + counts.get("raise", 0)
        + counts.get("fold", 0),
        1,
    )
    aggressive = counts.get("bet", 0) + counts.get("raise", 0)
    passive = counts.get("call", 0) + counts.get("check", 0)

    preflop_n = sum(1 for s in street_names if s == "preflop")
    postflop_n = sum(1 for s in street_names if s not in {"", "preflop"})
    nonzero_amount = sum(1 for v in amount_bb if v > 0.0)
    hero_actions = sum(1 for s in actor_seats if s == hero_seat and hero_seat > 0)
    button_actions = sum(1 for s in actor_seats if s == button_seat and button_seat > 0)

    pot_delta = [max(0.0, a - b) for a, b in zip(pot_after, pot_before)]
    monotonic = sum(
        1 for prev, cur in zip(pot_after, pot_after[1:]) if cur + 1e-9 >= prev
    )

    return {
        "schema_player_count": player_count,
        "schema_seat_utilization": _safe_div(player_count, max_seats),
        "schema_action_count": action_count,
        "schema_street_count": street_count,
        "schema_call_share": _safe_div(counts.get("call", 0), meaningful),
        "schema_check_share": _safe_div(counts.get("check", 0), meaningful),
        "schema_fold_share": _safe_div(counts.get("fold", 0), meaningful),
        "schema_bet_share": _safe_div(counts.get("bet", 0), meaningful),
        "schema_raise_share": _safe_div(counts.get("raise", 0), meaningful),
        "schema_blind_share": _safe_div(
            counts.get("small_blind", 0) + counts.get("big_blind", 0) + counts.get("ante", 0),
            max(1.0, action_count),
        ),
        "schema_allin_share": _safe_div(counts.get("all_in", 0), max(1.0, action_count)),
        "schema_aggression_share": _safe_div(aggressive, max(1.0, action_count)),
        "schema_passive_share": _safe_div(passive, max(1.0, action_count)),
        "schema_preflop_share": _safe_div(preflop_n, max(1.0, action_count)),
        "schema_postflop_share": _safe_div(postflop_n, max(1.0, action_count)),
        "schema_action_entropy": _entropy(action_types),
        "schema_actor_entropy": _entropy(actor_seats),
        "schema_street_entropy": _entropy(street_names),
        "schema_unique_actor_share": _safe_div(len(set(actor_seats)), max(1.0, player_count)),
        "schema_actor_switch_rate": _safe_div(
            sum(1 for prev, cur in zip(actor_seats, actor_seats[1:]) if prev != cur),
            max(len(actor_seats) - 1, 1),
        ),
        "schema_actor_run_max_share": _max_run_share(actor_seats),
        "schema_action_run_max_share": _max_run_share(action_types),
        "schema_amount_mean_bb": _mean(amount_bb),
        "schema_amount_std_bb": _std(amount_bb),
        "schema_amount_q90_bb": _quantile(amount_bb, 0.9),
        "schema_amount_max_bb": max(amount_bb) if amount_bb else 0.0,
        "schema_nonzero_amount_share": _safe_div(nonzero_amount, max(1.0, action_count)),
        "schema_pot_before_mean_bb": _mean(pot_before),
        "schema_pot_after_mean_bb": _mean(pot_after),
        "schema_pot_delta_mean_bb": _mean(pot_delta),
        "schema_pot_growth_bb": (
            max(pot_after) - min(pot_before) if pot_after and pot_before else 0.0
        ),
        "schema_pot_monotonic_rate": _safe_div(monotonic, max(len(pot_after) - 1, 1)),
        "schema_raise_to_share": _safe_div(raise_to_present, max(1.0, action_count)),
        "schema_call_to_share": _safe_div(call_to_present, max(1.0, action_count)),
        "schema_starting_stack_mean_bb": _mean(stack_bb),
        "schema_starting_stack_std_bb": _std(stack_bb),
        "schema_starting_stack_iqr_bb": _quantile(stack_bb, 0.75) - _quantile(stack_bb, 0.25),
        "schema_hero_action_share": _safe_div(hero_actions, max(1.0, action_count)),
        "schema_button_action_share": _safe_div(button_actions, max(1.0, action_count)),
        "schema_hero_button_same": float(hero_seat > 0 and hero_seat == button_seat),
    }


def _aggregate_feature(prefix: str, values: list[float], out: dict[str, float]) -> None:
    out[f"{prefix}_mean"] = _mean(values)
    out[f"{prefix}_std"] = _std(values)
    out[f"{prefix}_min"] = min(values) if values else 0.0
    out[f"{prefix}_max"] = max(values) if values else 0.0
    out[f"{prefix}_q10"] = _quantile(values, 0.1)
    out[f"{prefix}_q50"] = _quantile(values, 0.5)
    out[f"{prefix}_q90"] = _quantile(values, 0.9)




# ---------------------------------------------------------------------------
# Hand-level action n-gram tokens.
#
# Technique adapted from the MIT-licensed poker44-handngram-miner
# (github.com/Yaroslav98214/poker44-handngram-miner): tokenize each hand's
# action stream into street+action+size-bucket tokens, count unigrams/bigrams/
# trigrams plus per-position action counts, and aggregate counts per chunk.
#
# Design choices verified against real captured live traffic (2026-07-06):
#   * Counts are normalized PER HAND at emission (see chunk_features) so the
#     features stay comparable between ~34-hand benchmark batches and the
#     ~80-100-hand chunks validators send live (raw counts scale with chunk
#     size and would go out-of-distribution).
#   * The "pos<rel><act>" token uses (actor_seat - button_seat) % max_seats.
#     The validator payload view hardcodes button_seat=0, so in practice this
#     is the hand's action-order seat alias modulo table size -- a positional
#     signal, but NOT true button distance.
#   * _NGRAM_VOCAB is a FIXED vocabulary (tokens present in >=50 chunks of the
#     v1.12 benchmark as of 2026-07-06). A fixed vocabulary keeps the feature
#     schema identical across training runs and at inference; unseen tokens
#     are ignored, absent tokens emit 0.0.
#   * Names are sanitized ("|" -> "__", "?" -> "Q") so downstream tooling never
#     sees special characters.
# ---------------------------------------------------------------------------

_NGRAM_ACTION_CODES = {
    "fold": "F",
    "call": "C",
    "raise": "R",
    "check": "K",
    "bet": "B",
    "action": "A",
    "all_in": "I",
}

_NGRAM_VOCAB: tuple[str, ...] = (
    'fBm', 'fBm|fCm', 'fBm|fCm|tBp', 'fBm|fCm|tK0', 'fBm|fCs', 'fBm|fF0',
    'fBm|fF0|fF0', 'fBm|fRp', 'fBm|fRp|fF0', 'fBm|tBp', 'fBm|tK0', 'fBm|tK0|rK0',
    'fBo', 'fBo|fF0', 'fBp', 'fBp|fCm', 'fBp|fCm|tK0', 'fBp|fCp',
    'fBp|fF0', 'fBp|fF0|fF0', 'fBp|fRp', 'fBp|fRp|fF0', 'fBp|tBp', 'fBp|tK0',
    'fBs', 'fBs|fF0', 'fCm', 'fCm|tBm', 'fCm|tBp', 'fCm|tBp|tF0',
    'fCm|tK0', 'fCm|tK0|rBp', 'fCm|tK0|rK0', 'fCm|tK0|tBp', 'fCm|tK0|tK0', 'fCp',
    'fCp|tBp', 'fCp|tBp|tF0', 'fCp|tK0', 'fCs', 'fCs|tK0', 'fF0',
    'fF0|fF0', 'fF0|fF0|fF0', 'fF0|tBp', 'fF0|tK0', 'fK0', 'fK0|fBm',
    'fK0|fBm|fCm', 'fK0|fBm|fF0', 'fK0|fBm|fRp', 'fK0|fBp', 'fK0|fBp|fF0', 'fK0|fBs',
    'fK0|fBs|fF0', 'fK0|fCm', 'fK0|fCm|tK0', 'fK0|fF0', 'fK0|fF0|fF0', 'fK0|fK0',
    'fK0|fK0|tBm', 'fK0|fK0|tBp', 'fK0|fK0|tK0', 'fK0|fRp', 'fK0|rBp', 'fK0|rBp|rF0',
    'fK0|rK0', 'fK0|rK0|rK0', 'fK0|tBm', 'fK0|tBm|tF0', 'fK0|tBo', 'fK0|tBp',
    'fK0|tBp|rBp', 'fK0|tBp|rK0', 'fK0|tBp|tF0', 'fK0|tBs', 'fK0|tBs|tF0', 'fK0|tCm',
    'fK0|tF0', 'fK0|tF0|tF0', 'fK0|tK0', 'fK0|tK0|rBm', 'fK0|tK0|rBp', 'fK0|tK0|rK0',
    'fK0|tK0|tBm', 'fK0|tK0|tBp', 'fK0|tK0|tF0', 'fK0|tK0|tK0', 'fRp', 'fRp|fCp',
    'fRp|fF0', 'len', 'nseats', 'pCm', 'pCm|fBm', 'pCm|fBm|fCm',
    'pCm|fBm|fF0', 'pCm|fBp', 'pCm|fBp|fF0', 'pCm|fCm', 'pCm|fCm|tK0', 'pCm|fK0',
    'pCm|fK0|fBm', 'pCm|fK0|fBp', 'pCm|fK0|fCm', 'pCm|fK0|fF0', 'pCm|fK0|fK0', 'pCm|fK0|tBm',
    'pCm|fK0|tBp', 'pCm|fK0|tK0', 'pCm|pCm', 'pCm|pCs', 'pCm|pCs|fK0', 'pCm|pF0',
    'pCm|pF0|fBm', 'pCm|pF0|fBp', 'pCm|pF0|fK0', 'pCm|pF0|pCm', 'pCm|pF0|pCs', 'pCm|pF0|pF0',
    'pCm|pF0|pK0', 'pCm|pF0|pRo', 'pCm|pF0|pRp', 'pCm|pRo', 'pCm|pRo|pF0', 'pCm|pRp',
    'pCm|pRp|pF0', 'pCp', 'pCp|fBp', 'pCp|fK0', 'pCp|pF0', 'pCs',
    'pCs|fBm', 'pCs|fBm|fF0', 'pCs|fBp', 'pCs|fBp|fF0', 'pCs|fF0', 'pCs|fK0',
    'pCs|fK0|fBm', 'pCs|fK0|fBp', 'pCs|fK0|fF0', 'pCs|fK0|fK0', 'pCs|fK0|tK0', 'pCs|pCs',
    'pCs|pF0', 'pCs|pK0', 'pCs|pK0|fK0', 'pCs|pRo', 'pF0', 'pF0|fBm',
    'pF0|fBm|fCm', 'pF0|fBm|fF0', 'pF0|fBm|tK0', 'pF0|fBp', 'pF0|fBp|fF0', 'pF0|fCm',
    'pF0|fCm|tBp', 'pF0|fCm|tK0', 'pF0|fCp', 'pF0|fF0', 'pF0|fK0', 'pF0|fK0|fBm',
    'pF0|fK0|fBp', 'pF0|fK0|fF0', 'pF0|fK0|fK0', 'pF0|fK0|rK0', 'pF0|fK0|tBm', 'pF0|fK0|tBp',
    'pF0|fK0|tK0', 'pF0|fRp', 'pF0|pCm', 'pF0|pCm|fBm', 'pF0|pCm|fBp', 'pF0|pCm|fCm',
    'pF0|pCm|fK0', 'pF0|pCm|pCs', 'pF0|pCm|pF0', 'pF0|pCm|pRo', 'pF0|pCm|pRp', 'pF0|pCp',
    'pF0|pCs', 'pF0|pCs|fBm', 'pF0|pCs|fBp', 'pF0|pCs|fK0', 'pF0|pCs|pF0', 'pF0|pCs|pK0',
    'pF0|pCs|pRo', 'pF0|pF0', 'pF0|pF0|fBm', 'pF0|pF0|fBp', 'pF0|pF0|fCm', 'pF0|pF0|fK0',
    'pF0|pF0|pCm', 'pF0|pF0|pCs', 'pF0|pF0|pF0', 'pF0|pF0|pK0', 'pF0|pF0|pRo', 'pF0|pF0|pRp',
    'pF0|pK0', 'pF0|pK0|fK0', 'pF0|pRo', 'pF0|pRo|fBm', 'pF0|pRo|fK0', 'pF0|pRo|pCm',
    'pF0|pRo|pF0', 'pF0|pRo|pRo', 'pF0|pRp', 'pF0|pRp|fK0', 'pF0|pRp|pCm', 'pF0|pRp|pCp',
    'pF0|pRp|pF0', 'pF0|pRp|pRo', 'pK0', 'pK0|fBm', 'pK0|fBm|fF0', 'pK0|fBp',
    'pK0|fK0', 'pK0|fK0|fBp', 'pK0|fK0|fF0', 'pK0|fK0|fK0', 'pK0|fK0|tK0', 'pK0|pCs',
    'pK0|pCs|fK0', 'pK0|pRo', 'pK0|pRo|pCm', 'pK0|pRo|pF0', 'pRo', 'pRo|fBm',
    'pRo|fBp', 'pRo|fK0', 'pRo|pCm', 'pRo|pCm|fBm', 'pRo|pCm|fBp', 'pRo|pCm|fK0',
    'pRo|pCm|pCs', 'pRo|pCm|pF0', 'pRo|pCp', 'pRo|pCp|fK0', 'pRo|pCp|pF0', 'pRo|pCs',
    'pRo|pF0', 'pRo|pF0|fBm', 'pRo|pF0|fK0', 'pRo|pF0|pCm', 'pRo|pF0|pCs', 'pRo|pF0|pF0',
    'pRo|pF0|pRo', 'pRo|pF0|pRp', 'pRo|pRo', 'pRo|pRo|pF0', 'pRo|pRp', 'pRo|pRp|pF0',
    'pRp', 'pRp|fK0', 'pRp|pCm', 'pRp|pCm|fBm', 'pRp|pCm|fK0', 'pRp|pCm|pF0',
    'pRp|pCp', 'pRp|pCs', 'pRp|pF0', 'pRp|pF0|fK0', 'pRp|pF0|pCm', 'pRp|pF0|pCs',
    'pRp|pF0|pF0', 'pRp|pF0|pRo', 'pRp|pRo', 'pRp|pRo|pCm', 'pRp|pRo|pF0', 'pos0B',
    'pos0C', 'pos0F', 'pos0K', 'pos0R', 'pos1B', 'pos1C',
    'pos1F', 'pos1K', 'pos1R', 'pos2B', 'pos2C', 'pos2F',
    'pos2K', 'pos2R', 'pos3B', 'pos3C', 'pos3F', 'pos3K',
    'pos3R', 'pos4B', 'pos4C', 'pos4F', 'pos4K', 'pos4R',
    'pos5B', 'pos5C', 'pos5F', 'pos5K', 'pos5R', 'rBm',
    'rBm|rCm', 'rBm|rF0', 'rBp', 'rBp|rCp', 'rBp|rF0', 'rBs',
    'rBs|rF0', 'rCm', 'rCp', 'rCs', 'rF0', 'rK0',
    'rK0|rBp', 'rK0|rCp', 'rK0|rF0', 'rK0|rK0', 'rRp', 'rRp|rCp',
    'tBm', 'tBm|rBp', 'tBm|rK0', 'tBm|rK0|rK0', 'tBm|tCm', 'tBm|tF0',
    'tBo', 'tBp', 'tBp|rBp', 'tBp|rBp|rCp', 'tBp|rBp|rF0', 'tBp|rK0',
    'tBp|rK0|rK0', 'tBp|tCm', 'tBp|tCp', 'tBp|tCp|rBp', 'tBp|tF0', 'tBs',
    'tBs|tF0', 'tCm', 'tCm|rBp', 'tCm|rBp|rF0', 'tCm|rK0', 'tCm|rK0|rK0',
    'tCp', 'tCp|rBp', 'tCp|rBp|rF0', 'tCp|rK0', 'tCs', 'tCs|rK0',
    'tF0', 'tF0|tF0', 'tK0', 'tK0|rBm', 'tK0|rBm|rCm', 'tK0|rBm|rF0',
    'tK0|rBp', 'tK0|rBp|rCp', 'tK0|rBp|rF0', 'tK0|rBs', 'tK0|rK0', 'tK0|rK0|rF0',
    'tK0|rK0|rK0', 'tK0|tBm', 'tK0|tBm|tF0', 'tK0|tBp', 'tK0|tBp|tF0', 'tK0|tF0',
    'tK0|tK0', 'tK0|tK0|rBm', 'tK0|tK0|rBp', 'tK0|tK0|rK0', 'tRp', 'tRp|tCp',
    'tRp|tF0',
)


def _sanitize_ngram_token(token: str) -> str:
    return token.replace("|", "__").replace("?", "Q")


def _hand_ngram_doc(hand: dict[str, Any]) -> Counter:
    """Bag-of-ngrams document for a single sanitized hand."""
    actions = hand.get("actions") or []
    metadata = hand.get("metadata") or {}
    button_seat = metadata.get("button_seat")
    max_seats = metadata.get("max_seats") or 6

    tokens: list[str] = []
    grams: Counter = Counter()
    acting_seats: set = set()
    for action in actions:
        street = (action.get("street") or "x")[:1]
        act = _NGRAM_ACTION_CODES.get(action.get("action_type") or "x", "X")
        amount = _safe_float(action.get("amount"), 0.0)
        pot_before = _safe_float(action.get("pot_before"), 0.0)
        if amount <= 0:
            bucket = "0"
        elif pot_before <= 0:
            bucket = "?"
        else:
            ratio = amount / pot_before
            bucket = "s" if ratio < 0.4 else ("m" if ratio < 0.9 else ("p" if ratio < 1.5 else "o"))
        token = street + act + bucket
        tokens.append(token)
        grams[token] += 1
        try:
            rel = (int(action.get("actor_seat")) - int(button_seat)) % int(max_seats)
            grams["pos" + str(rel) + act] += 1
        except Exception:
            pass
        acting_seats.add(action.get("actor_seat"))
    for i in range(len(tokens) - 1):
        grams[tokens[i] + "|" + tokens[i + 1]] += 1
        if i + 2 < len(tokens):
            grams[tokens[i] + "|" + tokens[i + 1] + "|" + tokens[i + 2]] += 1
    grams["len"] = len(tokens)
    grams["nseats"] = len(acting_seats)
    return grams


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    if not chunk:
        return {"hand_count": 0.0}

    out: dict[str, float] = {"hand_count": float(len(chunk))}
    per_hand = [_hand_features(hand) for hand in chunk]
    feature_names = sorted(per_hand[0].keys())

    for name in feature_names:
        series = [float(features[name]) for features in per_hand]
        _aggregate_feature(name, series, out)

    action_signatures: list[tuple[str, ...]] = []
    actor_signatures: list[tuple[int, ...]] = []
    street_signatures: list[tuple[str, ...]] = []
    amount_bucket_signatures: list[tuple[str, ...]] = []

    high_aggressive = 0
    low_action_entropy = 0
    high_actor_entropy = 0
    long_action_hand = 0

    for hand, feats in zip(chunk, per_hand):
        actions = hand.get("actions") or []
        action_types = tuple(str((a or {}).get("action_type") or "").lower().strip() for a in actions)
        actor_seq = tuple(
            _safe_int((a or {}).get("actor_seat"), 0) for a in actions if _safe_int((a or {}).get("actor_seat"), 0) > 0
        )
        street_seq = tuple(str((a or {}).get("street") or "").lower().strip() for a in actions)
        amounts = [
            max(0.0, _safe_float((a or {}).get("normalized_amount_bb"), 0.0))
            for a in actions
        ]
        amount_buckets = tuple(_amount_bucket(value) for value in amounts)

        action_signatures.append(action_types)
        actor_signatures.append(actor_seq)
        street_signatures.append(street_seq)
        amount_bucket_signatures.append(amount_buckets)

        high_aggressive += int(feats["schema_aggression_share"] >= 0.35)
        low_action_entropy += int(feats["schema_action_entropy"] <= 0.35)
        high_actor_entropy += int(feats["schema_actor_entropy"] >= 0.75)
        long_action_hand += int(feats["schema_action_count"] >= 12.0)

    n = float(len(chunk))
    out["schema_action_signature_top_share"] = _safe_div(max(Counter(action_signatures).values()), n)
    out["schema_action_signature_unique_share"] = _safe_div(len(set(action_signatures)), n)
    out["schema_actor_signature_top_share"] = _safe_div(max(Counter(actor_signatures).values()), n)
    out["schema_actor_signature_unique_share"] = _safe_div(len(set(actor_signatures)), n)
    out["schema_street_signature_top_share"] = _safe_div(max(Counter(street_signatures).values()), n)
    out["schema_street_signature_unique_share"] = _safe_div(len(set(street_signatures)), n)
    out["schema_amount_bucket_signature_top_share"] = _safe_div(
        max(Counter(amount_bucket_signatures).values()), n
    )
    out["schema_amount_bucket_signature_unique_share"] = _safe_div(
        len(set(amount_bucket_signatures)), n
    )
    out["schema_high_aggression_hand_rate"] = _safe_div(high_aggressive, n)
    out["schema_low_action_entropy_hand_rate"] = _safe_div(low_action_entropy, n)
    out["schema_high_actor_entropy_hand_rate"] = _safe_div(high_actor_entropy, n)
    out["schema_long_action_hand_rate"] = _safe_div(long_action_hand, n)

    # ---- hand-level n-gram tokens (fixed vocabulary, per-hand-normalized) ----
    ngram_totals: Counter = Counter()
    for hand in chunk:
        ngram_totals.update(_hand_ngram_doc(hand))
    for token in _NGRAM_VOCAB:
        out["schema_ngram_" + _sanitize_ngram_token(token)] = _safe_div(
            float(ngram_totals.get(token, 0.0)), n
        )

    # ---- cross-hand coherence block (ported from pd-coast V4) ----
    out.update(coherent_feature_dict(chunk))
    return out
