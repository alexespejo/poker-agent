"""Monte Carlo Effective Hand Strength (EHS) estimator.

In addition to the uniform-sampling EHS estimator, this module provides a
*range-conditioned* mode: callers may pass a ``range_filter`` that maps an
opponent hole-card percentile (0 = worst starting hand, 1 = best) to a soft
weight in [0, 1].  The sampler then computes a weighted win rate so equity is
measured against the opponent's *plausible* holdings rather than a uniform
random hand.  See ``make_range_filter`` and ``preflop_percentile``.
"""

from __future__ import annotations

import math
import random
from typing import Callable

from poker_agent.card import Card, Rank, Suit

try:
    from treys import Card as TreysCard, Evaluator as TreysEvaluator
    _TREYS = True
    _treys_eval = TreysEvaluator()
    _RANK_STR = {
        2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
        8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A',
    }

    def _to_treys(card: Card) -> int:
        return TreysCard.new(_RANK_STR[card.rank.value] + str(card.suit))

except ImportError:
    _TREYS = False
    from poker_agent.hand_eval import evaluate_hand  # type: ignore[assignment]

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

# Full deck of 52 cards, built once at module load
_FULL_DECK: list[Card] = [Card(rank, suit) for suit in Suit for rank in Rank]


# ---------------------------------------------------------------------------
# Preflop hand-strength percentile model
# ---------------------------------------------------------------------------
#
# To condition equity on an opponent's range we need a cheap, stable notion of
# how strong a starting hand is relative to the full 1326-combo universe.  We
# precompute (once, lazily, deterministically) the heads-up equity-vs-random of
# each of the 169 canonical starting-hand types, then convert that ordering
# into a percentile in [0, 1] weighted by the number of combos each type spans
# (pair = 6, suited = 4, offsuit = 12).  AA → ~1.0, 72o → ~0.0.

# Combo key -> percentile.  Key is (high_rank, low_rank, suited_bool).
_PREFLOP_PERCENTILE: dict[tuple[int, int, bool], float] | None = None


def _combo_count(hi: int, lo: int, suited: bool) -> int:
    if hi == lo:
        return 6          # pocket pair
    return 4 if suited else 12


def _canonical_key(c1: Card, c2: Card) -> tuple[int, int, bool]:
    """Map two cards to (high_rank, low_rank, suited)."""
    r1, r2 = c1.rank.value, c2.rank.value
    hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
    suited = (hi != lo) and (c1.suit == c2.suit)
    return hi, lo, suited


def _equity_vs_random(hole_t, board_pool_t, n_samples: int, rng: random.Random) -> float:
    """Heads-up preflop equity of a treys hole hand vs a random hand (no board)."""
    total = 0.0
    pool = list(board_pool_t)
    for _ in range(n_samples):
        rng.shuffle(pool)
        opp_t = pool[:2]
        board_t = pool[2:7]
        h = _treys_eval.evaluate(board_t, hole_t)
        o = _treys_eval.evaluate(board_t, opp_t)
        if h < o:
            total += 1.0
        elif h == o:
            total += 0.5
    return total / n_samples


def _build_preflop_percentile_table() -> dict[tuple[int, int, bool], float]:
    """Compute the percentile table once, deterministically.

    Uses a fixed-seed local RNG so the ordering is reproducible and never
    touches the global ``random`` state.  Falls back to a Chen-style heuristic
    ordering when treys is unavailable.
    """
    ranks = list(range(2, 15))
    canon: list[tuple[int, int, bool]] = []
    for i, hi in enumerate(reversed(ranks)):
        for lo in list(reversed(ranks))[i:]:
            if hi == lo:
                canon.append((hi, lo, False))
            else:
                canon.append((hi, lo, True))
                canon.append((hi, lo, False))

    equities: dict[tuple[int, int, bool], float] = {}

    if _TREYS:
        rng = random.Random(0)
        for hi, lo, suited in canon:
            # Realise concrete cards for this canonical type.
            if hi == lo:
                c1 = Card(Rank(hi), Suit.SPADES)
                c2 = Card(Rank(lo), Suit.HEARTS)
            elif suited:
                c1 = Card(Rank(hi), Suit.SPADES)
                c2 = Card(Rank(lo), Suit.SPADES)
            else:
                c1 = Card(Rank(hi), Suit.SPADES)
                c2 = Card(Rank(lo), Suit.HEARTS)
            hole_t = [_to_treys(c1), _to_treys(c2)]
            dead = {(c1.rank, c1.suit), (c2.rank, c2.suit)}
            pool_t = [_to_treys(c) for c in _FULL_DECK if (c.rank, c.suit) not in dead]
            equities[(hi, lo, suited)] = _equity_vs_random(hole_t, pool_t, 200, rng)
    else:
        # Heuristic fallback (Chen-like): high cards, pairs, suitedness, connectedness.
        for hi, lo, suited in canon:
            score = (hi + lo) / 28.0
            if hi == lo:
                score += 0.35
            if suited:
                score += 0.06
            gap = hi - lo
            if hi != lo and gap <= 2:
                score += 0.04
            equities[(hi, lo, suited)] = score

    # Convert ordering into combo-weighted percentiles in [0, 1].
    ordered = sorted(equities.items(), key=lambda kv: kv[1])
    total_combos = sum(_combo_count(*k) for k, _ in ordered)
    table: dict[tuple[int, int, bool], float] = {}
    cumulative = 0
    for key, _eq in ordered:
        n = _combo_count(*key)
        # Midpoint percentile of this type's combos within the universe.
        table[key] = (cumulative + n / 2.0) / total_combos
        cumulative += n
    return table


def preflop_percentile(c1: Card, c2: Card) -> float:
    """Return the combo-weighted percentile of a two-card starting hand in [0, 1]."""
    global _PREFLOP_PERCENTILE
    if _PREFLOP_PERCENTILE is None:
        _PREFLOP_PERCENTILE = _build_preflop_percentile_table()
    return _PREFLOP_PERCENTILE[_canonical_key(c1, c2)]


def make_range_filter(
    top_fraction: float,
    k: float = 10.0,
) -> Callable[[float], float]:
    """Return a soft range filter: percentile -> weight in [0, 1].  [Bug 4]

    Instead of a hard 0/1 step at the cutoff, we use a logistic ramp so hands
    near the boundary receive partial weight, making conditioned equity smooth
    and robust:

        weight = 1 / (1 + exp(-k * (percentile - cutoff)))

    ``top_fraction`` is the fraction of strongest hands the opponent is assumed
    to hold, so the cutoff sits at ``1 - top_fraction``.  ``k`` (≈8–12) controls
    the ramp sharpness.
    """
    top_fraction = max(0.0, min(1.0, top_fraction))
    cutoff = 1.0 - top_fraction

    def _weight(percentile: float) -> float:
        return 1.0 / (1.0 + math.exp(-k * (percentile - cutoff)))

    return _weight


def make_range_filter_band(
    low_cutoff: float,
    high_cutoff: float,
    k: float = 10.0,
) -> Callable[[float], float]:
    """Soft band filter: weight rises through ``low_cutoff`` and falls through
    ``high_cutoff`` (both logistic).  Used for cumulative calling ranges.  [Bug 4]
    """
    lo = max(0.0, min(1.0, low_cutoff))
    hi = max(0.0, min(1.0, high_cutoff))

    def _weight(percentile: float) -> float:
        rising = 1.0 / (1.0 + math.exp(-k * (percentile - lo)))
        falling = 1.0 / (1.0 + math.exp(k * (percentile - hi)))
        return rising * falling

    return _weight


def estimate_ehs(
    hole_cards: list[Card],
    community_cards: list[Card],
    dead_cards: list[Card],
    n_samples: int = 1000,
    range_filter: Callable[[float], float] | None = None,
) -> float:
    """Estimate Effective Hand Strength via Monte Carlo simulation.

    For each sample:
      1. Draw 2 random opponent hole cards from the remaining deck.
      2. Complete community cards to 5 total (randomly).
      3. Evaluate both 7-card hands and record win=1, tie=0.5, loss=0.

    If ``range_filter`` is provided it maps each sampled opponent hand's preflop
    percentile to a soft weight in [0, 1]; the returned value is the
    weight-normalized win rate, i.e. equity conditioned on the opponent's range.
    When the accumulated weight is too small to be reliable we fall back to the
    uniform (unweighted) estimate.

    Returns the mean win rate across all samples (float in [0, 1]).
    """
    dead_values = {(c.rank, c.suit) for c in dead_cards}
    remaining = [c for c in _FULL_DECK if (c.rank, c.suit) not in dead_values]

    n_community_needed = 5 - len(community_cards)
    total_cards_needed = 2 + n_community_needed

    if len(remaining) < total_cards_needed:
        return 0.5

    if _NUMPY:
        return _estimate_ehs_numpy(
            hole_cards, community_cards, remaining, n_community_needed,
            n_samples, range_filter,
        )
    return _estimate_ehs_pure(
        hole_cards, community_cards, remaining, n_community_needed,
        n_samples, range_filter,
    )


# Minimum accumulated weight (relative to n_samples) before we trust the
# range-conditioned estimate; below this we use the uniform estimate.
_MIN_WEIGHT_FRACTION = 0.05


def _estimate_ehs_pure(
    hole_cards: list[Card],
    community_cards: list[Card],
    remaining: list[Card],
    n_community_needed: int,
    n_samples: int,
    range_filter: Callable[[float], float] | None = None,
) -> float:
    """Monte Carlo EHS — treys lookup table when available, pure Python fallback."""
    total = 0.0          # unweighted win mass (fallback)
    w_total = 0.0        # accumulated weight
    w_win = 0.0          # weighted win mass

    if _TREYS:
        hero_t = [_to_treys(c) for c in hole_cards]
        comm_t = [_to_treys(c) for c in community_cards]
        pool_t = [_to_treys(c) for c in remaining]
        # Shuffle an index permutation so Card objects (for percentile lookup)
        # stay aligned with their treys ints.
        idx = list(range(len(remaining)))

        for _ in range(n_samples):
            random.shuffle(idx)
            i0, i1 = idx[0], idx[1]
            opp_t = [pool_t[i0], pool_t[i1]]
            board_t = comm_t + [pool_t[j] for j in idx[2: 2 + n_community_needed]]
            # treys: lower score = stronger hand
            h = _treys_eval.evaluate(board_t, hero_t)
            o = _treys_eval.evaluate(board_t, opp_t)
            result = 1.0 if h < o else (0.5 if h == o else 0.0)
            total += result
            if range_filter is not None:
                w = range_filter(preflop_percentile(remaining[i0], remaining[i1]))
                w_total += w
                w_win += w * result
    else:
        pool = list(remaining)
        for _ in range(n_samples):
            random.shuffle(pool)
            opp_hole = pool[:2]
            full_community = community_cards + pool[2: 2 + n_community_needed]
            hero_score = evaluate_hand(hole_cards + full_community)[0]
            opp_score  = evaluate_hand(opp_hole  + full_community)[0]
            result = 1.0 if hero_score > opp_score else (0.5 if hero_score == opp_score else 0.0)
            total += result
            if range_filter is not None:
                w = range_filter(preflop_percentile(opp_hole[0], opp_hole[1]))
                w_total += w
                w_win += w * result

    if range_filter is not None and w_total >= _MIN_WEIGHT_FRACTION * n_samples:
        return w_win / w_total
    return total / n_samples


def _estimate_ehs_numpy(
    hole_cards: list[Card],
    community_cards: list[Card],
    remaining: list[Card],
    n_community_needed: int,
    n_samples: int,
    range_filter: Callable[[float], float] | None = None,
) -> float:
    """NumPy-accelerated EHS — batch index sampling + treys evaluation loop."""
    import numpy as np

    n_remaining = len(remaining)
    cards_needed = 2 + n_community_needed

    indices = np.stack([
        np.random.choice(n_remaining, size=cards_needed, replace=False)
        for _ in range(n_samples)
    ])

    total = 0.0
    w_total = 0.0
    w_win = 0.0

    if _TREYS:
        hero_t = [_to_treys(c) for c in hole_cards]
        comm_t = [_to_treys(c) for c in community_cards]
        pool_t = [_to_treys(c) for c in remaining]

        for row in indices:
            i0, i1 = int(row[0]), int(row[1])
            opp_t   = [pool_t[i0], pool_t[i1]]
            board_t = comm_t + [pool_t[i] for i in row[2:]]
            h = _treys_eval.evaluate(board_t, hero_t)
            o = _treys_eval.evaluate(board_t, opp_t)
            result = 1.0 if h < o else (0.5 if h == o else 0.0)
            total += result
            if range_filter is not None:
                w = range_filter(preflop_percentile(remaining[i0], remaining[i1]))
                w_total += w
                w_win += w * result
    else:
        for row in indices:
            i0, i1 = int(row[0]), int(row[1])
            sampled = [remaining[i] for i in row]
            full_community = community_cards + sampled[2:]
            hero_score = evaluate_hand(hole_cards + full_community)[0]
            opp_score  = evaluate_hand(sampled[:2] + full_community)[0]
            result = 1.0 if hero_score > opp_score else (0.5 if hero_score == opp_score else 0.0)
            total += result
            if range_filter is not None:
                w = range_filter(preflop_percentile(remaining[i0], remaining[i1]))
                w_total += w
                w_win += w * result

    if range_filter is not None and w_total >= _MIN_WEIGHT_FRACTION * n_samples:
        return w_win / w_total
    return total / n_samples
