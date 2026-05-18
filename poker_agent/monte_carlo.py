"""Monte Carlo Effective Hand Strength (EHS) estimator."""

from __future__ import annotations

import random

from poker_agent.card import Card, Rank, Suit
from poker_agent.hand_eval import evaluate_hand

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

# Full deck of 52 cards, built once at module load
_FULL_DECK: list[Card] = [Card(rank, suit) for suit in Suit for rank in Rank]


def estimate_ehs(
    hole_cards: list[Card],
    community_cards: list[Card],
    dead_cards: list[Card],
    n_samples: int = 1000,
) -> float:
    """Estimate Effective Hand Strength via Monte Carlo simulation.

    For each sample:
      1. Draw 2 random opponent hole cards from the remaining deck.
      2. Complete community cards to 5 total (randomly).
      3. Evaluate both 7-card hands and record win=1, tie=0.5, loss=0.

    Returns the mean win rate across all samples (float in [0, 1]).

    Args:
        hole_cards:      The hero's 2 hole cards.
        community_cards: Community cards dealt so far (0–5).
        dead_cards:      All known cards to exclude from sampling (should include
                         hole_cards + community_cards at minimum).
        n_samples:       Number of Monte Carlo rollouts.
    """
    dead_set = set(id(c) for c in dead_cards)
    # Build remaining deck by value equality (not identity)
    dead_values = {(c.rank, c.suit) for c in dead_cards}
    remaining = [c for c in _FULL_DECK if (c.rank, c.suit) not in dead_values]

    n_community_needed = 5 - len(community_cards)
    total_cards_needed = 2 + n_community_needed  # opponent hole + board runout

    if len(remaining) < total_cards_needed:
        # Degenerate case — not enough cards to sample
        return 0.5

    if _NUMPY:
        return _estimate_ehs_numpy(
            hole_cards, community_cards, remaining, n_community_needed, n_samples
        )
    return _estimate_ehs_pure(
        hole_cards, community_cards, remaining, n_community_needed, n_samples
    )


def _estimate_ehs_pure(
    hole_cards: list[Card],
    community_cards: list[Card],
    remaining: list[Card],
    n_community_needed: int,
    n_samples: int,
) -> float:
    """Pure-Python Monte Carlo EHS estimation."""
    total = 0.0
    pool = list(remaining)

    for _ in range(n_samples):
        random.shuffle(pool)
        opp_hole = pool[:2]
        runout = pool[2 : 2 + n_community_needed]
        full_community = community_cards + runout

        hero_score = evaluate_hand(hole_cards + full_community)[0]
        opp_score = evaluate_hand(opp_hole + full_community)[0]

        if hero_score > opp_score:
            total += 1.0
        elif hero_score == opp_score:
            total += 0.5

    return total / n_samples


def _estimate_ehs_numpy(
    hole_cards: list[Card],
    community_cards: list[Card],
    remaining: list[Card],
    n_community_needed: int,
    n_samples: int,
) -> float:
    """NumPy-accelerated Monte Carlo EHS (samples drawn in batch, evaluated in Python loop)."""
    import numpy as np

    n_remaining = len(remaining)
    cards_needed = 2 + n_community_needed

    # Draw all samples at once using numpy choice without replacement per row
    indices = np.stack([
        np.random.choice(n_remaining, size=cards_needed, replace=False)
        for _ in range(n_samples)
    ])

    total = 0.0
    for row in indices:
        sampled = [remaining[i] for i in row]
        opp_hole = sampled[:2]
        runout = sampled[2:]
        full_community = community_cards + runout

        hero_score = evaluate_hand(hole_cards + full_community)[0]
        opp_score = evaluate_hand(opp_hole + full_community)[0]

        if hero_score > opp_score:
            total += 1.0
        elif hero_score == opp_score:
            total += 0.5

    return total / n_samples
