"""Monte Carlo Effective Hand Strength (EHS) estimator."""

from __future__ import annotations

import random

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
    """
    dead_values = {(c.rank, c.suit) for c in dead_cards}
    remaining = [c for c in _FULL_DECK if (c.rank, c.suit) not in dead_values]

    n_community_needed = 5 - len(community_cards)
    total_cards_needed = 2 + n_community_needed

    if len(remaining) < total_cards_needed:
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
    """Monte Carlo EHS — treys lookup table when available, pure Python fallback."""
    total = 0.0

    if _TREYS:
        hero_t = [_to_treys(c) for c in hole_cards]
        comm_t = [_to_treys(c) for c in community_cards]
        pool_t = [_to_treys(c) for c in remaining]

        for _ in range(n_samples):
            random.shuffle(pool_t)
            opp_t = pool_t[:2]
            board_t = comm_t + pool_t[2: 2 + n_community_needed]
            # treys: lower score = stronger hand
            h = _treys_eval.evaluate(board_t, hero_t)
            o = _treys_eval.evaluate(board_t, opp_t)
            if h < o:
                total += 1.0
            elif h == o:
                total += 0.5
    else:
        pool = list(remaining)
        for _ in range(n_samples):
            random.shuffle(pool)
            opp_hole = pool[:2]
            full_community = community_cards + pool[2: 2 + n_community_needed]
            hero_score = evaluate_hand(hole_cards + full_community)[0]
            opp_score  = evaluate_hand(opp_hole  + full_community)[0]
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
    """NumPy-accelerated EHS — batch index sampling + treys evaluation loop."""
    import numpy as np

    n_remaining = len(remaining)
    cards_needed = 2 + n_community_needed

    indices = np.stack([
        np.random.choice(n_remaining, size=cards_needed, replace=False)
        for _ in range(n_samples)
    ])

    total = 0.0

    if _TREYS:
        hero_t = [_to_treys(c) for c in hole_cards]
        comm_t = [_to_treys(c) for c in community_cards]
        pool_t = [_to_treys(c) for c in remaining]

        for row in indices:
            opp_t   = [pool_t[i] for i in row[:2]]
            board_t = comm_t + [pool_t[i] for i in row[2:]]
            h = _treys_eval.evaluate(board_t, hero_t)
            o = _treys_eval.evaluate(board_t, opp_t)
            if h < o:
                total += 1.0
            elif h == o:
                total += 0.5
    else:
        for row in indices:
            sampled = [remaining[i] for i in row]
            full_community = community_cards + sampled[2:]
            hero_score = evaluate_hand(hole_cards + full_community)[0]
            opp_score  = evaluate_hand(sampled[:2] + full_community)[0]
            if hero_score > opp_score:
                total += 1.0
            elif hero_score == opp_score:
                total += 0.5

    return total / n_samples
