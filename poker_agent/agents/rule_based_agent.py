"""Rule-based agent using preflop hand strength heuristics."""

from poker_agent.agents.base import Agent
from poker_agent.agents.random_agent import _legal_actions
from poker_agent.game import GameState


# ---------------------------------------------------------------------------
# Preflop hand strength table (Chen formula approximation, normalized 0–1)
# Keys: (high_rank, low_rank, suited) where high_rank >= low_rank (2–14)
# Values: raw Chen score (will be normalized)
# ---------------------------------------------------------------------------

def _chen_score(r1: int, r2: int, suited: bool) -> float:
    """Approximate Chen formula for a starting hand."""
    # Base score = value of highest card
    def card_val(r: int) -> float:
        if r == 14: return 10.0
        if r == 13: return 8.0
        if r == 12: return 7.0
        if r == 11: return 6.0
        return r / 2.0

    score = card_val(max(r1, r2))

    # Pair bonus
    if r1 == r2:
        score = max(score * 2, 5.0)
    else:
        # Suited bonus
        if suited:
            score += 2.0
        # Gap penalties
        gap = max(r1, r2) - min(r1, r2) - 1
        if gap == 1:
            score -= 1.0
        elif gap == 2:
            score -= 2.0
        elif gap == 3:
            score -= 4.0
        elif gap >= 4:
            score -= 5.0
        # Straight possibility bonus (connected low cards)
        if gap <= 2 and min(r1, r2) < 12:
            score += 1.0

    return score


def _build_preflop_table() -> dict[tuple[int, int, bool], float]:
    """Build and normalize a preflop strength table for all 169 hand types."""
    raw: dict[tuple[int, int, bool], float] = {}
    for r1 in range(2, 15):
        for r2 in range(2, r1 + 1):
            for suited in (True, False):
                if r1 == r2 and suited:
                    continue  # can't have a suited pair
                key = (r1, r2, suited)
                raw[key] = _chen_score(r1, r2, suited)

    min_score = min(raw.values())
    max_score = max(raw.values())
    span = max_score - min_score
    normalized = {k: (v - min_score) / span for k, v in raw.items()}
    return normalized


_PREFLOP_TABLE = _build_preflop_table()


def preflop_strength(hole_cards: list) -> float:  # hole_cards: list[Card]
    """Return normalized preflop hand strength in [0, 1]."""
    r1 = hole_cards[0].rank.value
    r2 = hole_cards[1].rank.value
    suited = hole_cards[0].suit == hole_cards[1].suit
    high, low = max(r1, r2), min(r1, r2)
    return _PREFLOP_TABLE.get((high, low, suited), 0.5)


class RuleBasedAgent(Agent):
    """Simple rule-based agent.

    Preflop:
      top 20% strength → raise
      top 50% strength → call
      otherwise → fold

    Postflop:
      call if call_amount <= pot // 3, else fold; check when no bet.
    """

    def act(self, game_state: GameState, player_id: int) -> tuple[str, int]:
        """Return (action, amount) based on hand strength heuristics."""
        legal = _legal_actions(game_state, player_id)
        state = game_state
        p = player_id

        if state.street == "preflop":
            return self._preflop_action(state, p, legal)
        return self._postflop_action(state, p, legal)

    def _preflop_action(
        self, state: GameState, p: int, legal: list[str]
    ) -> tuple[str, int]:
        """Preflop decision based on normalized hand strength."""
        strength = preflop_strength(state.hole_cards[p])

        if strength >= 0.80:  # top ~20%
            if "raise" in legal:
                amount = int(2.5 * state.big_blind)
                return "raise", max(state.min_raise, amount)
            if "call" in legal:
                return "call", 0
            return "check", 0

        if strength >= 0.50:  # top ~50%
            if "call" in legal:
                return "call", 0
            return "check", 0

        # Bottom 50%: fold if there's a bet, else check
        if "fold" in legal and state.current_bet > 0:
            return "fold", 0
        return "check", 0

    def _postflop_action(
        self, state: GameState, p: int, legal: list[str]
    ) -> tuple[str, int]:
        """Postflop: call small bets, fold to large ones, check for free."""
        if state.current_bet == 0:
            return "check", 0
        if state.current_bet <= state.pot // 3:
            return "call", 0
        if "fold" in legal:
            return "fold", 0
        return "call", 0
