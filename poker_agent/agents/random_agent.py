"""Random agent — selects uniformly from legal actions."""

import random

from poker_agent.agents.base import Agent
from poker_agent.game import GameState, PokerGame


class RandomAgent(Agent):
    """Chooses a uniformly random legal action each turn."""

    def act(self, game_state: GameState, player_id: int) -> tuple[str, int]:
        """Pick a random legal action; raise amount is random in [min_raise, 3×pot]."""
        # Build a temporary PokerGame-like helper to get legal actions
        # (legal_actions is a pure function of state, so we instantiate a minimal helper)
        legal = _legal_actions(game_state, player_id)
        action = random.choice(legal)

        amount = 0
        if action == "raise":
            stack = game_state.stacks[player_id]
            min_r = game_state.min_raise
            max_r = max(min_r, min(3 * max(game_state.pot, 1), stack - game_state.current_bet))
            amount = random.randint(min_r, max_r)

        return action, amount


def _legal_actions(state: GameState, player_id: int) -> list[str]:
    """Derive legal actions directly from state without a full PokerGame instance."""
    p = player_id
    actions: list[str] = []
    stack = state.stacks[p]

    if state.current_bet > 0:
        actions.append("fold")
        actions.append("call")  # may be all-in but still legal
    else:
        actions.append("check")

    effective = stack - state.current_bet
    if effective >= state.min_raise:
        actions.append("raise")

    return actions
