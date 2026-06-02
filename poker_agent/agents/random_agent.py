"""Random baseline agent — uniform random legal actions."""

from __future__ import annotations

import random

from poker_agent.agents.base import Agent
from poker_agent.agents.utils import _legal_actions
from poker_agent.game import GameState


class RandomAgent(Agent):
    """Picks uniformly at random from the legal action set.

    When raising, the amount is a random integer between min_raise and
    min(3 × pot, stack).
    """

    def act(self, game_state: GameState, player_id: int) -> tuple[str, int]:
        """Return a random legal (action, amount) pair."""
        state = game_state
        p = player_id
        legal = _legal_actions(state, p)
        action = random.choice(legal)

        if action == "raise":
            min_raise = state.min_raise
            max_raise = min(3 * state.pot, state.stacks[p])
            amount = random.randint(min_raise, max(max_raise, min_raise))
            return "raise", amount

        return action, 0
