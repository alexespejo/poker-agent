"""EHS agent — Monte Carlo hand strength + pot-odds decision policy."""

from __future__ import annotations

from poker_agent.agents.base import Agent
from poker_agent.agents.random_agent import _legal_actions
from poker_agent.game import GameState
from poker_agent.monte_carlo import estimate_ehs


class EHSAgent(Agent):
    """Makes decisions purely from Monte Carlo EHS and pot-odds.

    Decision logic:
      - No bet (call_amount == 0):
          check  if ehs < 0.55
          raise  if ehs >= 0.55
      - Facing a bet:
          raise  if ehs > pot_odds + raise_threshold
          call   if ehs > pot_odds
          fold   otherwise

    Raise sizing (Kelly-inspired):
      raise_amount = pot * (ehs - 0.5) * 2,
      clipped to [min_raise, 0.75 * stack].
    """

    def __init__(
        self,
        n_samples: int = 500,
        raise_threshold: float = 0.15,
        verbose: bool = False,
    ) -> None:
        self.n_samples = n_samples
        self.raise_threshold = raise_threshold
        self.verbose = verbose
        self._ehs_cache: dict[tuple, float] = {}

    def act(self, game_state: GameState, player_id: int) -> tuple[str, int]:
        """Return (action, amount) based on EHS vs pot odds."""
        state = game_state
        p = player_id
        legal = _legal_actions(state, p)

        hole = state.hole_cards[p]
        community = state.community_cards
        cache_key = (tuple(hole), tuple(community))
        if cache_key in self._ehs_cache:
            ehs = self._ehs_cache[cache_key]
        else:
            dead = hole + community
            ehs = estimate_ehs(hole, community, dead, self.n_samples)
            self._ehs_cache[cache_key] = ehs

        call_amount = state.current_bet
        pot = state.pot

        pot_odds = call_amount / (pot + call_amount) if call_amount > 0 else 0.0

        stack = state.stacks[p]
        min_raise = state.min_raise

        action, amount = self._decide(ehs, pot_odds, call_amount, pot, stack, min_raise, legal)

        if self.verbose:
            print(f"  [EHS] P{p} {state.street}: ehs={ehs:.3f} pot_odds={pot_odds:.3f} "
                  f"call={call_amount} pot={pot} → {action}"
                  + (f" {amount}" if action == "raise" else ""))

        return action, amount

    def _decide(
        self,
        ehs: float,
        pot_odds: float,
        call_amount: int,
        pot: int,
        stack: int,
        min_raise: int,
        legal: list[str],
    ) -> tuple[str, int]:
        """Core pot-odds decision logic."""
        if call_amount == 0:
            # No bet to face — check or raise
            if ehs >= 0.55 and "raise" in legal:
                amount = self._raise_size(ehs, pot, stack, min_raise)
                return "raise", amount
            return "check", 0

        # Facing a bet
        if ehs > pot_odds + self.raise_threshold and "raise" in legal:
            amount = self._raise_size(ehs, pot, stack, min_raise)
            return "raise", amount

        if ehs > pot_odds:
            if "call" in legal:
                return "call", 0
            return "check", 0

        # EHS doesn't justify calling
        if "fold" in legal:
            return "fold", 0
        return "check", 0

    def _raise_size(self, ehs: float, pot: int, stack: int, min_raise: int) -> int:
        """Kelly-inspired raise sizing."""
        kelly = int(pot * (ehs - 0.5) * 2)
        max_raise = int(0.75 * stack)
        return max(min_raise, min(kelly, max_raise))
