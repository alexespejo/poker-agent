"""Tight-aggressive (TAG) benchmarking agent.

Plays a narrow range of starting hands (VPIP ~0.25-0.30) but bets and raises
aggressively when it does enter (AF ~3.0-5.0). Folds strong hands rarely
(FTR ~0.20-0.30). Designed to exercise FullAgent's opponent model along the
tight-aggressive axis — the model should detect low VPIP + high AF and tighten
its calling standard accordingly.
"""

from __future__ import annotations

from poker_agent.agents.base import Agent
from poker_agent.agents.utils import _legal_actions
from poker_agent.game import GameState
from poker_agent.monte_carlo import estimate_ehs


class TightAggressiveAgent(Agent):
    """Tight preflop ranges, aggressive postflop betting.

    Preflop (EHS from hole cards only):
      ehs >= 0.62              → raise (Kelly sizing, min 2.5x big blind)
      0.55 <= ehs < 0.62       → call if facing a bet <= 1 BB, else fold
      ehs < 0.55               → fold to any bet, check if free

    Postflop (EHS from hole + community):
      ehs > pot_odds + 0.08 and raise legal → raise (Kelly sizing)
      ehs > pot_odds                        → call
      otherwise                             → fold (or check if free)
    """

    raise_threshold: float = 0.08

    def __init__(self, n_samples: int = 200, verbose: bool = False) -> None:
        self.n_samples = n_samples
        self.verbose = verbose

    def reset(self) -> None:
        """No persistent state."""
        pass

    def act(self, game_state: GameState, player_id: int) -> tuple[str, int]:
        """Return (action, amount) for the tight-aggressive policy."""
        state = game_state
        p = player_id
        legal = _legal_actions(state, p)

        hole = state.hole_cards[p]
        community = state.community_cards

        # Cache EHS per (hole, community) within this act() call only.
        ehs_cache: dict[tuple, float] = {}
        cache_key = (tuple(hole), tuple(community))
        if cache_key in ehs_cache:
            ehs = ehs_cache[cache_key]
        else:
            dead = hole + community
            ehs = estimate_ehs(hole, community, dead, self.n_samples)
            ehs_cache[cache_key] = ehs

        call_amount = state.current_bet
        pot = state.pot
        stack = state.stacks[p]
        min_raise = state.min_raise
        big_blind = state.big_blind

        if state.street == "preflop":
            action, amount = self._preflop(
                ehs, call_amount, pot, stack, min_raise, big_blind, legal
            )
        else:
            pot_odds = call_amount / (pot + call_amount) if call_amount > 0 else 0.0
            action, amount = self._postflop(
                ehs, pot_odds, call_amount, pot, stack, min_raise, legal
            )

        if self.verbose:
            print(
                f"  [TAG] P{p} {state.street}: ehs={ehs:.3f} call={call_amount} "
                f"pot={pot} → {action}" + (f" {amount}" if action == "raise" else "")
            )

        return action, amount

    def _preflop(
        self,
        ehs: float,
        call_amount: int,
        pot: int,
        stack: int,
        min_raise: int,
        big_blind: int,
        legal: list[str],
    ) -> tuple[str, int]:
        """Tight preflop range with aggressive raising."""
        if ehs >= 0.62 and "raise" in legal:
            amount = self._raise_size(ehs, pot, stack, min_raise)
            amount = max(amount, int(2.5 * big_blind))
            amount = min(amount, int(0.75 * stack))
            amount = max(min_raise, amount)
            return "raise", amount

        if 0.55 <= ehs < 0.62:
            if call_amount == 0:
                return "check", 0
            if call_amount <= big_blind and "call" in legal:
                return "call", 0
            if "fold" in legal:
                return "fold", 0
            return "check", 0

        # ehs < 0.55
        if call_amount > 0 and "fold" in legal:
            return "fold", 0
        return "check", 0

    def _postflop(
        self,
        ehs: float,
        pot_odds: float,
        call_amount: int,
        pot: int,
        stack: int,
        min_raise: int,
        legal: list[str],
    ) -> tuple[str, int]:
        """Aggressive postflop betting based on EHS vs pot odds."""
        if ehs > pot_odds + self.raise_threshold and "raise" in legal:
            amount = self._raise_size(ehs, pot, stack, min_raise)
            return "raise", amount

        if ehs > pot_odds:
            if "call" in legal:
                return "call", 0
            return "check", 0

        if call_amount > 0 and "fold" in legal:
            return "fold", 0
        return "check", 0

    def _raise_size(self, ehs: float, pot: int, stack: int, min_raise: int) -> int:
        """Kelly-inspired raise sizing, capped at 75% of stack."""
        kelly = int(pot * (ehs - 0.5) * 2)
        return max(min_raise, min(kelly, int(0.75 * stack)))
