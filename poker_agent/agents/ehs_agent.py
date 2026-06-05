"""EHS agent — Monte Carlo hand strength decision policy."""

from __future__ import annotations

from poker_agent.agents.base import Agent
from poker_agent.agents.utils import _legal_actions
from poker_agent.game import GameState
from poker_agent.monte_carlo import estimate_ehs


class EHSAgent(Agent):
    """Decisions from Monte Carlo EHS only (using pot odds for bet decisions).

    - No bet: check if ehs < open_bar (~0.55), else raise
    - Facing a bet: raise if ehs > pot_odds + raise_threshold;
                    call if ehs > pot_odds; else fold

    FullAgent reuses _decide() with use_pot_odds=True for pot-odds-aware play.

    Raise sizing (Kelly-inspired):
      raise_amount = pot * (ehs - 0.5) * 2,
      clipped to [min_raise, 0.75 * stack].
    """

    EHS_CALL_BAR = 0.50
    EHS_OPEN_BAR = 0.55

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
        """Return (action, amount) based on EHS strength bars & pot odds."""
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
        stack = state.stacks[p]
        min_raise = state.min_raise

        # Calculate pot odds if facing a bet, else 0.0
        pot_odds = (call_amount / (pot + call_amount)) if call_amount > 0 else 0.0

        action, amount = self._decide(
            ehs, pot_odds, call_amount, pot, stack, min_raise, legal,
            use_pot_odds=True,
        )

        if self.verbose:
        
            print(f"  [EHS] P{p} {state.street}: ehs={ehs:.3f} "
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
        raise_threshold: float | None = None,
        call_adj: float = 0.0,
        use_pot_odds: bool = True,
    ) -> tuple[str, int]:
        """Core decision logic.

        EHSAgent uses pot-odds-aware bars (use_pot_odds=True).

        raise_threshold: if provided, overrides self.raise_threshold for this
            decision only.  FullAgent uses this to pass a dynamically adjusted
            threshold without mutating the agent's persistent state.

        call_adj: opponent-model equity shift applied ONLY to the marginal
            call-vs-fold comparison.
        """
        threshold = raise_threshold if raise_threshold is not None else self.raise_threshold

        if call_amount == 0:
            open_bar = max(0.40, self.EHS_OPEN_BAR - (self.raise_threshold - threshold))
            if ehs >= open_bar and "raise" in legal:
                amount = self._raise_size(ehs, pot, stack, min_raise)
                return "raise", amount
            return "check", 0

        call_ehs = max(0.0, min(1.0, ehs + call_adj))
        if use_pot_odds:
            raise_bar = pot_odds + threshold
            call_bar = pot_odds
        else:
            raise_bar = self.EHS_OPEN_BAR + threshold
            call_bar = self.EHS_CALL_BAR

        if ehs > raise_bar and "raise" in legal:
            amount = self._raise_size(ehs, pot, stack, min_raise)
            return "raise", amount

        if call_ehs > call_bar:
            if "call" in legal:
                return "call", 0
            return "check", 0

        if "fold" in legal:
            return "fold", 0
        return "check", 0

    def _raise_size(self, ehs: float, pot: int, stack: int, min_raise: int) -> int:
        """Kelly-inspired raise sizing."""
        kelly = int(pot * (ehs - 0.5) * 2)
        max_raise = int(0.75 * stack)
        return max(min_raise, min(kelly, max_raise))
