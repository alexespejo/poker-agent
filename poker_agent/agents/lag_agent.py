"""Loose-aggressive (LAG) benchmarking agent.

Enters a wide range of pots preflop (VPIP ~0.65-0.75) and applies postflop
pressure with semi-bluffs and value bets (AF ~2.5-4.0). Folds to strong raises
sometimes but not reflexively (FTR ~0.35-0.50). Designed to exercise FullAgent's
VPIP + high AF and recognise it can call lighter.
"""

from __future__ import annotations

import random

from poker_agent.agents.base import Agent
from poker_agent.agents.utils import _legal_actions
from poker_agent.game import GameState
from poker_agent.monte_carlo import estimate_ehs


class LooseAggressiveAgent(Agent):
    """Wide preflop ranges, frequent postflop aggression with semi-bluffs.

    Preflop (EHS from hole cards only):
      ehs >= 0.52              → raise (Kelly sizing, min 2x big blind)
      0.40 <= ehs < 0.52       → call (limp/flat)
      ehs < 0.40               → fold to a bet > 1 BB, check if free

    Postflop (EHS from hole + community):
      ehs > pot_odds + 0.06 and raise legal                     → raise (value)
      ehs > pot_odds - 0.05 and raise legal and pot > 2 BB      → raise 40% of
                                                                   the time (semi-bluff)
      ehs > pot_odds - 0.10                                     → call
      otherwise                                                 → fold (or check)
    """

    value_threshold: float = 0.06
    semibluff_threshold: float = 0.05
    call_threshold: float = 0.10
    semibluff_prob: float = 0.40

    def __init__(
        self,
        n_samples: int = 200,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.n_samples = n_samples
        self.verbose = verbose
        self._rng = random.Random(seed)

    def reset(self) -> None:
        """No persistent state."""
        pass

    def act(self, game_state: GameState, player_id: int) -> tuple[str, int]:
        """Return (action, amount) for the loose-aggressive policy."""
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
                ehs, pot_odds, call_amount, pot, stack, min_raise, big_blind, legal
            )

        if self.verbose:
            print(
                f"  [LAG] P{p} {state.street}: ehs={ehs:.3f} call={call_amount} "
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
        """Wide preflop range with frequent raising.

        Premiums (ehs >= 0.62) raise/re-raise for value. Marginal openers
        (0.52 <= ehs < 0.62) open-raise when first in but flat a re-raise rather
        than stacking off light against a tight 3-betting range — this keeps the
        range wide and aggressive without the spewy 4-bet variance.
        """
        facing_raise = call_amount > big_blind

        if ehs >= 0.62 and "raise" in legal:
            return "raise", self._open_raise_size(ehs, pot, stack, min_raise, big_blind)

        if 0.52 <= ehs < 0.62:
            if not facing_raise and "raise" in legal:
                return "raise", self._open_raise_size(ehs, pot, stack, min_raise, big_blind)
            if "call" in legal:
                return "call", 0
            return "check", 0

        if 0.40 <= ehs < 0.52:
            if call_amount == 0:
                return "check", 0
            if "call" in legal:
                return "call", 0
            return "check", 0

        # ehs < 0.40
        if call_amount > big_blind and "fold" in legal:
            return "fold", 0
        if call_amount > 0 and "call" in legal:
            return "call", 0
        return "check", 0

    def _postflop(
        self,
        ehs: float,
        pot_odds: float,
        call_amount: int,
        pot: int,
        stack: int,
        min_raise: int,
        big_blind: int,
        legal: list[str],
    ) -> tuple[str, int]:
        """Aggressive postflop play with semi-bluffs."""
        # Value raise
        if ehs > pot_odds + self.value_threshold and "raise" in legal:
            amount = self._raise_size(ehs, pot, stack, min_raise, cap=0.75)
            return "raise", amount

        # Semi-bluff / pressure bet: marginal equity, meaningful pot, fire 40%.
        # Sized as a genuine pressure bet (half-pot floor) — a min-raise stab is
        # not pressure and just gets called by passive opponents. Still capped at
        # 50% of stack to limit variance, per the loose-aggressive profile.
        if (
            ehs > pot_odds - self.semibluff_threshold
            and "raise" in legal
            and pot > 2 * big_blind
            and self._rng.random() < self.semibluff_prob
        ):
            amount = self._raise_size(ehs, pot, stack, min_raise, cap=0.5, floor_frac=0.5)
            return "raise", amount

        # Call light
        if ehs > pot_odds - self.call_threshold:
            if "call" in legal:
                return "call", 0
            return "check", 0

        if call_amount > 0 and "fold" in legal:
            return "fold", 0
        return "check", 0

    def _open_raise_size(
        self, ehs: float, pot: int, stack: int, min_raise: int, big_blind: int
    ) -> int:
        """Preflop raise sizing: Kelly with a 2 BB floor, capped at 75% stack."""
        kelly = int(pot * (ehs - 0.5) * 2)
        amount = max(kelly, int(2 * big_blind))
        amount = min(amount, int(0.75 * stack))
        return max(min_raise, amount)

    def _raise_size(
        self,
        ehs: float,
        pot: int,
        stack: int,
        min_raise: int,
        cap: float,
        floor_frac: float = 0.0,
    ) -> int:
        """Kelly-inspired raise sizing, capped at `cap` fraction of stack.

        floor_frac raises the bet to at least that fraction of the pot — used
        for semi-bluffs so the pressure bet is large enough to fold out passive
        opponents rather than getting flatted at min-raise size.
        """
        kelly = int(pot * (ehs - 0.5) * 2)
        amount = max(kelly, int(floor_frac * pot))
        return max(min_raise, min(amount, int(cap * stack)))
