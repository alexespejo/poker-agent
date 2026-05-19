"""Full agent — EHS + opponent modeling."""

from __future__ import annotations

from poker_agent.agents.base import Agent
from poker_agent.agents.ehs_agent import EHSAgent
from poker_agent.agents.random_agent import _legal_actions
from poker_agent.game import GameState
from poker_agent.monte_carlo import estimate_ehs
from poker_agent.opponent_model import OpponentModel


class FullAgent(Agent):
    """Combines Monte Carlo EHS with an opponent model that adjusts decisions.

    The opponent model persists across hands within a session — it accumulates
    VPIP, PFR, and aggression stats over time and uses them to adjust the
    effective EHS before applying the pot-odds decision policy.

        adjusted_ehs = clamp(ehs + opponent_model.get_range_multiplier(), 0.0, 1.0)

    Explicit reset() clears the opponent model. The model is NOT reset between
    hands — cross-hand learning is the point.
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
        self._opponent_model = OpponentModel()
        self._ehs_core = EHSAgent(n_samples=n_samples, raise_threshold=raise_threshold)
        self._history_id: int | None = None   # id() of the current hand's history list
        self._history_processed: int = 0      # entries already fed to the model
        self._hand_number: int = 0            # monotonically increasing hand counter
        self._ehs_cache: dict[tuple, float] = {}
        # Populated after every act() call — readable by visual display code
        self.last_decision: dict = {
            "ehs": 0.0, "adjusted_ehs": 0.0,
            "multiplier": 0.0, "pot_odds": 0.0,
        }

    @property
    def opponent_model(self) -> OpponentModel:
        """Expose the opponent model for external inspection."""
        return self._opponent_model

    def reset(self) -> None:
        """Clear the opponent model and all session state."""
        self._opponent_model.reset()
        self._history_id = None
        self._history_processed = 0
        self._hand_number = 0

    def act(self, game_state: GameState, player_id: int) -> tuple[str, int]:
        """Return (action, amount) using EHS adjusted by opponent modeling."""
        state = game_state
        p = player_id
        opp = 1 - p
        legal = _legal_actions(state, p)

        # Feed any new opponent actions into the model before deciding
        self._sync_opponent_model(state, opp)

        # Raw EHS — cached per (hole, community) within a hand
        hole = state.hole_cards[p]
        community = state.community_cards
        cache_key = (tuple(hole), tuple(community))
        if cache_key in self._ehs_cache:
            ehs = self._ehs_cache[cache_key]
        else:
            dead = hole + community
            ehs = estimate_ehs(hole, community, dead, self.n_samples)
            self._ehs_cache[cache_key] = ehs

        # Adjust for opponent tendencies
        multiplier = self._opponent_model.get_range_multiplier()
        adjusted_ehs = max(0.0, min(1.0, ehs + multiplier))

        call_amount = state.current_bet
        pot = state.pot
        pot_odds = call_amount / (pot + call_amount) if call_amount > 0 else 0.0
        stack = state.stacks[p]
        min_raise = state.min_raise

        action, amount = self._ehs_core._decide(
            adjusted_ehs, pot_odds, call_amount, pot, stack, min_raise, legal
        )

        self.last_decision = {
            "ehs": ehs,
            "adjusted_ehs": adjusted_ehs,
            "multiplier": multiplier,
            "pot_odds": pot_odds,
        }

        if self.verbose:
            print(
                f"  [Full] P{p} {state.street}: ehs={ehs:.3f} adj={adjusted_ehs:.3f} "
                f"mult={multiplier:+.3f} pot_odds={pot_odds:.3f} → {action}"
                + (f" {amount}" if action == "raise" else "")
            )

        return action, amount

    # ------------------------------------------------------------------
    # Opponent model synchronization
    # ------------------------------------------------------------------

    def _sync_opponent_model(self, state: GameState, opp: int) -> None:
        """Feed new opponent actions from betting_history into the model.

        Uses id(betting_history) to detect hand boundaries — reset() in
        PokerGame always creates a new list, so a changed id means a new hand.
        Uses state.street for all newly-seen entries; this is accurate because
        we process incrementally (each new entry arrived during the current
        street or earlier in this call's street transition).
        """
        history = state.betting_history
        current_id = id(history)

        # New hand detected — finalize the previous hand's per-hand stats
        if current_id != self._history_id:
            if self._history_id is not None:
                self._opponent_model._finalize_hand()
            self._history_id = current_id
            self._history_processed = 0
            self._hand_number += 1
            self._ehs_cache.clear()

        # Process entries added since our last act() call
        for i in range(self._history_processed, len(history)):
            player, action, amount = history[i]
            if player == opp and action != "blind":
                self._opponent_model.update(action, state.street, self._hand_number)

        self._history_processed = len(history)

    def finalize_session(self) -> None:
        """Flush the last hand's per-hand stats. Call after simulation ends."""
        self._opponent_model._finalize_hand()
