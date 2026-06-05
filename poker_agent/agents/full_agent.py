"""Full agent — EHS + opponent modeling."""

from __future__ import annotations

from typing import Literal

from poker_agent.agents.base import Agent
from poker_agent.agents.ehs_agent import EHSAgent
from poker_agent.agents.utils import _legal_actions
from poker_agent.game import GameState
from poker_agent.learning_mode import OpponentLearningMode
from poker_agent.monte_carlo import estimate_ehs
from poker_agent.opponent_model import OpponentModel, OpponentModelSnapshot

Phase = Literal["observe", "apply", "frozen_apply"]


class FullAgent(Agent):
    """Combines Monte Carlo EHS with an opponent model that adjusts decisions.

    The opponent model persists across hands and returns two adjustments:

        call_adj              — additive shift applied to EHS when deciding
                                whether to call or fold the opponent's bet.
                                Negative when opponent is tight/strong.

        raise_threshold_reduction — subtracted from raise_threshold when deciding
                                    whether to bet/raise.  Positive when opponent
                                    folds to raises often (steal more pots).

    learning_mode controls warm-up vs live learning (see OpponentLearningMode).
    """

    def __init__(
        self,
        n_samples: int = 500,
        raise_threshold: float = 0.15,
        verbose: bool = False,
        learning_mode: OpponentLearningMode = OpponentLearningMode.LIVE,
    ) -> None:
        self.n_samples = n_samples
        self.raise_threshold = raise_threshold
        self.verbose = verbose
        self.learning_mode = learning_mode
        self._opponent_model = OpponentModel()
        self._ehs_core = EHSAgent(n_samples=n_samples, raise_threshold=raise_threshold)
        self._history_processed: int = 0
        self._last_hand_id: int | None = None
        self._hero_raised: bool = False
        self._ehs_cache: dict[tuple, float] = {}
        self._phase: Phase = "apply"
        self._frozen_snapshot: OpponentModelSnapshot | None = None
        self._warmup_diag: dict | None = None
        self.last_decision: dict = {
            "ehs": 0.0, "adjusted_ehs": 0.0,
            "call_adj": 0.0, "threshold_reduction": 0.0,
            "effective_threshold": 0.0, "pot_odds": 0.0,
            "phase": "apply",
            "hands_seen": 0,
        }

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def opponent_model(self) -> OpponentModel:
        """Expose the opponent model for external inspection."""
        return self._opponent_model

    @property
    def warmup_diagnostics(self) -> dict | None:
        """Stats and adjustments at end of warm-up (None if no warm-up ran)."""
        return self._warmup_diag

    def reset(self) -> None:
        """Clear the opponent model and all session state."""
        self._opponent_model.reset()
        self._history_processed = 0
        self._last_hand_id = None
        self._hero_raised = False
        self._ehs_cache.clear()
        self._frozen_snapshot = None
        self._warmup_diag = None
        self._phase = "apply"

    def begin_session(self, warmup_hands: int) -> None:
        """Called by run_simulation before the first hand."""
        if self.learning_mode == OpponentLearningMode.LIVE or warmup_hands == 0:
            self._phase = "apply"
        else:
            self._phase = "observe"

    def begin_scored_phase(self) -> None:
        """Transition from warm-up to scored segment; net chips reset externally."""
        if self._hero_raised:
            self._opponent_model.notify_faced_raise("fold")
            self._hero_raised = False
        self._opponent_model._finalize_hand()

        call_adj, thr_red = self._opponent_model.get_range_multiplier()
        self._warmup_diag = {
            "hands_seen": self._opponent_model.hands_seen,
            "vpip": self._opponent_model.vpip,
            "af": self._opponent_model.aggression_factor,
            "ftr": self._opponent_model.fold_to_raise_rate,
            "call_adj": call_adj,
            "thr_red": thr_red,
        }

        if self.learning_mode == OpponentLearningMode.WARMUP_THEN_FROZEN:
            self._frozen_snapshot = self._opponent_model.snapshot()
            self._phase = "frozen_apply"
        else:
            self._phase = "apply"

        self._clear_hand_sync_state()

    def _clear_hand_sync_state(self) -> None:
        self._history_processed = 0
        self._last_hand_id = None
        self._hero_raised = False
        self._ehs_cache.clear()

    def _get_adjustments(self) -> tuple[float, float]:
        if self._phase == "observe":
            return 0.0, 0.0
        if self._phase == "frozen_apply" and self._frozen_snapshot is not None:
            return self._opponent_model.get_range_multiplier_from_snapshot(
                self._frozen_snapshot
            )
        return self._opponent_model.get_range_multiplier()

    def act(self, game_state: GameState, player_id: int) -> tuple[str, int]:
        """Return (action, amount) using EHS adjusted by opponent modeling."""
        state = game_state
        p = player_id
        opp = 1 - p
        legal = _legal_actions(state, p)

        if self._phase != "frozen_apply":
            self._sync_opponent_model(state, opp)

        hole = state.hole_cards[p]
        community = state.community_cards
        cache_key = (tuple(hole), tuple(community))
        if cache_key in self._ehs_cache:
            ehs = self._ehs_cache[cache_key]
        else:
            dead = hole + community
            ehs = estimate_ehs(hole, community, dead, self.n_samples)
            self._ehs_cache[cache_key] = ehs

        call_adj, threshold_reduction = self._get_adjustments()
        adjusted_ehs = max(0.0, min(1.0, ehs + call_adj))
        effective_threshold = max(0.02, self.raise_threshold - threshold_reduction)

        call_amount = state.current_bet
        pot = state.pot
        pot_odds = call_amount / (pot + call_amount) if call_amount > 0 else 0.0
        stack = state.stacks[p]
        min_raise = state.min_raise

        action, amount = self._ehs_core._decide(
            ehs, pot_odds, call_amount, pot, stack, min_raise, legal,
            raise_threshold=effective_threshold,
            call_adj=call_adj,
            use_pot_odds=True,
        )

        self._hero_raised = (action == "raise")

        self.last_decision = {
            "ehs": ehs,
            "adjusted_ehs": adjusted_ehs,
            "call_adj": call_adj,
            "threshold_reduction": threshold_reduction,
            "effective_threshold": effective_threshold,
            "pot_odds": pot_odds,
            "phase": self._phase,
            "hands_seen": self._opponent_model.hands_seen,
        }

        if self.verbose:
            print(
                f"  [Full] P{p} {state.street} ({self._phase}): ehs={ehs:.3f} "
                f"adj={adjusted_ehs:.3f} call_adj={call_adj:+.3f} "
                f"thr={effective_threshold:.3f} pot_odds={pot_odds:.3f} → {action}"
                + (f" {amount}" if action == "raise" else "")
            )

        return action, amount

    def _sync_opponent_model(self, state: GameState, opp: int) -> None:
        """Feed new opponent actions from betting_history into the model."""
        history = state.betting_history

        if self._last_hand_id is not None and state.hand_id > self._last_hand_id:
            if self._hero_raised:
                self._opponent_model.notify_faced_raise("fold")

            for _ in range(state.hand_id - self._last_hand_id):
                self._opponent_model._hand_open = True
                self._opponent_model._finalize_hand()
            self._history_processed = 0
            self._ehs_cache.clear()
            self._hero_raised = False

        self._last_hand_id = state.hand_id

        for i in range(self._history_processed, len(history)):
            player, action, amount = history[i]
            if player == opp and action != "blind":
                if self._hero_raised:
                    self._opponent_model.notify_faced_raise(action)
                    self._hero_raised = False
                self._opponent_model.update(action, state.street, state.hand_id)

        self._history_processed = len(history)
        self._opponent_model._hand_open = True

    def finalize_session(self) -> None:
        """Flush the last hand's per-hand stats. Call after simulation ends."""
        if self._hero_raised:
            self._opponent_model.notify_faced_raise("fold")
            self._hero_raised = False
        self._opponent_model._finalize_hand()
