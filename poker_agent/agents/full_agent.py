"""Full agent — EHS + opponent modeling."""

from __future__ import annotations

from poker_agent.agents.base import Agent
from poker_agent.agents.ehs_agent import EHSAgent
from poker_agent.agents.utils import _legal_actions
from poker_agent.game import GameState
from poker_agent.monte_carlo import estimate_ehs
from poker_agent.opponent_model import OpponentModel


class FullAgent(Agent):
    """Combines Monte Carlo EHS with an opponent model that adjusts decisions.

    The opponent model persists across hands and returns two adjustments:

        call_adj              — additive shift applied to EHS when deciding
                                whether to call or fold the opponent's bet.
                                Negative when opponent is tight/strong.

        raise_threshold_reduction — subtracted from raise_threshold when deciding
                                    whether to bet/raise.  Positive when opponent
                                    folds to raises often (steal more pots).

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
        self._history_processed: int = 0      # entries already fed to the model
        self._last_hand_id: int | None = None  # game hand_id from last act()
        self._ehs_cache: dict[tuple, float] = {}
        self._hero_raised: bool = False        # True if our last unresponded action was a raise
        # Populated after every act() call — readable by visual display code
        self.last_decision: dict = {
            "ehs": 0.0, "adjusted_ehs": 0.0,
            "call_adj": 0.0, "threshold_reduction": 0.0,
            "effective_threshold": 0.0, "pot_odds": 0.0,
        }

    @property
    def opponent_model(self) -> OpponentModel:
        """Expose the opponent model for external inspection."""
        return self._opponent_model

    def reset(self) -> None:
        """Clear the opponent model and all session state."""
        self._opponent_model.reset()
        self._history_processed = 0
        self._last_hand_id = None
        self._hero_raised = False

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

        # Adjust for opponent tendencies (split: defensive call adj + offensive raise adj)
        call_adj, threshold_reduction = self._opponent_model.get_range_multiplier()
        adjusted_ehs = max(0.0, min(1.0, ehs + call_adj))
        effective_threshold = max(0.02, self.raise_threshold - threshold_reduction)

        call_amount = state.current_bet
        pot = state.pot
        pot_odds = call_amount / (pot + call_amount) if call_amount > 0 else 0.0
        stack = state.stacks[p]
        min_raise = state.min_raise

        action, amount = self._ehs_core._decide(
            adjusted_ehs, pot_odds, call_amount, pot, stack, min_raise, legal,
            raise_threshold=effective_threshold,
        )

        # Track our raise so the terminal-fold heuristic can fire next hand.
        # This must be set AFTER deciding — _hero_raised is consumed at the
        # start of the next act() call (in _sync_opponent_model) to detect
        # whether the opponent folded to end the hand.
        self._hero_raised = (action == "raise")

        self.last_decision = {
            "ehs": ehs,
            "adjusted_ehs": adjusted_ehs,
            "call_adj": call_adj,
            "threshold_reduction": threshold_reduction,
            "effective_threshold": effective_threshold,
            "pot_odds": pot_odds,
        }

        if self.verbose:
            print(
                f"  [Full] P{p} {state.street}: ehs={ehs:.3f} adj={adjusted_ehs:.3f} "
                f"call_adj={call_adj:+.3f} thr={effective_threshold:.3f} "
                f"pot_odds={pot_odds:.3f} → {action}"
                + (f" {amount}" if action == "raise" else "")
            )

        return action, amount

    # ------------------------------------------------------------------
    # Opponent model synchronization
    # ------------------------------------------------------------------

    def _sync_opponent_model(self, state: GameState, opp: int) -> None:
        """Feed new opponent actions from betting_history into the model.

        Detects hand boundaries via state.hand_id (incremented on each reset()).
        Finalizes once per completed hand, including hands where hero never acted.

        Terminal-fold heuristic
        ───────────────────────
        When hero raises and the opponent folds to end the hand, the game loop
        calls game.step(fold) and exits WITHOUT calling hero's act() again.
        This means the opponent's terminal fold is never seen in a subsequent
        sync call.  We detect this by checking _hero_raised at hand boundaries:
        if it is still True when the next hand starts, hero's last unresponded
        raise ended the hand — the opponent must have folded.  We record it.

        This heuristic is exact in heads-up: the only way hero raises and is
        never called again for that hand is if the opponent folded.
        """
        history = state.betting_history

        if self._last_hand_id is not None and state.hand_id > self._last_hand_id:
            # If _hero_raised is still set, opponent folded terminally to our raise
            if self._hero_raised:
                self._opponent_model.notify_faced_raise("fold")

            # Finalize completed hands, then reset per-hand state
            for _ in range(state.hand_id - self._last_hand_id):
                self._opponent_model._hand_open = True
                self._opponent_model._finalize_hand()
            self._history_processed = 0
            self._ehs_cache.clear()
            self._hero_raised = False

        self._last_hand_id = state.hand_id

        # Process entries added since our last act() call.
        # _hero_raised is NOT set here from history — it is set at the end of
        # act() based on the action being returned.  That way it always reflects
        # hero's most recent decision, not the previously-seen history state.
        for i in range(self._history_processed, len(history)):
            player, action, amount = history[i]
            if player == opp and action != "blind":
                if self._hero_raised:
                    # Opponent responded to our raise in the current hand
                    self._opponent_model.notify_faced_raise(action)
                    self._hero_raised = False   # consumed — wait for next raise
                self._opponent_model.update(action, state.street, state.hand_id)

        self._history_processed = len(history)
        self._opponent_model._hand_open = True

    def finalize_session(self) -> None:
        """Flush the last hand's per-hand stats. Call after simulation ends.

        Also captures any terminal fold from the very last hand of the session
        (the one that would normally be detected at the start of the next hand).
        """
        if self._hero_raised:
            self._opponent_model.notify_faced_raise("fold")
            self._hero_raised = False
        self._opponent_model._finalize_hand()
