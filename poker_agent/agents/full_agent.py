"""Full agent — EHS + opponent modeling."""

from __future__ import annotations

import random

from poker_agent.agents.base import Agent
from poker_agent.agents.ehs_agent import EHSAgent
from poker_agent.agents.utils import _legal_actions
from poker_agent.game import GameState
from poker_agent.monte_carlo import estimate_ehs
from poker_agent.opponent_model import OpponentModel

# Exploitation tuning (Bug 5). Bluffing only ramps in once we have a profile.
_EXPLOIT_MIN_CONFIDENCE = 0.2     # no exploitation below this confidence
_FTR_BLUFF_PIVOT = 0.4            # FTR above this is exploitable fold-equity
_BLUFF_PROB_SCALE = 2.0          # scales confidence*(FTR-pivot) into a probability
_BLUFF_PROB_CAP = 0.75           # never bluff more often than this
_STATION_FTR = 0.30              # FTR below this = sticky caller (value, never bluff)
_STATION_VPIP = 0.50             # VPIP above this = loose caller
_STATION_POT_CONTROL = 0.30      # raise-threshold premium vs a station (pot control)


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
        seed: int | None = None,
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
        # Seeded RNG for all stochastic exploit decisions (reproducibility).
        self._rng = random.Random(seed)
        self._last_exploit: str | None = None  # which exploit fired on last act()
        # Populated after every act() call — readable by visual display code
        self.last_decision: dict = {
            "ehs": 0.0, "adjusted_ehs": 0.0,
            "call_adj": 0.0, "threshold_reduction": 0.0,
            "effective_threshold": 0.0, "pot_odds": 0.0,
            "range_filter": None, "exploit": None,
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
        self._last_exploit = None
        self._ehs_cache.clear()

    def act(self, game_state: GameState, player_id: int) -> tuple[str, int]:
        """Return (action, amount) using EHS adjusted by opponent modeling."""
        state = game_state
        p = player_id
        opp = 1 - p
        legal = _legal_actions(state, p)

        # Feed any new opponent actions into the model before deciding
        self._sync_opponent_model(state, opp)

        call_amount = state.current_bet
        pot = state.pot
        pot_odds = call_amount / (pot + call_amount) if call_amount > 0 else 0.0
        stack = state.stacks[p]
        min_raise = state.min_raise

        model = self._opponent_model

        # Select a range filter from the opponent's profile. When we are facing a
        # bet, the opponent has put chips in aggressively → condition our equity
        # on their (loosely-estimated) raising range so we stop committing light
        # into a raise war. With no bet to face there is no aggression signal, so
        # we keep the unconditioned (uniform) estimate.
        range_filter = None
        raise_frac_token = None
        if call_amount > 0 and model.hands_seen > 0:
            range_filter = model.make_range_filter_for_action("raise")
            raise_frac_token = round(model.get_raise_range_fraction(), 2)

        # Range-conditioned EHS — cached per (hole, community, filter) within a hand
        hole = state.hole_cards[p]
        community = state.community_cards
        cache_key = (tuple(hole), tuple(community), raise_frac_token)
        if cache_key in self._ehs_cache:
            ehs = self._ehs_cache[cache_key]
        else:
            dead = hole + community
            ehs = estimate_ehs(hole, community, dead, self.n_samples, range_filter=range_filter)
            self._ehs_cache[cache_key] = ehs

        # Adjust for opponent tendencies (split: defensive call adj + offensive raise adj)
        call_adj, threshold_reduction = model.get_range_multiplier()

        # [Bug 1] Tightness must not be double-counted. When a range filter is
        # active the conditioned EHS already accounts for the opponent's range,
        # so we do NOT additionally apply get_range_multiplier()'s call_adj — that
        # additive multiplier is a fallback for the unfiltered path only.
        if range_filter is not None:
            adjusted_ehs = max(0.0, min(1.0, ehs))
        else:
            adjusted_ehs = max(0.0, min(1.0, ehs + call_adj))

        # [Bug 5] Pot control vs a station. A loose opponent that almost never
        # folds to raises (low FTR, high VPIP) cannot be bluffed and punishes
        # bloated pots with marginal hands — the source of the losing raise war.
        # Against that profile we RAISE our re-raise bar (raise only for clear
        # value, otherwise call/check), scaled by confidence. Against folders the
        # opposite happens via threshold_reduction below.
        conf = model.confidence
        station = (
            conf >= _EXPLOIT_MIN_CONFIDENCE
            and model.fold_to_raise_rate < _STATION_FTR
            and model.vpip > _STATION_VPIP
        )
        station_increase = _STATION_POT_CONTROL * conf if station else 0.0
        effective_threshold = max(
            0.02, self.raise_threshold - threshold_reduction + station_increase
        )

        action, amount = self._decide(
            adjusted_ehs, pot_odds, call_amount, pot, stack, min_raise, legal,
            effective_threshold,
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
            "range_filter": raise_frac_token,
            "exploit": self._last_exploit,
        }

        if self.verbose:
            filt = f" rf={raise_frac_token}" if raise_frac_token is not None else ""
            expl = f" EXPLOIT={self._last_exploit}" if self._last_exploit else ""
            print(
                f"  [Full] P{p} {state.street}: ehs={ehs:.3f} adj={adjusted_ehs:.3f} "
                f"call_adj={call_adj:+.3f} thr={effective_threshold:.3f} "
                f"pot_odds={pot_odds:.3f}{filt} → {action}"
                + (f" {amount}" if action == "raise" else "")
                + expl
            )

        return action, amount

    # ------------------------------------------------------------------
    # Decision policy (EHS core + opponent-model exploitation)
    # ------------------------------------------------------------------

    def _decide(
        self,
        ehs: float,
        pot_odds: float,
        call_amount: int,
        pot: int,
        stack: int,
        min_raise: int,
        legal: list[str],
        effective_threshold: float,
    ) -> tuple[str, int]:
        """EHS/pot-odds decision plus FTR-driven exploitation.  [Bug 5]

        We start from the shared EHS core decision, then layer two exploits that
        attack EHSAgent's two leaks — it never bluffs and it folds whenever
        ehs <= pot_odds:

          • Bluff-raise (no bet to face): when our equity is weak but the
            opponent folds to raises often, raise as a steal a meaningful
            fraction of the time, sized up so fold equity is maximised.
          • Bluff-raise vs a marginal bet: when we'd otherwise fold/call, raise
            to fold them off, again only when fold-to-raise is high.

        Both fire with probability ``confidence * max(0, FTR - 0.4) * scale``,
        gated off entirely below ``_EXPLOIT_MIN_CONFIDENCE`` so early hands play
        straightforwardly. Against a confirmed station (low FTR, high VPIP) we do
        the opposite: never bluff, and size raises for value.
        All randomness draws from the seeded ``self._rng``.
        """
        self._last_exploit = None
        model = self._opponent_model
        action, amount = self._ehs_core._decide(
            ehs, pot_odds, call_amount, pot, stack, min_raise, legal,
            raise_threshold=effective_threshold,
        )

        conf = model.confidence
        ftr = model.fold_to_raise_rate
        vpip = model.vpip

        # Exploitation stays off until we actually have a profile.
        if conf < _EXPLOIT_MIN_CONFIDENCE or "raise" not in legal:
            if action == "raise":
                amount = self._sized_raise(ehs, pot, stack, min_raise, ftr, vpip, conf)
            return action, amount

        bluff_prob = min(
            _BLUFF_PROB_CAP,
            conf * max(0.0, ftr - _FTR_BLUFF_PIVOT) * _BLUFF_PROB_SCALE,
        )

        # Exploit 1 — steal an unopened pot with a weak hand against a folder.
        if call_amount == 0 and action == "check" and ehs < 0.5 and bluff_prob > 0.0:
            if self._rng.random() < bluff_prob:
                self._last_exploit = "bluff_open"
                return "raise", self._bluff_size(pot, stack, min_raise)

        # Exploit 2 — raise a marginal bet to fold them off (semi/pure bluff).
        if call_amount > 0 and action in ("fold", "call") and bluff_prob > 0.0:
            if self._rng.random() < bluff_prob:
                self._last_exploit = "bluff_steal"
                return "raise", self._bluff_size(pot, stack, min_raise)

        # No exploit fired — size raises for value (extra when facing a station).
        if action == "raise":
            amount = self._sized_raise(ehs, pot, stack, min_raise, ftr, vpip, conf)
        return action, amount

    def _bluff_size(self, pot: int, stack: int, min_raise: int) -> int:
        """Large bet-sizing for bluffs — fold equity rises with size.  [Bug 5]"""
        target = int(0.85 * max(pot, 2 * min_raise))
        cap = int(0.75 * stack)
        return max(min_raise, min(target, cap))

    def _sized_raise(
        self,
        ehs: float,
        pot: int,
        stack: int,
        min_raise: int,
        ftr: float,
        vpip: float,
        conf: float,
    ) -> int:
        """Value raise sizing. Against a confirmed station, size up for value.  [Bug 5]"""
        base = self._ehs_core._raise_size(ehs, pot, stack, min_raise)
        if conf >= _EXPLOIT_MIN_CONFIDENCE and ftr < _STATION_FTR and vpip > _STATION_VPIP:
            # Sticky caller: charge a premium with strong hands, never bluff them.
            value = int(pot * (ehs - 0.5) * 3)
            cap = int(0.75 * stack)
            return max(base, min(value, cap)) if ehs > 0.5 else base
        return base

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
