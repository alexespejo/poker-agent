"""Opponent modeling — tracks betting statistics to adjust EHS estimates."""

from __future__ import annotations


class OpponentModel:
    """Tracks opponent betting patterns across hands within a session.

    Stats accumulated (reset only on explicit reset(), not between hands):
      - VPIP  (Voluntarily Put money In Pot): fraction of hands where opponent
              voluntarily bet or called preflop (excludes the forced big blind post)
      - PFR   (Preflop Raise rate): fraction of hands with a preflop raise
      - AF    (Aggression Factor): postflop (raises) / postflop (calls)
      - FTR   (Fold To Raise rate): fraction of raises opponent folds to

    get_range_multiplier() returns an EHS adjustment in [-0.10, +0.07]:
      - Tight opponent (low VPIP)  → their betting range is stronger
                                     → reduce our effective EHS by up to -0.10
      - Aggressive opponent (high AF) → they bluff more
                                     → increase our effective EHS by up to +0.07
      Both adjustments scale with sample size (discounted below 20 hands).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self.hands_seen: int = 0

        # Preflop voluntary action tracking
        self._hands_vpip: int = 0       # hands where opp voluntarily entered pot
        self._hands_pfr: int = 0        # hands with a preflop raise
        self._current_hand_vpip: bool = False
        self._current_hand_pfr: bool = False

        # Postflop aggression
        self._postflop_raises: int = 0
        self._postflop_calls: int = 0

        # Fold to raise
        self._faced_raises: int = 0
        self._folded_to_raises: int = 0

        # Track whether we're in a new hand (to count per-hand stats once)
        self._last_hand: int = -1

    # ------------------------------------------------------------------
    # Public update interface
    # ------------------------------------------------------------------

    def update(self, action: str, street: str, hand_number: int) -> None:
        """Record an observed opponent action.

        Call once per opponent action with the action string ('fold', 'call',
        'check', 'raise'), the current street, and the hand number.
        """
        if hand_number != self._last_hand:
            self._finalize_hand()
            self._last_hand = hand_number
            self._current_hand_vpip = False
            self._current_hand_pfr = False

        if street == "preflop":
            self._update_preflop(action)
        else:
            self._update_postflop(action)

    def notify_faced_raise(self, response: str) -> None:
        """Record whether the opponent folded to a raise we made.

        Call after the opponent responds to our raise: response is 'fold',
        'call', or 'raise'.
        """
        self._faced_raises += 1
        if response == "fold":
            self._folded_to_raises += 1

    def finalize_session(self) -> None:
        """Flush the last hand's per-hand stats. Call after simulation ends."""
        self._finalize_hand()

    # ------------------------------------------------------------------
    # Derived statistics (properties)
    # ------------------------------------------------------------------

    @property
    def vpip(self) -> float:
        """Voluntarily Put money In Pot rate (0–1). 0.0 if no data."""
        if self.hands_seen == 0:
            return 0.0
        return self._hands_vpip / self.hands_seen

    @property
    def pfr(self) -> float:
        """Preflop Raise rate (0–1). 0.0 if no data."""
        if self.hands_seen == 0:
            return 0.0
        return self._hands_pfr / self.hands_seen

    @property
    def aggression_factor(self) -> float:
        """Postflop (raises) / (calls). High = aggressive/bluffy. 0.0 if no data."""
        if self._postflop_calls == 0:
            # All-raises-no-calls → very aggressive; cap at 10.0
            return min(10.0, float(self._postflop_raises)) if self._postflop_raises > 0 else 0.0
        return self._postflop_raises / self._postflop_calls

    @property
    def fold_to_raise_rate(self) -> float:
        """Fraction of our raises that opponent folds to. 0.0 if no data."""
        if self._faced_raises == 0:
            return 0.0
        return self._folded_to_raises / self._faced_raises

    # ------------------------------------------------------------------
    # EHS adjustment
    # ------------------------------------------------------------------

    def get_range_multiplier(self) -> float:
        """Return an additive EHS adjustment based on opponent tendencies.

        Positive = raise our effective EHS (opponent is bluffy / weak).
        Negative = lower our effective EHS (opponent plays tight / strong).
        Clamped to [-0.10, +0.12].

        Scaled by min(1.0, hands_seen / 20) to discount with sparse data.
        FTR adjustment uses its own weight: min(1.0, faced_raises / 10).
        """
        sample_weight = min(1.0, self.hands_seen / 20)

        tight_adj = 0.0
        aggro_adj = 0.0
        ftr_adj = 0.0

        # Tight player: VPIP well below 0.35 → their hands are stronger on average
        if self.hands_seen > 0 and self.vpip < 0.35:
            deviation = 0.35 - self.vpip          # max deviation ≈ 0.35
            tight_adj = -(deviation / 0.35) * 0.10  # scale to [-0.10, 0]

        # Aggressive/bluffy player: AF well above 2.0
        if self.hands_seen > 0 and self.aggression_factor > 2.0:
            deviation = min(self.aggression_factor - 2.0, 8.0)  # cap at 8 above threshold
            aggro_adj = (deviation / 8.0) * 0.07                # scale to [0, +0.07]

        # Fold-to-raise: opponent folds often → we can bluff/semi-bluff more
        if self._faced_raises > 0 and self.fold_to_raise_rate > 0.5:
            ftr_weight = min(1.0, self._faced_raises / 10)
            deviation = min(self.fold_to_raise_rate - 0.5, 0.5)  # max 0.5 above threshold
            ftr_adj = (deviation / 0.5) * 0.08 * ftr_weight       # scale to [0, +0.08]

        raw = tight_adj + aggro_adj
        adjusted = raw * sample_weight + ftr_adj
        return max(-0.10, min(0.12, adjusted))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_preflop(self, action: str) -> None:
        """Update preflop voluntary action flags."""
        if action in ("call", "raise"):
            self._current_hand_vpip = True
        if action == "raise":
            self._current_hand_pfr = True

    def _update_postflop(self, action: str) -> None:
        """Update postflop aggression counters."""
        if action == "raise":
            self._postflop_raises += 1
        elif action == "call":
            self._postflop_calls += 1

    def _finalize_hand(self) -> None:
        """Flush per-hand booleans into hand-level counters.

        Sets _last_hand back to -1 after flushing so repeated calls
        (e.g. finalize_session() followed by the next hand's sync) are no-ops.
        """
        if self._last_hand == -1:
            return  # no hand started, or already finalized
        self.hands_seen += 1
        if self._current_hand_vpip:
            self._hands_vpip += 1
        if self._current_hand_pfr:
            self._hands_pfr += 1
        self._last_hand = -1  # prevent double-counting
