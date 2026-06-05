"""Opponent modeling — tracks betting statistics to adjust EHS estimates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpponentModelSnapshot:
    """Frozen copy of OpponentModel counters for warmup_then_frozen."""

    hands_seen: int
    _hands_vpip: int
    _hands_pfr: int
    _postflop_raises: int
    _postflop_calls: int
    _faced_raises: int
    _folded_to_raises: int


class OpponentModel:
    """Tracks opponent betting patterns across hands within a session.

    Stats accumulated (reset only on explicit reset(), not between hands):
      - VPIP  (Voluntarily Put money In Pot): fraction of hands where opponent
              voluntarily bet or called preflop (excludes the forced big blind post)
      - PFR   (Preflop Raise rate): fraction of hands with a preflop raise
      - AF    (Aggression Factor): postflop (raises) / postflop (calls)
      - FTR   (Fold To Raise rate): fraction of raises opponent folds to

    get_range_multiplier() returns (call_adj, raise_threshold_reduction):
      - Tight opponent (low VPIP)  → their betting range is stronger
                                     → reduce our call equity (call_adj < 0)
      - Tight opponent (low VPIP)  → also folds most hands preflop
                                     → widen our opening range to steal
                                       (raise_threshold_reduction > 0)
      Aggression (AF) is deliberately NOT used to loosen our calls: without
      showdown data a high AF cannot be distinguished from a value-heavy
      maniac, so loosening into it loses chips.  Adjustments scale with sample
      size (discounted below 20 hands).
    """

    def __init__(self) -> None:
        self.reset()

    def snapshot(self) -> OpponentModelSnapshot:
        """Capture current stats (e.g. at end of warm-up)."""
        return OpponentModelSnapshot(
            hands_seen=self.hands_seen,
            _hands_vpip=self._hands_vpip,
            _hands_pfr=self._hands_pfr,
            _postflop_raises=self._postflop_raises,
            _postflop_calls=self._postflop_calls,
            _faced_raises=self._faced_raises,
            _folded_to_raises=self._folded_to_raises,
        )

    def load_snapshot(self, snap: OpponentModelSnapshot) -> None:
        """Replace live counters with a snapshot (does not reset hand-boundary flags)."""
        self.hands_seen = snap.hands_seen
        self._hands_vpip = snap._hands_vpip
        self._hands_pfr = snap._hands_pfr
        self._postflop_raises = snap._postflop_raises
        self._postflop_calls = snap._postflop_calls
        self._faced_raises = snap._faced_raises
        self._folded_to_raises = snap._folded_to_raises

    def get_range_multiplier_from_snapshot(
        self, snap: OpponentModelSnapshot,
    ) -> tuple[float, float]:
        """Adjustments as if snap were the live model (for frozen_apply phase)."""
        saved = self.snapshot()
        self.load_snapshot(snap)
        try:
            return self.get_range_multiplier()
        finally:
            self.load_snapshot(saved)

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
        self._hand_open: bool = False  # True once a hand is underway (even w/o updates)

    # ------------------------------------------------------------------
    # Public update interface
    # ------------------------------------------------------------------

    def update(self, action: str, street: str, hand_number: int) -> None:
        """Record an observed opponent action.

        Call once per opponent action with the action string ('fold', 'call',
        'check', 'raise'), the current street, and the hand number.
        """
        if hand_number != self._last_hand:
            # Sync may have already finalized (_last_hand == -1); avoid double-count.
            if self._last_hand != -1:
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

    def get_range_multiplier(self) -> tuple[float, float]:
        """Return (call_adj, raise_threshold_reduction).

        call_adj — additive EHS adjustment applied ONLY to the marginal
            call-vs-fold decision against the opponent's bet (FullAgent passes
            this through to _decide; it no longer inflates our opening range or
            bet sizing).
              Negative = opponent plays tight/strong → need a better hand to call.
            Clamped to [-0.12, +0.10].

        raise_threshold_reduction — amount to subtract from raise_threshold when
            deciding whether to initiate a bet/raise.  Positive lowers the raise
            bar (steal/bluff more), driven by two signals:
              - the opponent folds to our raises often (FTR), and
              - the opponent is very tight preflop (low VPIP) and folds most hands.
            Clamped to [0, +0.10].

        Both are scaled by min(1.0, hands_seen / 20) to discount sparse data.
        The FTR signal uses its own weight: min(1.0, faced_raises / 10).

        Aggression (AF) is NOT used to loosen calls — without showdown data a
        high AF cannot be distinguished from a value-heavy maniac.
        """
        sample_weight = min(1.0, self.hands_seen / 20)

        # ── Defensive component (call_adj) ──────────────────────────────────

        tight_adj = 0.0
        aggro_adj = 0.0

        # Tight player: VPIP < 0.35 → their range is stronger → need better hand to call
        if self.hands_seen > 0 and self.vpip < 0.35:
            deviation = 0.35 - self.vpip
            tight_adj = -(deviation / 0.35) * 0.10   # up to -0.10

        # Aggression alone is NOT evidence of bluffing.  A high AF is just as
        # likely a value-heavy maniac (loose-aggressive) as a bluffer, and with
        # no showdown data we cannot tell them apart.  We therefore never *loosen*
        # on AF: calling lighter into unexplained aggression is exactly what
        # regressed against the LAG archetype (a value/semi-bluff bettor).  The
        # safe response to heavy aggression is disciplined pot-odds calling.
        aggro_adj = 0.0

        # Tight-aggressive: low VPIP AND high AF → their bets are even more
        # value-heavy.  A modest extra tighten is justified, but keep it small so
        # we don't over-fold the marginal calls that plain EHS already wins with.
        if self.hands_seen > 0 and self.aggression_factor > 2.0 and self.vpip < 0.40:
            ta_bonus = min((self.aggression_factor - 2.0) / 8.0, 1.0) * 0.02
            tight_adj -= ta_bonus                     # extra tightening (max -0.02)

        call_adj = (tight_adj + aggro_adj) * sample_weight
        call_adj = max(-0.12, min(0.10, call_adj))

        # ── Offensive component (raise_threshold_reduction) ──────────────────

        raise_threshold_reduction = 0.0
        if self._faced_raises >= 5:
            ftr_weight = min(1.0, self._faced_raises / 10)
            if self.fold_to_raise_rate > 0.5:
                excess = min(self.fold_to_raise_rate - 0.5, 0.5)  # up to 0.5 above threshold
                raise_threshold_reduction = (excess / 0.5) * 0.08 * ftr_weight  # up to +0.08

        # Preflop fold-equity steal: a very tight opponent (low VPIP) folds most
        # of its hands, so widen our betting/opening range to attack those folds.
        # This is the lever that exploits a tight player whose FTR signal never
        # fires (a tight-aggressive opponent rarely folds *to a raise* once in,
        # but folds the vast majority of hands before entering).
        if self.hands_seen >= 10 and self.vpip < 0.30:
            vpip_dev = (0.30 - self.vpip) / 0.30
            raise_threshold_reduction = max(
                raise_threshold_reduction, vpip_dev * 0.06 * sample_weight
            )

        return call_adj, raise_threshold_reduction

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

        Counts the hand if _hand_open (set by FullAgent each act) or if
        update() recorded opponent actions (_last_hand != -1). Repeated calls
        after a flush are no-ops.
        """
        if not self._hand_open and self._last_hand == -1:
            return
        self.hands_seen += 1
        if self._current_hand_vpip:
            self._hands_vpip += 1
        if self._current_hand_pfr:
            self._hands_pfr += 1
        self._hand_open = False
        self._last_hand = -1
        self._current_hand_vpip = False
        self._current_hand_pfr = False
