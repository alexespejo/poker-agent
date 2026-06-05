"""Opponent modeling — tracks betting statistics to adjust EHS estimates."""

from __future__ import annotations

from typing import Callable

from poker_agent.monte_carlo import make_range_filter, make_range_filter_band

# Logistic ramp sharpness for range filters (Bug 4). ~8–12 keeps boundaries soft.
_RANGE_FILTER_K = 10.0


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

    @property
    def confidence(self) -> float:
        """Confidence in the learned profile, 0–1 (saturates at 20 hands)."""
        return min(1.0, self.hands_seen / 20)

    # ------------------------------------------------------------------
    # Range estimation (used to build range filters for conditioned EHS)
    # ------------------------------------------------------------------

    def get_raise_range_fraction(self) -> float:
        """Fraction of starting hands the opponent is assumed to raise.  [Bug 2]

        The previous design conditioned on only a fixed-tight top slice (~0.25),
        which models a nit even against a loose, wide-raising opponent and biases
        our conditioned equity badly downward.  Instead we ground the estimate in
        observed looseness:

          • Blend a *loosened* uniform prior (0.45, up from 0.25) toward the
            measured PFR as confidence grows.
          • Floor the result at the VPIP-implied width (0.5 * VPIP) so a loose
            opponent is never modeled as tight.
          • Refuse to collapse below ~0.30 unless we are confident
            (confidence >= 0.5) AND measured PFR is genuinely low (< 0.30); this
            prevents premature tightening on small samples.
        """
        conf = self.confidence
        prior = 0.45  # loosened uniform prior (was effectively 0.25)
        blended_pfr = (1.0 - conf) * prior + conf * self.pfr
        frac = max(blended_pfr, 0.5 * self.vpip)  # never model a loose opp as tight

        if not (conf >= 0.5 and self.pfr < 0.30):
            frac = max(frac, 0.30)

        return max(0.05, min(1.0, frac))

    def get_estimated_vpip_width(self) -> float:
        """Assumed fraction of hands the opponent enters the pot with (0–1).

        Blends a loose prior (0.55) toward measured VPIP as confidence grows.
        Used as the wide edge of the cumulative calling range.
        """
        conf = self.confidence
        prior = 0.55
        return max(0.10, min(1.0, (1.0 - conf) * prior + conf * self.vpip))

    def make_range_filter_for_action(
        self, action: str
    ) -> Callable[[float], float] | None:
        """Return a soft percentile->weight filter for an opponent action.

        "raise" → top ``get_raise_range_fraction()`` of hands (Bug 2).
        "call"  → cumulative calling range: everything inside the VPIP range that
                  was *not* raised, i.e. percentiles in [1 - VPIP, 1 - raise_frac),
                  softly weighted and guaranteed to span at least ~25% of hands so
                  we never estimate equity against a near-empty sliver (Bug 3).
        """
        raise_frac = self.get_raise_range_fraction()

        if action == "raise":
            return make_range_filter(raise_frac, k=_RANGE_FILTER_K)

        if action == "call":
            vpip = self.get_estimated_vpip_width()
            high = 1.0 - raise_frac          # below the raising range
            low = 1.0 - vpip                 # above the fold range
            # [Bug 3] Guarantee a substantial band (>= 25% of hands), never a sliver.
            if high - low < 0.25:
                low = high - 0.25
            low = max(0.0, low)
            high = min(1.0, max(high, low + 0.25))
            return make_range_filter_band(low, high, k=_RANGE_FILTER_K)

        return None

    def summary(self) -> str:
        """One-line snapshot of the learned profile and derived range estimates."""
        return (
            f"hands={self.hands_seen} VPIP={self.vpip:.2f} PFR={self.pfr:.2f} "
            f"AF={self.aggression_factor:.2f} FTR={self.fold_to_raise_rate:.2f} "
            f"conf={self.confidence:.2f} raise_range_frac={self.get_raise_range_fraction():.2f}"
        )

    # ------------------------------------------------------------------
    # EHS adjustment
    # ------------------------------------------------------------------

    def get_range_multiplier(self) -> tuple[float, float]:
        """Return (call_adj, raise_threshold_reduction).

        call_adj — additive EHS adjustment applied when deciding whether to
            call or fold the opponent's bet.
              Negative = opponent plays tight/strong → need a better hand to call.
              Positive = opponent is loose-aggressive/bluffy → can call lighter.
            Clamped to [-0.12, +0.10].

        raise_threshold_reduction — amount to subtract from raise_threshold when
            deciding whether to initiate a bet/raise.  Positive means lower the
            raise bar (bluff more) because the opponent folds to raises often.
            Clamped to [0, +0.10].

        Both are scaled by min(1.0, hands_seen / 20) to discount sparse data.
        The FTR signal uses its own weight: min(1.0, faced_raises / 10).

        Design notes
        ────────────
        • AF adjustment is gated on VPIP > 0.50.  A player with high AF and
          LOW VPIP is tight-aggressive — their bets are value-heavy, not bluffs.
          Treating them as bluffy (and calling more) is the source of the previous
          regression against EHSAgent.  We instead tighten our call standard
          further when we see this tight-aggressive profile.
        • FTR drives raise_threshold_reduction, not call_adj.  Folding to our
          raises means we should steal more pots, not call more of their bets.
        """
        sample_weight = min(1.0, self.hands_seen / 20)

        # ── Defensive component (call_adj) ──────────────────────────────────

        tight_adj = 0.0
        aggro_adj = 0.0

        # Tight player: VPIP < 0.35 → their range is stronger → need better hand to call
        if self.hands_seen > 0 and self.vpip < 0.35:
            deviation = 0.35 - self.vpip
            tight_adj = -(deviation / 0.35) * 0.10   # up to -0.10

        # Loose-aggressive / bluffy: high AF AND high VPIP
        # Guard: only apply when VPIP > 0.50.  Without the VPIP guard, a tight
        # polarised player (folds weak hands, raises strong ones → low VPIP, high
        # AF) would be mislabelled as bluffy, causing us to call more into strength.
        if self.hands_seen > 0 and self.aggression_factor > 2.0 and self.vpip > 0.50:
            deviation = min(self.aggression_factor - 2.0, 8.0)
            aggro_adj = (deviation / 8.0) * 0.07     # up to +0.07

        # Tight-aggressive: low VPIP AND high AF → their bets are even more value-heavy
        if self.hands_seen > 0 and self.aggression_factor > 2.0 and self.vpip < 0.40:
            ta_bonus = min((self.aggression_factor - 2.0) / 8.0, 1.0) * 0.05
            tight_adj -= ta_bonus                     # extra tightening (max -0.05)

        call_adj = (tight_adj + aggro_adj) * sample_weight
        call_adj = max(-0.12, min(0.10, call_adj))

        # ── Offensive component (raise_threshold_reduction) ──────────────────

        raise_threshold_reduction = 0.0
        if self._faced_raises >= 5:
            ftr_weight = min(1.0, self._faced_raises / 10)
            if self.fold_to_raise_rate > 0.5:
                excess = min(self.fold_to_raise_rate - 0.5, 0.5)  # up to 0.5 above threshold
                raise_threshold_reduction = (excess / 0.5) * 0.08 * ftr_weight  # up to +0.08

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
