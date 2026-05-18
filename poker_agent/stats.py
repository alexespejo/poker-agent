"""Session logging and statistics for poker simulations."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _ActionRecord:
    hand_number: int
    street: str
    player: int
    action: str
    amount: int
    ehs: float | None
    pot: int
    stack0: int
    stack1: int


class SessionLogger:
    """Logs every action and provides summary statistics."""

    def __init__(self) -> None:
        self._records: list[_ActionRecord] = []
        self._net_chips: list[int] = [0, 0]
        self._hands: int = 0

    def log_action(
        self,
        hand_number: int,
        street: str,
        player: int,
        action: str,
        amount: int,
        ehs: float | None,
        pot: int,
        stacks: list[int],
    ) -> None:
        """Record a single action."""
        self._records.append(_ActionRecord(
            hand_number=hand_number,
            street=street,
            player=player,
            action=action,
            amount=amount,
            ehs=ehs,
            pot=pot,
            stack0=stacks[0],
            stack1=stacks[1],
        ))

    def record_hand_result(self, rewards: list[int]) -> None:
        """Update net chip totals at the end of a hand."""
        self._net_chips[0] += rewards[0]
        self._net_chips[1] += rewards[1]
        self._hands += 1

    def save_log(self, filepath: str) -> None:
        """Write all action records to a CSV file."""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "hand_number", "street", "player", "action",
                "amount", "ehs", "pot", "stack0", "stack1",
            ])
            for r in self._records:
                writer.writerow([
                    r.hand_number, r.street, r.player, r.action,
                    r.amount, "" if r.ehs is None else f"{r.ehs:.4f}",
                    r.pot, r.stack0, r.stack1,
                ])

    def print_summary(self, big_blind: int = 10) -> None:
        """Print action distribution, average EHS by street, and mbb/hand."""
        if not self._records:
            print("  (no records)")
            return

        # Action distribution per player
        action_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in self._records:
            action_counts[r.player][r.action] += 1

        for p in sorted(action_counts):
            counts = action_counts[p]
            total = sum(counts.values()) or 1
            print(f"\n  Player {p} action distribution:")
            for act in ("fold", "call", "check", "raise"):
                cnt = counts.get(act, 0)
                print(f"    {act:8s}: {cnt:6,}  ({cnt/total*100:.1f}%)")

        # Average EHS by street
        ehs_by_street: dict[str, list[float]] = defaultdict(list)
        for r in self._records:
            if r.ehs is not None:
                ehs_by_street[r.street].append(r.ehs)

        if ehs_by_street:
            print("\n  Average EHS by street:")
            for street in ("preflop", "flop", "turn", "river"):
                vals = ehs_by_street.get(street, [])
                if vals:
                    print(f"    {street:8s}: {sum(vals)/len(vals):.3f}  (n={len(vals)})")

        # mbb/hand
        if self._hands > 0:
            for p in range(2):
                mbb = (self._net_chips[p] / self._hands / big_blind) * 1000
                print(f"\n  Player {p} mbb/hand: {mbb:+.1f}")

    def pot_odds_violations(self, player_id: int) -> int:
        """Count hands where player called with EHS < pot_odds (should be 0)."""
        violations = 0
        # Group records by hand
        by_hand: dict[int, list[_ActionRecord]] = defaultdict(list)
        for r in self._records:
            by_hand[r.hand_number].append(r)

        for records in by_hand.values():
            for r in records:
                if r.player != player_id or r.action != "call" or r.ehs is None:
                    continue
                pot_odds = r.amount / (r.pot + r.amount) if (r.pot + r.amount) > 0 else 0.0
                if r.ehs < pot_odds:
                    violations += 1
        return violations
