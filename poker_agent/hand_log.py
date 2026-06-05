"""Per-hand text log for heads-up simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, TextIO

from poker_agent.card import Card
from poker_agent.game import GameState
from poker_agent.hand_eval import evaluate_hand

_SUIT_SYM = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
_RANK_STR = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}


def format_card(card: Card) -> str:
    """Plain-text card (e.g. Ah, Td)."""
    return f"{_RANK_STR[card.rank.value]}{_SUIT_SYM[str(card.suit)]}"


def format_cards(cards: list[Card]) -> str:
    if not cards:
        return "—"
    return "  ".join(format_card(c) for c in cards)


def _mbb(net_chips: int, hands: int, big_blind: int) -> float:
    if hands <= 0:
        return 0.0
    return (net_chips / hands / big_blind) * 1000


def action_from_betting_history(state: GameState) -> tuple[int, str, int]:
    """Return (player, action, chips) for the action most recently applied in step()."""
    player, action, amount = state.betting_history[-1]
    return player, action, amount


def _format_action(player: int, action: str, amount: int, names: tuple[str, str]) -> str:
    label = names[player]
    if action == "blind":
        return f"    {label}: blind {amount}"
    if action == "fold":
        return f"    {label}: FOLD"
    if action == "check":
        return f"    {label}: check"
    if action == "call":
        return f"    {label}: call {amount}"
    if action == "raise":
        return f"    {label}: raise {amount}"
    return f"    {label}: {action} {amount}"


@dataclass
class _StreetBlock:
    street: str
    board: list[Card]
    lines: list[str] = field(default_factory=list)


@dataclass
class _HandBuffer:
    hand_num: int
    dealer: int
    hand_id: int
    hole_cards: list[list[Card]]
    streets: list[_StreetBlock] = field(default_factory=list)
    _current: _StreetBlock | None = None

    def ensure_street(self, street: str, board: list[Card]) -> None:
        if self._current is not None and self._current.street == street:
            return
        self._current = _StreetBlock(street=street, board=list(board))
        self.streets.append(self._current)

    def add_action(self, street: str, board: list[Card], line: str) -> None:
        self.ensure_street(street, board)
        assert self._current is not None
        self._current.lines.append(line)


class HandLog:
    """Append formatted per-hand records to a text file."""

    def __init__(
        self,
        path: str | Path,
        agent_names: tuple[str, str],
        *,
        stack_size: int,
        big_blind: int,
        n_hands: int,
        pairing_label: str | None = None,
        warmup_hands: int = 0,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self._path.open("w", encoding="utf-8")
        self._names = agent_names
        self._stack_size = stack_size
        self._big_blind = big_blind
        self._n_hands = n_hands
        self._warmup_hands = warmup_hands
        self._pairing = pairing_label or f"{agent_names[0]} vs {agent_names[1]}"
        self._buffer: _HandBuffer | None = None
        self._write_header()

    @property
    def path(self) -> Path:
        return self._path

    def _write(self, text: str = "") -> None:
        self._file.write(text + "\n")

    def _write_header(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        w = 78
        self._write("=" * w)
        self._write(f"  HAND LOG — {self._pairing}")
        self._write("=" * w)
        self._write(f"  Agent 0: {self._names[0]}")
        self._write(f"  Agent 1: {self._names[1]}")
        if self._warmup_hands > 0:
            self._write(
                f"  Scored hands: {self._n_hands:,}  |  Warm-up (not logged): "
                f"{self._warmup_hands:,}  |  Stack/hand: {self._stack_size:,}  |  BB: {self._big_blind}"
            )
        else:
            self._write(
                f"  Hands: {self._n_hands:,}  |  Stack/hand: {self._stack_size:,}  |  BB: {self._big_blind}"
            )
        self._write(f"  Started: {ts}")
        self._write(
            "  (Stacks reset each hand; running net + mbb/hand are cumulative over scored hands.)"
        )
        self._write("=" * w)
        self._write()

    def begin_hand(self, hand_num: int, dealer: int, state: GameState) -> None:
        self._buffer = _HandBuffer(
            hand_num=hand_num,
            dealer=dealer,
            hand_id=state.hand_id,
            hole_cards=[list(state.hole_cards[0]), list(state.hole_cards[1])],
        )
        self._buffer.ensure_street(state.street, state.community_cards)
        for player, action, amount in state.betting_history:
            if action == "blind":
                line = _format_action(player, action, amount, self._names)
                self._buffer.add_action(state.street, state.community_cards, line)

    def record_action(
        self,
        street: str,
        player: int,
        action: str,
        amount: int,
        community_cards: list[Card],
    ) -> None:
        """Record one action. *amount* is chips actually moved (from betting_history)."""
        if self._buffer is None:
            return
        line = _format_action(player, action, amount, self._names)
        self._buffer.add_action(street, community_cards, line)

    def end_hand(
        self,
        state: GameState,
        rewards: list[int],
        net_chips: list[int],
    ) -> None:
        if self._buffer is None:
            return
        buf = self._buffer
        self._buffer = None

        hand_num = buf.hand_num
        n0, n1 = self._names
        dealer_name = self._names[buf.dealer]
        sep = "-" * 78

        self._write(sep)
        self._write(
            f"Hand {hand_num}/{self._n_hands}  (id={buf.hand_id})  "
            f"dealer={dealer_name}"
        )
        self._write(sep)
        self._write(f"  {n0}: [{format_cards(buf.hole_cards[0])}]")
        self._write(f"  {n1}: [{format_cards(buf.hole_cards[1])}]")
        self._write(f"  Pot at end: {state.pot}")

        for block in buf.streets:
            self._write()
            board_str = format_cards(block.board) if block.board else "—"
            self._write(f"  {block.street.upper()}  board: {board_str}")
            for line in block.lines:
                self._write(line)

        self._write()
        self._write_result(state, rewards)

        r0, r1 = rewards[0], rewards[1]
        self._write()
        self._write(f"  Hand delta:  {n0} {r0:+d}   {n1} {r1:+d}")
        hands_done = hand_num
        self._write(
            f"  Running net: {n0} {net_chips[0]:+d} ({_mbb(net_chips[0], hands_done, self._big_blind):+.2f} mbb/hand)"
        )
        self._write(
            f"               {n1} {net_chips[1]:+d} ({_mbb(net_chips[1], hands_done, self._big_blind):+.2f} mbb/hand)"
        )
        self._write()

        if hand_num % 50 == 0:
            self._file.flush()

    def _write_result(self, state: GameState, rewards: list[int]) -> None:
        n0, n1 = self._names
        board = format_cards(state.community_cards) if state.community_cards else "—"

        folds = [p for p, act, _ in state.betting_history if act == "fold"]
        if folds:
            folder = self._names[folds[0]]
            winner = n1 if folds[0] == 0 else n0
            self._write(f"  OUTCOME: {folder} folded — {winner} wins pot ({state.pot})")
        elif rewards[0] == 0 and rewards[1] == 0:
            self._write("  OUTCOME: split pot (tie)")
        elif rewards[0] > 0 and rewards[1] > 0:
            self._write(f"  OUTCOME: split pot — board: {board}")
        elif rewards[0] > 0:
            self._write(f"  OUTCOME: {n0} wins showdown (+{rewards[0]}) — board: {board}")
        elif rewards[1] > 0:
            self._write(f"  OUTCOME: {n1} wins showdown (+{rewards[1]}) — board: {board}")
        else:
            self._write(f"  OUTCOME: — board: {board}")

        if state.community_cards and "fold" not in {a for _, a, _ in state.betting_history}:
            comm = state.community_cards
            _, desc0 = evaluate_hand(state.hole_cards[0] + comm)
            _, desc1 = evaluate_hand(state.hole_cards[1] + comm)
            self._write(f"  Showdown: {n0} = {desc0}   |   {n1} = {desc1}")

    def write_footer(self, net_chips: list[int], errors: int) -> None:
        w = 78
        self._write("=" * w)
        self._write("  SESSION SUMMARY")
        self._write("=" * w)
        hands = self._n_hands - errors
        for i, name in enumerate(self._names):
            self._write(
                f"  {name}: net {net_chips[i]:+d}  "
                f"({_mbb(net_chips[i], max(hands, 1), self._big_blind):+.2f} mbb/hand over {hands:,} hands)"
            )
        if errors:
            self._write(f"  Errors: {errors}")
        self._write("=" * w)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> HandLog:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
