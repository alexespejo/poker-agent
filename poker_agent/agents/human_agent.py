"""Human agent — prompts the terminal user for each action."""

from __future__ import annotations

from poker_agent.agents.base import Agent
from poker_agent.agents.random_agent import _legal_actions
from poker_agent.game import GameState

# ── ANSI helpers ───────────────────────────────────────────────────────────────
_R = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"

def _g(s): return f"{_GREEN}{s}{_R}"
def _r(s): return f"{_RED}{s}{_R}"
def _y(s): return f"{_YELLOW}{s}{_R}"
def _c(s): return f"{_CYAN}{s}{_R}"
def _b(s): return f"{_BOLD}{s}{_R}"
def _dim(s): return f"{_DIM}{s}{_R}"

_SUIT_SYM   = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
_SUIT_COLOR = {"c": _GREEN, "d": _RED, "h": _RED, "s": _BLUE}
_RANK_STR   = {2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",
               10:"T",11:"J",12:"Q",13:"K",14:"A"}

W = 64


def _fmt_card(card) -> str:
    rank = _RANK_STR[card.rank.value]
    suit = str(card.suit)
    return f"{_BOLD}{_SUIT_COLOR[suit]}{rank}{_SUIT_SYM[suit]}{_R}"


def _fmt_cards(cards) -> str:
    return "  ".join(_fmt_card(c) for c in cards) if cards else _dim("—")


def _hline(): print("─" * W)
def _dline(): print("═" * W)


class HumanAgent(Agent):
    """Interactive agent that prompts the human for every action."""

    def __init__(self) -> None:
        self._prev_history_len: int = 0
        self._prev_community_len: int = 0

    def reset_hand(self) -> None:
        self._prev_history_len = 0
        self._prev_community_len = 0

    def act(self, state: GameState, player_id: int) -> tuple[str, int]:
        p = player_id
        opp = 1 - p

        # ── Show new community cards (new street) ──────────────────────────
        if len(state.community_cards) > self._prev_community_len:
            new_cards = state.community_cards[self._prev_community_len:]
            print()
            _hline()
            street_label = _b(state.street.upper())
            print(f"  {street_label}: {_fmt_cards(new_cards)}")
            self._prev_community_len = len(state.community_cards)

        # ── Show opponent actions since our last turn ──────────────────────
        new_entries = state.betting_history[self._prev_history_len:]
        opp_actions = [e for e in new_entries if e[0] == opp and e[1] != "blind"]
        for player, action, amount in opp_actions:
            label = _dim("Opponent")
            if action == "fold":
                print(f"  {label} {_r('folds')}")
            elif action == "call":
                print(f"  {label} {_y('calls')}")
            elif action == "check":
                print(f"  {label} {_dim('checks')}")
            elif action == "raise":
                print(f"  {label} {_g(f'raises  (+{amount})')}")
        self._prev_history_len = len(state.betting_history)

        # ── Pot / stack summary ────────────────────────────────────────────
        pot_str   = _y(str(state.pot))
        your_str  = _c(str(state.stacks[p]))
        opp_str   = _magenta(str(state.stacks[opp]))
        print()
        print(f"  Pot {pot_str}  │  Your stack {your_str}  │  Opp stack {opp_str}")

        # ── Call amount info ───────────────────────────────────────────────
        legal = _legal_actions(state, p)
        if state.current_bet > 0:
            print(f"  To call: {_r(str(state.current_bet))}")

        # ── Build action prompt ────────────────────────────────────────────
        parts: list[str] = []
        if "fold"  in legal: parts.append(_r("[f] fold"))
        if "check" in legal: parts.append(_b("[c] check"))
        if "call"  in legal: parts.append(_y(f"[c] call {state.current_bet}"))
        if "raise" in legal: parts.append(_g("[r] raise"))

        print(f"  {' │ '.join(parts)}")

        while True:
            try:
                raw = input(f"\n  {_b('Your action')}: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                raise

            if raw in ("f", "fold") and "fold" in legal:
                return "fold", 0

            if raw in ("c", "call", "check"):
                if "call" in legal:
                    return "call", 0
                if "check" in legal:
                    return "check", 0

            if raw in ("r", "raise") and "raise" in legal:
                return self._prompt_raise(state, p)

            # Also accept raw number as raise amount
            if raw.isdigit() and "raise" in legal:
                amount = int(raw)
                if amount >= state.min_raise:
                    return "raise", min(amount, state.stacks[p] - state.current_bet)

            print(f"  {_r('Invalid.')}  Options: {', '.join(legal)}")

    def _prompt_raise(self, state: GameState, p: int) -> tuple[str, int]:
        min_r = state.min_raise
        max_r = state.stacks[p] - state.current_bet
        while True:
            try:
                raw = input(
                    f"  Raise by how much?  "
                    f"(min {_g(str(min_r))}, max {_g(str(max_r))}, pot {_y(str(state.pot))}): "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                raise
            if raw.isdigit():
                amount = int(raw)
                if min_r <= amount <= max_r:
                    return "raise", amount
                print(f"  {_r(f'Must be between {min_r} and {max_r}.')}")
            else:
                print(f"  {_r('Enter a whole number.')}")


def _magenta(s: str) -> str:
    return f"{_MAGENTA}{s}{_R}"
