"""Run N-hand simulations and collect statistics."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from poker_agent.agents.base import Agent
from poker_agent.game import PokerGame


def _update_progress(current: int, total: int, running_mbb: float) -> None:
    """Redraw a single-line progress bar (ASCII width so terminals don't wrap)."""
    pct = current / total * 100
    bar_w = 30
    filled = int(pct / 100 * bar_w)
    bar = "#" * filled + "-" * (bar_w - filled)
    color = "\033[32m" if running_mbb >= 0 else "\033[31m"
    reset = "\033[0m"
    line = (
        f"  [{bar}] {pct:5.1f}%  hand {current:>{len(str(total))}}/{total}"
        f"  mbb/hand: {color}{running_mbb:+.1f}{reset}"
    )
    if sys.stdout.isatty():
        sys.stdout.write(f"\r\033[K{line}")
        sys.stdout.flush()
    elif current == total:
        print(line)


def _finish_progress() -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\n")
        sys.stdout.flush()


@dataclass
class SimResults:
    """Results from a multi-hand simulation."""
    mbb_per_hand_agent0: float
    mbb_per_hand_agent1: float
    hands_played: int
    action_counts_agent0: dict[str, int]
    action_counts_agent1: dict[str, int]
    chip_history: list[int]   # agent0 net chips after each hand (cumulative)
    errors: int = 0
    win_ehs_by_street: dict[str, list[float]] = field(default_factory=dict)
    lose_ehs_by_street: dict[str, list[float]] = field(default_factory=dict)


def run_simulation(
    agent0: Agent,
    agent1: Agent,
    n_hands: int = 10_000,
    stack_size: int = 1000,
    big_blind: int = 10,
    verbose: bool = False,
    checkpoint_callback=None,
    show_progress: bool = True,
    collect_hero_ehs: bool = False,
) -> SimResults:
    """Simulate n_hands of heads-up poker between agent0 and agent1.

    Stacks reset each hand (non-tournament format).
    Dealer alternates each hand.
    mbb/hand = (net_chips / n_hands / big_blind) * 1000
    checkpoint_callback(hand_number, game, results_so_far) called if provided.

    If collect_hero_ehs is True, records agent0's raw EHS (from last_decision)
    per hero action, bucketed into win_ehs_by_street / lose_ehs_by_street.
    """
    game = PokerGame(stack_size=stack_size, big_blind=big_blind)
    agents = [agent0, agent1]

    net_chips = [0, 0]
    action_counts = [
        {"fold": 0, "call": 0, "check": 0, "raise": 0},
        {"fold": 0, "call": 0, "check": 0, "raise": 0},
    ]
    chip_history: list[int] = []
    errors = 0
    win_ehs: dict[str, list[float]] = {}
    lose_ehs: dict[str, list[float]] = {}
    _progress_interval = max(1, n_hands // 100) if show_progress else n_hands + 1

    for hand_num in range(n_hands):
        dealer = hand_num % 2  # alternate dealer each hand
        hand_records: list[tuple[str, float]] = []
        try:
            state = game.reset(dealer=dealer)
            done = False

            while not done:
                p = state.current_player
                action, amount = agents[p].act(state, p)

                if collect_hero_ehs and p == 0:
                    last_decision = getattr(agent0, "last_decision", None)
                    if isinstance(last_decision, dict) and "ehs" in last_decision:
                        hand_records.append((state.street, last_decision["ehs"]))

                # Clamp action to legal set (safety net)
                legal = game.legal_actions(state)
                if action not in legal:
                    action = legal[0]
                    amount = 0

                if action in action_counts[p]:
                    action_counts[p][action] += 1

                state, rewards, done = game.step(action, amount)

            if collect_hero_ehs and rewards[0] != 0:
                target = win_ehs if rewards[0] > 0 else lose_ehs
                for street, ehs in hand_records:
                    target.setdefault(street, []).append(ehs)

            net_chips[0] += rewards[0]
            net_chips[1] += rewards[1]
            chip_history.append(net_chips[0])

            current_hand = hand_num + 1
            if show_progress and (
                current_hand % _progress_interval == 0 or current_hand == n_hands
            ):
                running_mbb = (net_chips[0] / current_hand / big_blind) * 1000
                _update_progress(current_hand, n_hands, running_mbb)

            if verbose and hand_num < 5:
                print(f"  Hand {hand_num + 1}: rewards={rewards}, "
                      f"net={net_chips}, history={state.betting_history[-4:]}")

            if checkpoint_callback is not None:
                checkpoint_callback(hand_num + 1, game, net_chips[:])

        except Exception as e:
            errors += 1
            if verbose:
                print(f"  ERROR in hand {hand_num + 1}: {e}")
            chip_history.append(net_chips[0])

    if show_progress:
        _finish_progress()

    def mbb(chips: int) -> float:
        return (chips / n_hands / big_blind) * 1000

    return SimResults(
        mbb_per_hand_agent0=mbb(net_chips[0]),
        mbb_per_hand_agent1=mbb(net_chips[1]),
        hands_played=n_hands - errors,
        action_counts_agent0=action_counts[0],
        action_counts_agent1=action_counts[1],
        chip_history=chip_history,
        errors=errors,
        win_ehs_by_street=win_ehs,
        lose_ehs_by_street=lose_ehs,
    )
