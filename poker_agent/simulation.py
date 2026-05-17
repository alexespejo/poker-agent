"""Run N-hand simulations and collect statistics."""

from __future__ import annotations

from dataclasses import dataclass, field

from poker_agent.agents.base import Agent
from poker_agent.game import PokerGame


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


def run_simulation(
    agent0: Agent,
    agent1: Agent,
    n_hands: int = 10_000,
    stack_size: int = 1000,
    big_blind: int = 10,
    verbose: bool = False,
    checkpoint_callback=None,
) -> SimResults:
    """Simulate n_hands of heads-up poker between agent0 and agent1.

    Stacks reset each hand (non-tournament format).
    Dealer alternates each hand.
    mbb/hand = (net_chips / n_hands / big_blind) * 1000
    checkpoint_callback(hand_number, game, results_so_far) called if provided.
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

    for hand_num in range(n_hands):
        dealer = hand_num % 2  # alternate dealer each hand
        try:
            state = game.reset(dealer=dealer)
            done = False

            while not done:
                p = state.current_player
                action, amount = agents[p].act(state, p)

                # Clamp action to legal set (safety net)
                legal = game.legal_actions(state)
                if action not in legal:
                    action = legal[0]
                    amount = 0

                if action in action_counts[p]:
                    action_counts[p][action] += 1

                state, rewards, done = game.step(action, amount)

            net_chips[0] += rewards[0]
            net_chips[1] += rewards[1]
            chip_history.append(net_chips[0])

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
    )
