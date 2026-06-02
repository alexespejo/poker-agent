"""Shared utilities for agent implementations."""

from poker_agent.game import GameState


def _legal_actions(state: GameState, player_id: int) -> list[str]:
    """Derive legal actions directly from state without a full PokerGame instance."""
    p = player_id
    actions: list[str] = []
    stack = state.stacks[p]

    if state.current_bet > 0:
        actions.append("fold")
        actions.append("call")  # may be all-in but still legal
    else:
        actions.append("check")

    effective = stack - state.current_bet
    if effective >= state.min_raise:
        actions.append("raise")

    return actions
