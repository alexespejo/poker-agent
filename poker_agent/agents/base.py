"""Abstract base class for all poker agents."""

from abc import ABC, abstractmethod

from poker_agent.game import GameState


class Agent(ABC):
    """All agents must implement act(); optionally override reset()."""

    @abstractmethod
    def act(self, game_state: GameState, player_id: int) -> tuple[str, int]:
        """Choose an action given the current game state.

        Returns (action, amount) where action is one of:
          'fold', 'call', 'check', 'raise'
        and amount is the raise size if action=='raise', else 0.
        """
        ...

    def reset(self) -> None:
        """Called at the start of a new session; override to clear per-session state."""
        pass
