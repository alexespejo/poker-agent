"""Opponent-learning schedule for FullAgent."""

from __future__ import annotations

from enum import Enum


class OpponentLearningMode(str, Enum):
    """When FullAgent updates and applies the opponent model."""

    LIVE = "live"
    """Learn and apply adjustments from hand 1 (legacy behavior)."""

    WARMUP_THEN_ADAPT = "warmup_then_adapt"
    """Warm-up: observe only (neutral adjustments). Scored: learn + apply."""

    WARMUP_THEN_FROZEN = "warmup_then_frozen"
    """Warm-up: observe. Scored: apply snapshot from end of warm-up; no further updates."""


def parse_learning_mode(value: str) -> OpponentLearningMode:
    """Parse CLI / config string into OpponentLearningMode."""
    try:
        return OpponentLearningMode(value)
    except ValueError as e:
        valid = ", ".join(m.value for m in OpponentLearningMode)
        raise ValueError(f"Unknown full-learning mode {value!r}. Choose: {valid}") from e
