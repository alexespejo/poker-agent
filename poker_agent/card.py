"""Card, Suit, Rank, and Deck primitives for Texas Hold'em."""

import random
from enum import Enum


class Rank(Enum):
    """Card ranks from 2 to Ace."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self) -> str:
        symbols = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                   8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        return symbols[self.value]


class Suit(Enum):
    """Card suits."""
    CLUBS = 'c'
    DIAMONDS = 'd'
    HEARTS = 'h'
    SPADES = 's'

    def __str__(self) -> str:
        return self.value


class Card:
    """A single playing card."""

    def __init__(self, rank: Rank, suit: Suit) -> None:
        self.rank = rank
        self.suit = suit

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return f"Card({self.rank}, {self.suit})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))


class Deck:
    """Standard 52-card deck."""

    def __init__(self) -> None:
        self._cards: list[Card] = [
            Card(rank, suit)
            for suit in Suit
            for rank in Rank
        ]

    def shuffle(self) -> None:
        """Shuffle the deck in place."""
        random.shuffle(self._cards)

    def deal(self, n: int) -> list[Card]:
        """Remove and return n cards from the top of the deck."""
        if n > len(self._cards):
            raise ValueError(f"Cannot deal {n} cards; only {len(self._cards)} remain.")
        dealt = self._cards[-n:]
        self._cards = self._cards[:-n]
        return dealt

    def __len__(self) -> int:
        return len(self._cards)

    def remaining(self) -> list[Card]:
        """Return a copy of remaining cards."""
        return list(self._cards)
