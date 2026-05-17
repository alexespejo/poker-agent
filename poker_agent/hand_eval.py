"""Hand evaluator for Texas Hold'em — no external poker libraries."""

from itertools import combinations

from poker_agent.card import Card, Rank


# Hand category constants (higher = better)
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8

_BASE = 15       # ranks run 2–14; fits in one base-15 digit
_TIER = _BASE ** 5  # each hand category spans this many values (> any 5-rank tiebreaker)


def _score_five(cards: list[Card]) -> tuple[int, str]:
    """Score exactly 5 cards; return (integer_score, description)."""
    ranks = sorted([c.rank.value for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    is_flush = len(set(suits)) == 1

    # Detect straight (including wheel A-2-3-4-5)
    is_straight = False
    straight_high = 0
    if ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5:
        is_straight = True
        straight_high = ranks[0]
    elif ranks == [14, 5, 4, 3, 2]:  # wheel
        is_straight = True
        straight_high = 5

    # Count occurrences of each rank
    from collections import Counter
    counts = Counter(ranks)
    freq = sorted(counts.values(), reverse=True)  # e.g. [4,1] for quads

    def pack(category: int, *tiebreakers: int) -> int:
        """Encode hand as category * _TIER + tiebreaker, so category always dominates."""
        tie = 0
        for v in tiebreakers:
            tie = tie * _BASE + v
        return category * _TIER + tie

    if is_straight and is_flush:
        desc = "Royal Flush" if straight_high == 14 else "Straight Flush"
        return pack(STRAIGHT_FLUSH, straight_high), desc

    if freq[0] == 4:
        quad_rank = max(r for r, c in counts.items() if c == 4)
        kick = max(r for r, c in counts.items() if c == 1)
        return pack(FOUR_OF_A_KIND, quad_rank, kick), "Four of a Kind"

    if freq[0] == 3 and freq[1] == 2:
        trip_rank = max(r for r, c in counts.items() if c == 3)
        pair_rank = max(r for r, c in counts.items() if c == 2)
        return pack(FULL_HOUSE, trip_rank, pair_rank), "Full House"

    if is_flush:
        return pack(FLUSH, *ranks), "Flush"

    if is_straight:
        return pack(STRAIGHT, straight_high), "Straight"

    if freq[0] == 3:
        trip_rank = max(r for r, c in counts.items() if c == 3)
        kicks = sorted([r for r, c in counts.items() if c == 1], reverse=True)
        return pack(THREE_OF_A_KIND, trip_rank, *kicks), "Three of a Kind"

    if freq[0] == 2 and freq[1] == 2:
        pairs = sorted([r for r, c in counts.items() if c == 2], reverse=True)
        kick = max(r for r, c in counts.items() if c == 1)
        return pack(TWO_PAIR, *pairs, kick), "Two Pair"

    if freq[0] == 2:
        pair_rank = max(r for r, c in counts.items() if c == 2)
        kicks = sorted([r for r, c in counts.items() if c == 1], reverse=True)
        return pack(ONE_PAIR, pair_rank, *kicks), "One Pair"

    return pack(HIGH_CARD, *ranks), "High Card"


def evaluate_hand(cards: list[Card]) -> tuple[int, str]:
    """Evaluate the best 5-card hand from 5–7 cards.

    Returns (rank_score, description) where higher rank_score is a better hand.
    Hands can be compared directly with >.
    """
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")
    if len(cards) == 5:
        return _score_five(cards)
    best_score = -1
    best_desc = ""
    for combo in combinations(cards, 5):
        score, desc = _score_five(list(combo))
        if score > best_score:
            best_score = score
            best_desc = desc
    return best_score, best_desc


# ---------------------------------------------------------------------------
# Unit tests — run as assertions when this module is imported or executed
# ---------------------------------------------------------------------------

def _make_cards(*specs: str) -> list[Card]:
    """Build a list of Cards from strings like 'Ah', '2c', 'Td'."""
    rank_map = {'2': Rank.TWO, '3': Rank.THREE, '4': Rank.FOUR, '5': Rank.FIVE,
                '6': Rank.SIX, '7': Rank.SEVEN, '8': Rank.EIGHT, '9': Rank.NINE,
                'T': Rank.TEN, 'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING,
                'A': Rank.ACE}
    from poker_agent.card import Suit
    suit_map = {'c': Suit.CLUBS, 'd': Suit.DIAMONDS, 'h': Suit.HEARTS, 's': Suit.SPADES}
    result = []
    for spec in specs:
        r, s = spec[:-1], spec[-1]
        result.append(Card(rank_map[r], suit_map[s]))
    return result


def run_unit_tests() -> list[tuple[str, bool]]:
    """Return list of (description, passed) for each assertion."""
    results: list[tuple[str, bool]] = []

    def check(desc: str, condition: bool) -> None:
        results.append((desc, condition))

    # Pair beats high card
    pair = evaluate_hand(_make_cards('Ah', 'As', '2c', '3d', '7h'))[0]
    high = evaluate_hand(_make_cards('Ah', 'Kh', '2c', '3d', '7s'))[0]
    check("Pair beats High Card", pair > high)

    # AA beats KK
    aa = evaluate_hand(_make_cards('Ah', 'As', '2c', '3d', '7h'))[0]
    kk = evaluate_hand(_make_cards('Kh', 'Ks', '2c', '3d', '7h'))[0]
    check("AA beats KK", aa > kk)

    # Two pair beats one pair
    two_pair = evaluate_hand(_make_cards('Ah', 'As', 'Kc', 'Kd', '7h'))[0]
    check("Two Pair beats One Pair", two_pair > pair)

    # Three of a kind beats two pair
    trips = evaluate_hand(_make_cards('Ah', 'As', 'Ac', '3d', '7h'))[0]
    check("Three of a Kind beats Two Pair", trips > two_pair)

    # Straight beats trips
    straight = evaluate_hand(_make_cards('5h', '6s', '7c', '8d', '9h'))[0]
    check("Straight beats Three of a Kind", straight > trips)

    # Flush beats straight
    flush = evaluate_hand(_make_cards('2h', '5h', '7h', 'Th', 'Kh'))[0]
    check("Flush beats Straight", flush > straight)

    # Full house beats flush
    full_house = evaluate_hand(_make_cards('Ah', 'As', 'Ac', 'Kd', 'Ks'))[0]
    check("Full House beats Flush", full_house > flush)

    # Quads beat full house
    quads = evaluate_hand(_make_cards('Ah', 'As', 'Ac', 'Ad', 'Ks'))[0]
    check("Four of a Kind beats Full House", quads > full_house)

    # Straight flush beats quads
    sf = evaluate_hand(_make_cards('5h', '6h', '7h', '8h', '9h'))[0]
    check("Straight Flush beats Four of a Kind", sf > quads)

    # Royal flush beats regular straight flush
    royal = evaluate_hand(_make_cards('Th', 'Jh', 'Qh', 'Kh', 'Ah'))[0]
    check("Royal Flush beats Straight Flush", royal > sf)

    # Wheel straight (A-2-3-4-5) is a straight
    wheel_score, wheel_desc = evaluate_hand(_make_cards('Ah', '2s', '3c', '4d', '5h'))
    check("Wheel A-2-3-4-5 is a Straight", wheel_desc == "Straight")

    # Wheel beats trips
    check("Wheel Straight beats Three of a Kind", wheel_score > trips)

    # Higher straight beats lower straight
    high_straight = evaluate_hand(_make_cards('6h', '7s', '8c', '9d', 'Th'))[0]
    low_straight = evaluate_hand(_make_cards('2h', '3s', '4c', '5d', '6h'))[0]
    check("Higher Straight beats Lower Straight", high_straight > low_straight)

    # 7-card evaluation picks best 5
    seven_cards = _make_cards('Ah', 'As', 'Ac', 'Ad', 'Ks', '2c', '3d')
    score_7, desc_7 = evaluate_hand(seven_cards)
    check("7-card eval identifies Four of a Kind", desc_7 == "Four of a Kind")
    check("7-card score matches equivalent 5-card quads", score_7 == quads)

    # Split pot scenario: same hand strength
    hand_a = evaluate_hand(_make_cards('2h', '2s', '3c', '4d', '5h'))[0]
    hand_b = evaluate_hand(_make_cards('2c', '2d', '3h', '4s', '5c'))[0]
    check("Identical hand ranks are equal (split pot)", hand_a == hand_b)

    return results


if __name__ == "__main__":
    for desc, passed in run_unit_tests():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")
