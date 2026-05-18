# Milestone 1 — Environment & Agent Setup

## Goal
Build a fully functional heads-up Texas Hold'em simulator from scratch, along with two baseline agents (Random and Rule-Based). No external poker libraries — everything is implemented by hand to demonstrate understanding of the domain.

---

## What Was Built

### `poker_agent/card.py` — Card Primitives

Defines the fundamental building blocks of the card game:

- **`Rank` enum** — the 13 card ranks from Two (value=2) through Ace (value=14). Using integer values lets ranks be compared and sorted directly.
- **`Suit` enum** — the four suits (clubs, diamonds, hearts, spades), each with a single-character abbreviation (`c`, `d`, `h`, `s`).
- **`Card` class** — pairs a rank and suit. Supports string representation (`"Ah"`, `"2c"`), equality comparison, and hashing so cards can be stored in sets (useful for the Monte Carlo estimator later).
- **`Deck` class** — a standard 52-card deck. `shuffle()` randomizes order in place using Fisher-Yates; `deal(n)` pops and returns `n` cards from the top, raising an error if the deck runs short.

---

### `poker_agent/hand_eval.py` — Hand Evaluator

The hand evaluator ranks any 5–7 card hand and returns a single integer score, so two hands can be compared with `>` directly.

#### How scoring works

Hands fall into 9 categories (0 = High Card through 8 = Straight Flush). The score is encoded as:

```
score = category × 15⁵ + tiebreaker_pack
```

The `15⁵` multiplier (759,375) is larger than any possible tiebreaker value, so the category always dominates. Within a category, tiebreaker ranks are packed in order of significance using base-15 arithmetic. For example:

- A pair of Aces with K-Q-J kickers: `pack(1, 14, 13, 12, 11)` = one pair, ace, king, queen, jack
- A pair of Kings: `pack(1, 13, ...)` — always less than the aces pair

This design means the evaluator never needs tuples or multi-key comparison — a single integer per hand is sufficient.

#### 7-card evaluation

For 7-card hands (2 hole + 5 community), the evaluator enumerates all C(7,5) = 21 five-card combinations and returns the best score. This is correct by definition — it always finds the best possible 5-card hand.

#### Special case: the wheel straight

A-2-3-4-5 (the "wheel") is the lowest straight. The code detects it explicitly: if `ranks == [14, 5, 4, 3, 2]` after sorting, the straight high is set to 5 (not 14), so a wheel ranks below a 6-high straight.

#### Unit tests (16 assertions)

All standard hand rankings are verified: pair beats high card, AA beats KK, two pair beats one pair, trips beats two pair, straight beats trips, flush beats straight, full house beats flush, quads beats full house, straight flush beats quads, royal flush beats straight flush, wheel is correctly identified as a straight, 7-card evaluation picks the correct best hand, identical hands are equal (split pot).

---

### `poker_agent/game.py` — Texas Hold'em Environment

#### `GameState` dataclass

A snapshot of the game at any point in time. Passed to agents so they have full information to make decisions:

| Field | Description |
|---|---|
| `hole_cards` | List of 2-card lists, one per player |
| `community_cards` | Cards dealt to the board so far |
| `pot` | Total chips in the pot |
| `stacks` | Each player's remaining chips |
| `current_player` | Whose turn it is |
| `street` | Current betting round: preflop / flop / turn / river |
| `current_bet` | Chips the current player must add to call |
| `min_raise` | Minimum legal raise size |
| `betting_history` | Ordered list of (player, action, amount) tuples |
| `street_investment` | Chips each player has put in this street (for raise math) |
| `total_investment` | Chips each player has put in this hand (for reward calculation) |

#### `PokerGame` class

Manages the full game lifecycle:

**`reset(dealer)`** — starts a new hand. The dealer posts the small blind (BB÷2), the other player posts the big blind. In heads-up hold'em the dealer acts first preflop (which is non-standard vs. multi-player but correct for heads-up). Returns the initial `GameState`.

**`legal_actions(state)`** — derives the valid action set from the current state:
- `"fold"` and `"call"` are available when there is a bet to face
- `"check"` is available when there is no bet
- `"raise"` is available when the player has enough chips beyond the call amount to meet the minimum raise

**`step(action, amount)`** — advances the game by one action. Returns `(new_state, rewards, done)`. Key behaviors:
- Validates the action is legal
- On **raise**: adds call amount + raise extra to pot, recalculates `current_bet` and `min_raise` for the opponent
- On **call**: checks whether the street is complete (both players equally invested) with a special preflop rule — after the SB calls, the BB still gets one more action before the street ends
- On **street completion**: deals community cards (3 on flop, 1 on turn, 1 on river) and resets street investment to zero. Postflop first-to-act is the non-dealer.
- On **all-in**: if either player reaches zero chips, the remaining community cards are run out automatically and a showdown occurs
- On **showdown**: evaluates both 7-card hands; awards pot to the winner. Ties split the pot evenly (odd chip goes to big blind position)
- On **fold**: the non-folding player wins all chips in the pot

**Reward convention**: `rewards[i]` = net chip change for player `i` in that hand. Zero-sum: `rewards[0] + rewards[1] == 0` always.

---

### `poker_agent/agents/` — Baseline Agents

#### `base.py` — Abstract Agent

Defines the interface all agents must implement:
```python
def act(game_state, player_id) -> (action: str, amount: int)
def reset()  # called at session start; optional to override
```

#### `random_agent.py` — RandomAgent

Picks uniformly at random from the legal action set. When raising, the amount is a random integer between `min_raise` and `min(3 × pot, stack)`. This agent serves as the weakest possible baseline — any agent that can't beat it is broken.

#### `rule_based_agent.py` — RuleBasedAgent

Uses a hardcoded hand strength heuristic based on the Chen formula:

**Preflop** — assigns each of the 169 starting hand types a score based on:
- Highest card value (Aces score 10, Kings 8, Queens 7, Jacks 6, others = rank ÷ 2)
- Pair bonus (double the score, minimum 5)
- Suited bonus (+2)
- Gap penalty (−1 for 1-gap, −2 for 2-gap, −4 for 3-gap, −5 for 4+-gap)
- Connectedness bonus (+1 for small connected cards)

Scores are normalized to [0, 1]. Decision thresholds:
- Strength ≥ 0.80 (top ~20%) → raise 2.5× big blind
- Strength ≥ 0.50 (top ~50%) → call
- Below 0.50 → fold (if facing a bet) or check

**Postflop** — a simple fixed rule: call if the bet is ≤ ⅓ of the pot, otherwise fold. This avoids complex postflop hand evaluation while still being more sensible than the random agent.

---

### `poker_agent/simulation.py` — Simulation Engine

**`run_simulation(agent0, agent1, n_hands, stack_size, big_blind)`**

Runs `n_hands` of heads-up poker and returns a `SimResults` dataclass containing:

- `mbb_per_hand_agent0/1` — win rate in milli-big-blinds per hand (standard poker metric). Formula: `(net_chips / n_hands / big_blind) × 1000`. At 1000 mbb = 1 big blind profit per hand.
- `action_counts_agent0/1` — fold/call/check/raise frequencies
- `chip_history` — agent 0's cumulative net chip change after each hand
- `errors` — count of exceptions (should always be 0)

**Non-tournament format**: stacks reset to `stack_size` each hand. This isolates per-hand performance without bankroll effects. Dealer alternates every hand to eliminate positional bias over large samples.

---

## Verification Results

```
Hand evaluator unit tests: 16/16 PASS
10,000-hand simulation:    0 errors
Action distribution:       ~22% fold, ~22% call, ~18% check, ~38% raise (RandomAgent)
```

The ~38% raise frequency for the random agent is expected: it can raise preflop in most spots where it has chips, and postflop as well. The action distributions for both players are nearly identical, confirming there's no systematic bias from dealer position over large samples.
