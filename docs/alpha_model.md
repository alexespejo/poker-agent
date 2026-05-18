# Milestone 2 — Alpha: EHS Agent (Monte Carlo + Pot Odds)

## Goal
Replace hardcoded hand strength rules with a principled statistical estimate of winning probability (Effective Hand Strength), and connect it to a pot-odds decision framework. The resulting agent should consistently beat the random agent by making mathematically sound call/fold/raise decisions.

---

## Core Concept: Effective Hand Strength (EHS)

EHS is the probability that your current hand will win at showdown against a uniformly random opponent hand, accounting for the cards that will still be dealt. It answers: *"If I played this hand to completion against a random opponent, how often would I win?"*

Unlike static hand rankings (which just compare current card strength), EHS captures:
- How likely your hand is to **improve** on future streets
- How likely the **opponent is to improve** as well
- The interaction between both effects

For example, a flush draw on the flop has weak current strength but high EHS because it completes frequently and beats most made hands when it does.

---

## What Was Built

### `poker_agent/monte_carlo.py` — EHS Estimator

**`estimate_ehs(hole_cards, community_cards, dead_cards, n_samples=1000) → float`**

Computes EHS by running Monte Carlo rollouts:

```
For each sample:
  1. Build the "remaining deck" = 52 cards minus all dead_cards
     (hole cards + community cards already known to us)
  2. Draw 2 random cards as the opponent's hole cards
  3. Complete the board to 5 community cards (randomly)
  4. Evaluate both 7-card hands using the hand evaluator
  5. Score: win = 1.0, tie = 0.5, loss = 0.0

EHS = mean score across all samples
```

The result is a float in [0, 1] representing expected win rate. With `n_samples=500` (used during play), estimates are fast enough for 10,000-hand simulations while remaining accurate enough for good decisions.

#### Implementation notes

- **Dead card exclusion**: All known cards (hole + community) are excluded from sampling by value, not object identity — this matters because the Monte Carlo module maintains its own `_FULL_DECK` reference independent of the game's deck.
- **NumPy acceleration**: If NumPy is installed, samples are drawn in batch using `np.random.choice(..., replace=False)` for each row, then evaluated in a Python loop. If NumPy is unavailable, falls back to `random.shuffle` on a 50-card pool — slower but identical results.
- **Degenerate case guard**: If the remaining deck has fewer cards than needed to complete a sample, returns 0.5 (neutral) rather than crashing.

#### EHS values for reference hands

| Hand | Community | EHS | Notes |
|---|---|---|---|
| A♥A♠ | (none) | ~0.851 | Preflop — strongest starting hand |
| K♥K♠ | (none) | ~0.825 | Preflop — very strong, most random hands lose |
| 7♥2♣ | (none) | ~0.366 | Preflop — weakest starting hand |
| A♥2♥ | K♥7♥3♣ | ~0.654 | Nut flush draw — high equity from draw potential |

Note: KK vs a *uniformly random* 2-card hand is ~82.5%, not ~73%. The 73% figure commonly cited is KK's equity against a *calling range* (hands opponents would voluntarily play), which skews toward stronger holdings. Against truly random cards — as in this simulator — KK dominates a very large fraction of possible opponent hands.

---

### `poker_agent/agents/ehs_agent.py` — EHS Agent

The EHS agent makes every decision by computing its current EHS and comparing it to the pot odds.

#### Pot odds

**Pot odds** measure the minimum win rate needed to break even on a call:

```
pot_odds = call_amount / (pot + call_amount)
```

For example: pot = 100, bet to call = 25 → `pot_odds = 25/125 = 0.20`. If you win more than 20% of the time, calling is profitable.

#### Decision logic

```
If no bet to face (call_amount == 0):
    ehs < 0.55  →  check   (hand too weak to build the pot)
    ehs ≥ 0.55  →  raise   (hand has edge, extract value)

If facing a bet:
    ehs > pot_odds + 0.15  →  raise   (strong edge, build the pot)
    ehs > pot_odds          →  call    (positive EV, continue)
    ehs ≤ pot_odds          →  fold    (not worth the cost)
```

The `+0.15` raise threshold (configurable via `raise_threshold`) ensures the agent only raises when it has a meaningful edge, not just a marginal one. A thin call might be correct; a thin raise just builds the pot when the edge is small.

#### Raise sizing

Uses a Kelly criterion-inspired formula:

```
raise_amount = pot × (ehs − 0.5) × 2
```

The multiplier `(ehs − 0.5) × 2` converts EHS into an "edge" value: at EHS=0.75 (25% edge over breakeven), the raise is 0.5 × pot. At EHS=0.90, the raise is 0.8 × pot. This sizes bets proportionally to hand strength.

The result is clamped to `[min_raise, 0.75 × stack]` to ensure legality and avoid over-committing.

#### Why this beats the random agent

The random agent makes raises, calls, and folds with equal probability regardless of hand strength. The EHS agent:
- Folds weak hands (EHS well below pot odds) instead of calling off chips
- Raises strong hands aggressively instead of just checking
- Sizes raises based on actual hand strength

The result over 10,000 hands: **+7,246 mbb/hand** (EHS agent vs. random). The random agent essentially donates chips by calling raises with trash hands and folding when they have strong hands after being bet off them.

---

### `poker_agent/stats.py` — Session Logger

**`SessionLogger`** records every action during a simulation run for auditing and analysis.

Each record captures: hand number, street, player, action, amount, EHS (if the agent computed one), pot size, and both stack sizes.

#### Key methods

- **`log_action(...)`** — append one action record
- **`record_hand_result(rewards)`** — update cumulative net chips at hand end
- **`save_log(filepath)`** — write all records to CSV for external analysis
- **`print_summary()`** — console report of action distribution, average EHS by street, and mbb/hand
- **`pot_odds_violations(player_id)`** — audit method: counts calls where `ehs < pot_odds`. This should always be 0 for the EHS agent, and it is — verified over 1,000 logged hands.

#### Pot-odds violation check (sanity test)

To confirm the decision logic is implemented correctly, the test script runs 1,000 hands with full logging and scans every call action for cases where `ehs < pot_odds`. **Result: 0 violations.** The agent never calls when the math says to fold.

---

## Verification Results

```
AA preflop EHS:        0.851  (expect ~0.85)  ✓
72o preflop EHS:       0.366  (expect ~0.35)  ✓
Pot-odds violations:   0                      ✓
EHSAgent mbb/hand:     +7,246 vs RandomAgent  ✓ (target: > 0)
```

#### EHS Agent action distribution vs. Random

| Action | EHSAgent | RandomAgent |
|--------|----------|-------------|
| fold   | 7.2%     | ~22%        |
| call   | 19.7%    | ~22%        |
| check  | 29.0%    | ~18%        |
| raise  | 44.1%    | ~38%        |

The EHS agent folds far less than the random agent (7.2% vs 22%) because it only folds when EHS is below pot odds — which often isn't the case with a decent hand facing a small bet. It raises more (44% vs 38%) because it aggressively builds pots when it has an edge.

---

## Design Decisions

**Why Monte Carlo instead of a lookup table?**
A precomputed EHS table could be faster, but Monte Carlo naturally handles any board state — preflop, flop, turn, or river — with the same code. It also gets more accurate as `n_samples` increases, giving a simple knob for the accuracy/speed tradeoff (tuned in Milestone 4).

**Why 500 samples during play?**
At 500 samples, standard error is ≈ ±0.022, which is accurate enough to distinguish "clear call," "clear fold," and "marginal" situations. Running 10,000 hands at 500 samples/decision takes ~60–90 seconds — fast enough for development iteration. Analysis runs use 2,000 samples for tighter estimates.

**Why EHS and not equity?**
Equity is the probability of winning with the current cards only. EHS accounts for future cards, making it meaningful at all stages of the hand — not just after the river is dealt.
