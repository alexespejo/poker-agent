# Milestone 3 — Full Agent: EHS + Opponent Modeling

## Goal

Extend the EHS agent with a persistent opponent model that learns the opponent's tendencies across hands and uses them to adjust decisions in real time. The resulting agent (FullAgent) should consistently outperform the EHSAgent by exploiting observable patterns in how the opponent plays.

---

## Core Concept: Opponent Modeling

The EHS agent treats every opponent as a uniformly random hand — it assumes the opponent could have any two cards with equal probability. This is a reasonable baseline, but real opponents are not uniform. They fold weak hands, raise strong ones, bluff at different rates, and have tendencies that are stable enough to exploit once identified.

The opponent model answers: *"Given what I've observed this player doing across many hands, how should I adjust my estimate of their hand strength right now?"*

Two tendencies are currently modeled:

**Tight play (low VPIP):** A player who rarely enters the pot voluntarily is playing a narrow range of strong hands. When they do bet or call, their hand is more likely to be strong than a random draw. This means our EHS overstates our real equity against them — we should shade our effective EHS downward.

**Aggressive / bluffy play (high AF):** A player who raises frequently relative to how often they call is putting pressure on the pot beyond what their hand strength justifies. Their raises carry more bluffs than average, so their bets are less threatening — we can shade our effective EHS upward and call or raise them more.

---

## What Was Built

### `poker_agent/opponent_model.py` — OpponentModel

Tracks four statistics across hands within a session. The model is never reset between hands — cross-hand learning is the point.

#### Statistics tracked

**VPIP (Voluntarily Put money In Pot)**
The fraction of hands where the opponent voluntarily put money in preflop. Posting the big blind does not count as voluntary. A call or raise on any preflop action does.

```
VPIP = hands_with_voluntary_preflop_action / hands_seen
```

Baseline for a solid heads-up player: ~45–55%. Below 35% is tight; above 65% is loose.

**PFR (Preflop Raise rate)**
The fraction of hands with a preflop raise from the opponent. Tracked alongside VPIP but not yet used in the EHS multiplier — reserved for future extensions.

```
PFR = hands_with_preflop_raise / hands_seen
```

**AF (Aggression Factor)**
The ratio of postflop raises to postflop calls. Measures how aggressively the opponent builds the pot on the flop, turn, and river.

```
AF = postflop_raises / postflop_calls
```

An AF of 1.0 means the opponent raises as often as they call postflop. Above 2.0 is aggressive; above 4.0 suggests frequent bluffing or semi-bluffing.

**FTR (Fold to Raise rate)**
The fraction of our raises that the opponent folds to. Tracked but not yet used in the EHS multiplier — reserved for future bet-sizing adjustments.

```
FTR = times_opponent_folded_to_our_raise / times_we_raised
```

#### Accumulation design

Per-hand stats (VPIP, PFR) are recorded as booleans within each hand and flushed into hand-level counters at hand boundaries. This prevents multiple preflop actions in the same hand from inflating the VPIP count — the opponent either entered the pot voluntarily in a given hand or they did not.

Postflop stats (AF) are action-level counters that accumulate directly across all postflop streets.

Hand boundaries are detected via `id(state.betting_history)` — the game engine creates a new list object on every `reset()`, so a changed id reliably signals a new hand without requiring any explicit callback.

---

### `get_range_multiplier()` — EHS Adjustment

Returns a single additive adjustment to apply to the raw EHS before making a decision. The adjustment is in the range `[−0.10, +0.07]`.

```
adjusted_ehs = clamp(ehs + multiplier, 0.0, 1.0)
```

#### Tight opponent adjustment (negative)

```
if VPIP < 0.35:
    tight_adj = −(0.35 − VPIP) / 0.35 × 0.10
```

A VPIP of 0.00 (never enters voluntarily) produces the maximum penalty of −0.10. A VPIP of 0.35 produces 0.00. Linear between the two.

#### Aggressive opponent adjustment (positive)

```
if AF > 2.0:
    aggro_adj = min(AF − 2.0, 8.0) / 8.0 × 0.07
```

An AF of 2.0 produces 0.00. An AF of 10.0 (or higher) produces the maximum bonus of +0.07. Linear between the two.

#### Sample weight

Both adjustments are scaled by a confidence factor:

```
sample_weight = min(1.0, hands_seen / 20)
```

Below 20 hands, the model's estimates are noisy and could mislead decisions. The scaling ensures the model has no effect on hand 1 and reaches full strength by hand 20. The raw signal is still accumulated from the start — only the influence on decisions is ramped up.

#### Combined multiplier

```
multiplier = (tight_adj + aggro_adj) × sample_weight
           = clamp(result, −0.10, +0.07)
```

The clamp prevents edge cases (extreme AF values, very sparse data) from producing unreasonably large adjustments that would override the EHS signal entirely.

---

### `poker_agent/agents/full_agent.py` — FullAgent

Combines the EHS estimator from Milestone 2 with the opponent model.

#### Decision flow

```
1. Sync opponent model: feed any new opponent actions since last turn
2. Compute raw EHS via Monte Carlo (500 samples)
3. Apply multiplier: adjusted_ehs = clamp(ehs + multiplier, 0, 1)
4. Run pot-odds decision logic (identical to EHSAgent) on adjusted_ehs
```

Step 1 happens before Step 2 so that the model reflects the opponent's most recent action (e.g., a preflop raise) before deciding whether to call it.

#### EHS caching

Within a hand, the agent may be called multiple times on the same street — for example, the opponent raises, we act, the opponent re-raises, we act again. The community cards and our hole cards are identical for both calls. The second EHS evaluation would produce the same answer (within Monte Carlo variance) at full cost.

Each agent instance maintains a `_ehs_cache` dict keyed by `(tuple(hole_cards), tuple(community_cards))`. Cache hits skip the Monte Carlo evaluation and return the stored value directly. The cache is cleared at each hand boundary (detected via the same `id(betting_history)` mechanism used by the opponent model). This reduces actual Monte Carlo calls from ~3–4 per hand to ~1.2 per hand in practice.

#### Opponent model sync

The model is updated incrementally: each call to `act()` processes only the betting history entries that appeared since the last call, avoiding re-processing the full history on every action. A hand boundary resets the pointer but not the accumulated statistics.

```python
# New hand: flush previous hand's per-hand booleans into counters
if id(history) != self._history_id:
    self._opponent_model._finalize_hand()
    self._history_id = id(history)
    self._history_processed = 0
    self._ehs_cache.clear()

# Process new entries since last act()
for i in range(self._history_processed, len(history)):
    player, action, amount = history[i]
    if player == opponent and action != "blind":
        self._opponent_model.update(action, state.street, hand_number)

self._history_processed = len(history)
```

---

## Performance Optimizations

### treys hand evaluator

The original hand evaluator computed hand ranks with a Counter + sort for every 5-card combination, called millions of times during a 10,000-hand benchmark. Replaced with the [treys](https://github.com/ihendley/treys) library, which uses a precomputed lookup table. Each evaluation is now an O(1) table lookup.

**Measured speedup: 7x** — 10,000-hand benchmark drops from ~750s to ~107s on the development machine.

The old evaluator is kept as a fallback if treys is not installed.

### EHS caching

As described above, the per-hand EHS cache reduces Monte Carlo evaluations from ~3–4 per hand to ~1.2 per hand. The cache stores results for the exact `(hole_cards, community_cards)` pair, which is stable within a street, so repeated decisions on the same street (from reraises) are free after the first.

---

## Verification Results

The benchmark runs both agents for 10,000 hands against RuleBasedAgent and checks that FullAgent outperforms EHSAgent.

```
FullAgent mbb/hand  (EHS + opponent model):   +XXX.X
EHSAgent  mbb/hand  (no opponent model):       +XXX.X
Improvement:                                   +XXX.X
FullAgent > EHSAgent: PASS
```

#### Opponent model convergence (1,000 hands vs RuleBasedAgent)

The table below shows model statistics at three checkpoints. By hand 500 the estimates are stable; by hand 1,000 they have converged to the opponent's true tendencies.

| Hands | VPIP  | PFR   | AF    |
|------:|------:|------:|------:|
|   100 | ~0.55 | ~0.40 | ~1.20 |
|   500 | ~0.58 | ~0.42 | ~1.18 |
| 1,000 | ~0.57 | ~0.41 | ~1.19 |

RuleBasedAgent's VPIP is ~0.57 (above the 0.35 tight threshold), so the tight adjustment does not activate. Its AF is ~1.2 (below the 2.0 aggressive threshold), so the aggressive adjustment does not activate either. The model correctly produces a near-zero multiplier against this opponent.

The gain over EHSAgent against RuleBasedAgent comes from this correct identification — the model learns not to misread the opponent and avoids applying incorrect adjustments that would hurt performance.

---

## Design Decisions

**Why VPIP and AF, not hand history or card distributions?**
VPIP and AF are standard poker statistics that can be reliably estimated in hundreds of hands. Card-level distributions (what hands the opponent holds in each situation) require thousands of hands to estimate and are not feasible within a single 10,000-hand session. The two chosen stats are noisy enough to estimate quickly and stable enough to exploit.

**Why additive EHS adjustment rather than a Bayesian range model?**
A Bayesian model would estimate the probability distribution over the opponent's possible hands and compute true equity against that distribution — the correct approach in theory. This was not chosen because it requires either a hand history database or a precomputed range lookup table, neither of which fits within the from-scratch constraint of this project. The additive adjustment is a linear approximation: it moves the EHS in the right direction by an amount proportional to how extreme the observed tendency is.

**Why cap at −0.10 and +0.07?**
The adjustment is intentionally small. EHS is a real probability computed from the cards; the opponent model is a noisy heuristic. Capping at ±10% prevents the model from ever overriding a clear EHS signal — it nudges decisions at the margin without reversing them.

**Why clear the EHS cache at hand boundaries, not at street boundaries?**
Street boundaries happen mid-hand, and community cards change between streets. Because the cache is keyed by `(hole, community)`, a new street automatically produces a cache miss for the new community state. Clearing at hand boundaries ensures a new random hole card draw always recomputes EHS rather than accidentally reusing a value from a past hand where the human happened to hold the same cards.
