# Benchmark how-to (`v5.py`)

Reference for running heads-up benchmarks and comparing FullAgent opponent-learning modes. All commands assume the repo root and `python3 v5.py`.

## Defaults (no flags)

| Setting | Value |
|---------|-------|
| Scored hands per pairing | 5,000 (`--hands`) |
| Stack per hand | 1,000 chips |
| Big blind | 10 |
| EHSAgent policy | Fixed EHS bars only (no pot odds) |
| FullAgent policy | EHS + pot odds (via internal `_decide(use_pot_odds=True)`) |
| FullAgent learning | `live` |
| Warm-up hands | 0 |
| Pairings | Full round-robin (10 unordered pairs among RuleBased, EHS, Full, TAG, LAG) |

```bash
python3 v5.py
```

---

## All CLI flags

| Flag | Type | Default | Applies when |
|------|------|---------|----------------|
| `--smoke` | switch | off | Exits after smoke; **ignores all other benchmark flags** |
| `--hands N` | int | 5000 | Scored hands per pairing (mbb denominator) |
| `--warmup-hands N` | int | 0 | Additive observe hands before scoring on **Full pairings only** (see learning modes) |
| `--full-learning MODE` | choice | `live` | FullAgent only; see modes below |
| `--focus-full` | switch | off | Restrict pairings to those including Full |
| `--focus-ehs` | switch | off | Restrict pairings to those including EHS |
| `--parallel` | switch | off | Run pairings in separate processes |
| `--jobs N` | int | CPU ∩ pairings | With `--parallel`; max workers |
| `--hand-log PATH` | optional path | off | Write scored-hand text logs |

### `--hand-log` forms

| Invocation | Behavior |
|------------|----------|
| (omit) | No hand logs |
| `--hand-log` | Directory `results/hand-logs/run-YYYYMMDD-HHMMSS/` |
| `--hand-log path/to/file.txt` | Single file (one pairing) or base dir (multiple pairings) |
| Multiple pairings | One file per pairing: `{base}/{name0}-vs-{name1}.txt` |

Hand logs record **scored hands only**. Headers note warm-up count when `warmup_hands > 0`.

### `--smoke` (special)

Runs 3 fixed pairings × 20 hands with agent verbose output:

- TAG vs LAG  
- Full vs TAG  
- Full vs LAG  

Does **not** use `--hands`, `--warmup-hands`, `--full-learning`, `--focus-*`, `--parallel`, or `--hand-log`. FullAgent always uses default `live` learning.

```bash
python3 v5.py --smoke
```

---

## Pairing sets

Agents in the round-robin: **RuleBased**, **EHS**, **Full**, **TAG**, **LAG** (Random is not in `v5`).

| Flags | Pairings run | Count |
|-------|----------------|------:|
| (none) | All combinations | 10 |
| `--focus-full` | RuleBased–Full, EHS–Full, Full–TAG, Full–LAG | 4 |
| `--focus-ehs` | RuleBased–EHS, EHS–Full, EHS–TAG, EHS–LAG | 4 |
| `--focus-full --focus-ehs` | EHS–Full only (intersection) | 1 |

---

## Hand budget: scored vs warm-up

- **`--hands N`** = hands that count toward **mbb/hand** and net chips.
- **`--warmup-hands W`** = extra hands run **before** scoring on pairings that include **Full** (not in mbb, not in hand logs). Pairings without Full (e.g. RuleBased vs EHS) run **`N` scored hands only**.
- **Total simulated per Full pairing** = `W + N` when warm-up is active and `--full-learning` is not `live`.

Example: `--warmup-hands 500 --hands 5000 --full-learning warmup_then_adapt` → Full pairings play 5,500 hands (500 observe + 5,000 scored); RuleBased vs EHS plays 5,000 scored only. At the warm-up/scored boundary on Full pairings, cumulative net resets to 0 for both players.

---

## FullAgent learning modes (`--full-learning`)

Only affects agents named **Full** in a pairing. Opponents are unchanged.

| Mode | Warm-up (`W > 0`) | Scored segment | Model updates during scored? |
|------|-------------------|----------------|-----------------------------|
| `live` | Ignored (`W` treated as 0) | Learn + apply from hand 1 | Yes |
| `warmup_then_adapt` | Observe: play as raw EHS (`call_adj`/`thr_red` = 0), **update stats** | Apply adjustments; net reset at boundary | Yes (continues learning) |
| `warmup_then_frozen` | Same observe warm-up | Apply **snapshot** from end of warm-up | **No** (frozen profile) |

### Mode comparison (why run each)

- **`live`** — Baseline: learning and exploitation from the first scored hand (includes `sample_weight` ramp over ~20 hands).
- **`warmup_then_adapt`** — Scout opponent without exploitation, then measure mbb with a warm start **and** ongoing adaptation.
- **`warmup_then_frozen`** — Scout once, lock adjustments for the scored block; isolates “know the opponent” without post-warm-up drift.

Diagnostics (when Full is in the run) show end-of-session VPIP/AF/FTR plus, for warm-up modes, **warm-up** `call_adj` / `thr_red` at the phase boundary.

More detail: [full_agent.md](full_agent.md#learning-schedules-opponentlearningmode).

---

## Flag interaction matrix

| Combination | Result |
|-------------|--------|
| `--smoke` + anything | Only smoke runs; other flags ignored |
| `--warmup-hands W` + `--full-learning live` | Warning: warm-up ignored |
| Pairing without Full + warm-up modes | Warm-up skipped; `N` scored hands only |
| `--warmup-hands 0` + `warmup_then_*` | No observe phase; behaves like `live` for that pairing (still “adapt” or “frozen” with empty prior) |
| `--focus-full` + warm-up modes | Warm-up/scored on all 4 Full pairings |
| `--parallel` + `--hand-log` | Supported; each worker writes its pairing log |
| `--focus-full --focus-ehs` | Single pairing EHS vs Full (good for A/B learning experiments) |

---

## Recommended benchmark recipes

### Quick sanity

```bash
python3 v5.py --smoke
python3 v5.py --focus-full --hands 100
```

### Full round-robin (production-style)

```bash
python3 v5.py --hands 5000
python3 v5.py --parallel --jobs 6 --hands 5000
```

### FullAgent learning A/B (same scored hands, different modes)

Run three jobs with the same `N` and `W`; compare mbb and diagnostics.

```bash
# Baseline
python3 v5.py --focus-full --hands 5000 --full-learning live

# Scout then adapt
python3 v5.py --focus-full --hands 5000 --full-learning warmup_then_adapt --warmup-hands 500

# Scout then frozen profile
python3 v5.py --focus-full --hands 5000 --full-learning warmup_then_frozen --warmup-hands 500
```

### Single matchup (fast iteration)

```bash
python3 v5.py --focus-full --focus-ehs --hands 2000 --warmup-hands 200 --full-learning warmup_then_adapt
```

### Full vs EHS with hand review

```bash
python3 v5.py --focus-full --focus-ehs --hands 50 --hand-log results/hand-logs/my-run.txt
```

### Logged multi-pairing run

```bash
python3 v5.py --focus-full --hands 100 --hand-log
# → results/hand-logs/run-<timestamp>/full-vs-tag.txt, etc.
```

---

## Reading output

1. **Raw Results** — mbb/hand per agent per pairing (scored hands only).
2. **Win-Rate Matrix** — row agent’s mbb vs column opponent; `—` on diagonal.
3. **FullAgent Opponent-Model Diagnostics** — only when Full played:
   - Session: `VPIP`, `AF`, `FTR`, `call_adj`, `thr_red`
   - With warm-up: `warm-up: hands=… call_adj=… thr_red=…` (profile at start of scored segment)

**mbb/hand** = `(net_chips / scored_hands / big_blind) × 1000`. Positive = that seat won chips over the scored sample.

Exit code **1** if any pairing reported errors.

---

## Experiment checklist

When comparing learning modes fairly:

1. Use the same `--hands` and `--warmup-hands` across runs.
2. Prefer `--focus-full` (or `--focus-full --focus-ehs`) to limit runtime.
3. Note `live` ignores `--warmup-hands`.
4. For `warmup_then_frozen`, session diagnostics may show growing `hands_seen` but decisions use the warm-up snapshot.
5. Save stdout or redirect to `results/behavior-agent-performance/` if you keep run logs by hand.

---

## Related scripts (not `v5`)

| Script | Purpose |
|--------|---------|
| `v4.py` | Older FullAgent grid/eval pipeline (`--eval-hands`, `--selfplay`, etc.) |
| `v3.py` | Interactive Full vs RuleBased demo |
| `play.py` | Human vs agent |

For opponent-model design: [full_agent.md](full_agent.md).
