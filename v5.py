"""Milestone 5 — five-archetype round-robin benchmark.

Runs every agent (Random, RuleBased, EHS, Full, TAG, LAG) head-to-head and
reports per-pairing win rates plus FullAgent's opponent-model diagnostics. The
TAG and LAG archetypes exist to exercise FullAgent's opponent model: TAG is
tight-aggressive (low VPIP, high AF) and LAG is loose-aggressive (high VPIP,
high AF). The diagnostics section confirms the model reads each profile.

Usage:
  python3 v5.py --smoke               # quick smoke test: 20 hands, verbose output
  python3 v5.py                       # full round-robin (5,000 hands per pairing)
  python3 v5.py --hands 2000          # faster run for testing
  python3 v5.py --parallel --jobs 6   # parallel pairings
  python3 v5.py --focus-full          # only FullAgent matchups
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))

from poker_agent.agents.ehs_agent import EHSAgent
from poker_agent.agents.full_agent import FullAgent
from poker_agent.agents.lag_agent import LooseAggressiveAgent
from poker_agent.agents.random_agent import RandomAgent
from poker_agent.agents.rule_based_agent import RuleBasedAgent
from poker_agent.agents.tag_agent import TightAggressiveAgent
from poker_agent.simulation import run_simulation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STACK_SIZE = 1000
BIG_BLIND = 10
DEFAULT_HANDS = 5_000

# FullAgent: best config from v4.
FULL_SAMPLES = 1000
FULL_RAISE_THRESHOLD = 0.10

# EHSAgent baseline config.
EHS_SAMPLES = 500
EHS_RAISE_THRESHOLD = 0.15

# Archetype agents.
ARCH_SAMPLES = 200
LAG_SEED = 42

# Agent ordering — itertools.combinations over this list reproduces the exact
# round-robin pairing order documented in the benchmark spec.
AGENTS = ["RuleBased", "EHS", "Full", "TAG", "LAG"]

DISPLAY = {
    "Random": "RandomAgent",
    "RuleBased": "RuleBasedAgent",
    "EHS": "EHSAgent",
    "Full": "FullAgent",
    "TAG": "TAG",
    "LAG": "LAG",
}


def make_agent(name: str):
    """Construct an agent instance from its short name."""
    if name == "Random":
        return RandomAgent()
    if name == "RuleBased":
        return RuleBasedAgent()
    if name == "EHS":
        return EHSAgent(n_samples=EHS_SAMPLES, raise_threshold=EHS_RAISE_THRESHOLD)
    if name == "Full":
        return FullAgent(n_samples=FULL_SAMPLES, raise_threshold=FULL_RAISE_THRESHOLD)
    if name == "TAG":
        return TightAggressiveAgent(n_samples=ARCH_SAMPLES)
    if name == "LAG":
        return LooseAggressiveAgent(n_samples=ARCH_SAMPLES, seed=LAG_SEED)
    raise ValueError(f"Unknown agent: {name}")


def _full_diag(full_agent: FullAgent) -> dict:
    """Snapshot FullAgent's opponent-model stats (call after finalize_session)."""
    m = full_agent.opponent_model
    call_adj, thr_red = m.get_range_multiplier()
    return {
        "vpip": m.vpip,
        "af": m.aggression_factor,
        "ftr": m.fold_to_raise_rate,
        "call_adj": call_adj,
        "thr_red": thr_red,
        "hands_seen": m.hands_seen,
    }


# ---------------------------------------------------------------------------
# Pairing execution
# ---------------------------------------------------------------------------

@dataclass
class PairingResult:
    name0: str
    name1: str
    mbb0: float
    mbb1: float
    actions0: dict
    actions1: dict
    errors: int
    full_diag: dict | None = None
    full_side: int | None = None


def run_pairing(name0: str, name1: str, n_hands: int, show_progress: bool = False) -> PairingResult:
    """Run a single pairing; finalize and snapshot any FullAgent involved."""
    a0 = make_agent(name0)
    a1 = make_agent(name1)
    sim = run_simulation(
        a0, a1,
        n_hands=n_hands,
        stack_size=STACK_SIZE,
        big_blind=BIG_BLIND,
        show_progress=show_progress,
    )

    diag = None
    full_side = None
    if name0 == "Full":
        a0.finalize_session()
        diag = _full_diag(a0)
        full_side = 0
    if name1 == "Full":
        a1.finalize_session()
        if diag is None:
            diag = _full_diag(a1)
            full_side = 1

    return PairingResult(
        name0=name0,
        name1=name1,
        mbb0=sim.mbb_per_hand_agent0,
        mbb1=sim.mbb_per_hand_agent1,
        actions0=sim.action_counts_agent0,
        actions1=sim.action_counts_agent1,
        errors=sim.errors,
        full_diag=diag,
        full_side=full_side,
    )


def _pairing_worker(args: tuple[str, str, int]) -> PairingResult:
    """ProcessPoolExecutor entry point — receives names, builds agents locally."""
    name0, name1, n_hands = args
    return run_pairing(name0, name1, n_hands, show_progress=False)


def build_pairings(focus_full: bool) -> list[tuple[str, str]]:
    """All unordered pairs, optionally restricted to FullAgent matchups."""
    pairs = list(itertools.combinations(AGENTS, 2))
    if focus_full:
        pairs = [p for p in pairs if "Full" in p]
    return pairs


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def fmt_actions(counts: dict) -> str:
    return (
        f"fold={counts.get('fold', 0)} call={counts.get('call', 0)} "
        f"check={counts.get('check', 0)} raise={counts.get('raise', 0)}"
    )


def fmt_diag(diag: dict) -> str:
    return (
        f"VPIP={diag['vpip']:.2f}  AF={diag['af']:.1f}  FTR={diag['ftr']:.2f}  |  "
        f"call_adj={diag['call_adj']:+.3f}  thr_red={diag['thr_red']:+.3f}"
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke() -> None:
    """Run 3 representative pairings for 20 hands with full verbose output."""
    pairings = [
        ("TAG", "LAG"),
        ("Full", "TAG"),
        ("Full", "LAG"),
    ]

    results: list[tuple[str, str, object, FullAgent | None]] = []
    for name0, name1 in pairings:
        a0 = make_agent(name0)
        a1 = make_agent(name1)
        a0.verbose = True
        a1.verbose = True

        print(f"\n----- Smoke: {name0} vs {name1} (20 hands, verbose) -----")
        error_msg = None
        sim = None
        try:
            sim = run_simulation(
                a0, a1,
                n_hands=20,
                stack_size=STACK_SIZE,
                big_blind=BIG_BLIND,
                verbose=True,
                show_progress=False,
            )
            if name0 == "Full":
                a0.finalize_session()
            if name1 == "Full":
                a1.finalize_session()
        except Exception as e:  # noqa: BLE001 — report, don't assert
            error_msg = str(e)

        full_agent = a0 if name0 == "Full" else (a1 if name1 == "Full" else None)
        results.append((name0, name1, sim if error_msg is None else error_msg, full_agent))

    # Summary block
    section("Smoke Test — 20 hands each")
    for name0, name1, sim, full_agent in results:
        print(f"\n  {DISPLAY[name0]} vs {DISPLAY[name1]}")
        if isinstance(sim, str):
            print(f"    ✗ ERROR: {sim}")
            continue

        print(f"    {DISPLAY[name0]:<10} mbb/hand: {sim.mbb_per_hand_agent0:+.1f}   "
              f"actions: {fmt_actions(sim.action_counts_agent0)}")
        print(f"    {DISPLAY[name1]:<10} mbb/hand: {sim.mbb_per_hand_agent1:+.1f}   "
              f"actions: {fmt_actions(sim.action_counts_agent1)}")

        if full_agent is not None:
            diag = _full_diag(full_agent)
            print(f"    Opponent model after {sim.hands_played} hands:")
            print(f"      {fmt_diag(diag)}")
            if diag["hands_seen"] < 20:
                print("      (model still warming up — sample_weight < 1.0 until 20 hands)")

        print("    ✓ no errors")

    print("\n  All smoke tests passed. Run without --smoke for the full benchmark.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Round-robin benchmark
# ---------------------------------------------------------------------------

def run_benchmark(n_hands: int, focus_full: bool, parallel: bool, jobs: int | None) -> None:
    pairs = build_pairings(focus_full)
    section(
        f"Round-Robin Benchmark — {len(pairs)} pairings × {n_hands:,} hands"
        + ("  (FullAgent matchups only)" if focus_full else "")
    )

    results: list[PairingResult] = []

    if parallel:
        n_workers = max(1, jobs) if jobs is not None else min(os.cpu_count() or 1, len(pairs))
        print(f"  Running {len(pairs)} pairings in parallel ({n_workers} workers)...\n")
        tasks = [(n0, n1, n_hands) for n0, n1 in pairs]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_pairing_worker, t): (t[0], t[1]) for t in tasks}
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                print(f"    done: {res.name0} vs {res.name1}  "
                      f"({res.mbb0:+.1f} / {res.mbb1:+.1f}, errors={res.errors})")
    else:
        for i, (name0, name1) in enumerate(pairs, 1):
            print(f"  Pairing {i}/{len(pairs)}: {name0} vs {name1}...")
            res = run_pairing(name0, name1, n_hands, show_progress=True)
            results.append(res)

    # Keep results in the canonical pairing order for stable output.
    order = {pair: idx for idx, pair in enumerate(pairs)}
    results.sort(key=lambda r: order[(r.name0, r.name1)])

    _print_raw_table(results)
    _print_matrix(results)
    _print_full_diagnostics(results)

    total_errors = sum(r.errors for r in results)
    print(f"\n  Total errors across all pairings: {total_errors}")
    if total_errors:
        sys.exit(1)


def _print_raw_table(results: list[PairingResult]) -> None:
    section("Raw Results")
    print(f"  {'Pairing':<32}{'agent0 mbb/hand':>16}{'agent1 mbb/hand':>18}")
    print(f"  {'-' * 63}")
    for r in results:
        label = f"{r.name0} vs {r.name1}"
        print(f"  {label:<32}{r.mbb0:>+16.1f}{r.mbb1:>+18.1f}")


def _print_matrix(results: list[PairingResult]) -> None:
    section("Win-Rate Matrix (mbb/hand)")

    matrix: dict[str, dict[str, float]] = {a: {} for a in AGENTS}
    for r in results:
        matrix[r.name0][r.name1] = r.mbb0
        matrix[r.name1][r.name0] = r.mbb1

    col_w = 13
    header = f"  {'Agent':<16}"
    for opp in AGENTS:
        header += f"{'vs ' + opp:>{col_w}}"
    header += f"{'avg':>{col_w}}"
    print(header)
    print(f"  {'-' * (16 + col_w * (len(AGENTS) + 1) - 2)}")

    for agent in AGENTS:
        row = f"  {DISPLAY[agent]:<16}"
        vals: list[float] = []
        for opp in AGENTS:
            if opp == agent:
                row += f"{'—':>{col_w}}"
            elif opp in matrix[agent]:
                v = matrix[agent][opp]
                vals.append(v)
                row += f"{v:>+{col_w}.1f}"
            else:
                row += f"{'·':>{col_w}}"
        avg = sum(vals) / len(vals) if vals else 0.0
        row += f"{avg:>+{col_w}.1f}"
        print(row)


def _print_full_diagnostics(results: list[PairingResult]) -> None:
    diag_results = [r for r in results if r.full_diag is not None]
    if not diag_results:
        return

    section("FullAgent Opponent-Model Diagnostics")
    print("  Confirms the model reads each archetype before judging win rate.\n")
    for r in diag_results:
        opponent = r.name1 if r.full_side == 0 else r.name0
        label = f"FullAgent vs {DISPLAY[opponent]}"
        print(f"  {label:<24} —  {fmt_diag(r.full_diag)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 5 — round-robin benchmark")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a quick 20-hand verbose smoke test and exit")
    parser.add_argument("--hands", type=int, default=DEFAULT_HANDS,
                        help=f"Hands per pairing (default: {DEFAULT_HANDS})")
    parser.add_argument("--parallel", action="store_true",
                        help="Run pairings in parallel (multiprocessing)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Max parallel workers (default: min(CPU count, num pairings))")
    parser.add_argument("--focus-full", action="store_true",
                        help="Only run FullAgent matchups")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
        return

    run_benchmark(
        n_hands=args.hands,
        focus_full=args.focus_full,
        parallel=args.parallel,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
