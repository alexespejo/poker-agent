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
  python3 v5.py --focus-ehs           # only EHSAgent matchups
  python3 v5.py --hand-log logs.txt   # per-hand log (one file per pairing if multiple)
  python3 v5.py --focus-full --full-learning warmup_then_adapt --warmup-hands 500

See docs/benchmark-howto.md for all flags, learning modes, and recipe commands.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from poker_agent.agents.ehs_agent import EHSAgent
from poker_agent.agents.full_agent import FullAgent
from poker_agent.learning_mode import OpponentLearningMode, parse_learning_mode
from poker_agent.agents.lag_agent import LooseAggressiveAgent
from poker_agent.agents.random_agent import RandomAgent
from poker_agent.agents.rule_based_agent import RuleBasedAgent
from poker_agent.agents.tag_agent import TightAggressiveAgent
from poker_agent.hand_log import HandLog
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


def make_agent(
    name: str,
    *,
    full_learning_mode: OpponentLearningMode = OpponentLearningMode.LIVE,
):
    """Construct an agent instance from its short name."""
    if name == "Random":
        return RandomAgent()
    if name == "RuleBased":
        return RuleBasedAgent()
    if name == "EHS":
        return EHSAgent(n_samples=EHS_SAMPLES, raise_threshold=EHS_RAISE_THRESHOLD)
    if name == "Full":
        return FullAgent(
            n_samples=FULL_SAMPLES,
            raise_threshold=FULL_RAISE_THRESHOLD,
            learning_mode=full_learning_mode,
        )
    if name == "TAG":
        return TightAggressiveAgent(n_samples=ARCH_SAMPLES)
    if name == "LAG":
        return LooseAggressiveAgent(n_samples=ARCH_SAMPLES, seed=LAG_SEED)
    raise ValueError(f"Unknown agent: {name}")


def _full_diag(full_agent: FullAgent) -> dict:
    """Snapshot FullAgent's opponent-model stats (call after finalize_session)."""
    m = full_agent.opponent_model
    call_adj, thr_red = m.get_range_multiplier()
    diag = {
        "vpip": m.vpip,
        "pfr": m.pfr,
        "af": m.aggression_factor,
        "ftr": m.fold_to_raise_rate,
        "call_adj": call_adj,
        "thr_red": thr_red,
        "hands_seen": m.hands_seen,
        "phase": full_agent.phase,
        "learning_mode": full_agent.learning_mode.value,
    }
    warmup = full_agent.warmup_diagnostics
    if warmup is not None:
        diag["warmup_hands_seen"] = warmup["hands_seen"]
        diag["warmup_call_adj"] = warmup["call_adj"]
        diag["warmup_thr_red"] = warmup["thr_red"]
        diag["warmup_vpip"] = warmup["vpip"]
    return diag


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
    warmup_hands: int = 0


def _hand_log_path(
    base: str | Path,
    name0: str,
    name1: str,
    *,
    multi_pairing: bool,
) -> Path:
    """Resolve log path; append pairing slug when several pairings share one base."""
    path = Path(base)
    if multi_pairing or path.suffix == "":
        slug = f"{name0.lower()}-vs-{name1.lower()}"
        if path.suffix:
            path = path.with_name(f"{path.stem}-{slug}{path.suffix}")
        else:
            path = path / f"{slug}.txt"
    return path


def run_pairing(
    name0: str,
    name1: str,
    n_hands: int,
    show_progress: bool = False,
    hand_log_path: str | Path | None = None,
    multi_pairing_log: bool = False,
    warmup_hands: int = 0,
    full_learning_mode: OpponentLearningMode = OpponentLearningMode.LIVE,
) -> PairingResult:
    """Run a single pairing; finalize and snapshot any FullAgent involved."""
    has_full = "Full" in (name0, name1)
    effective_warmup = warmup_hands if has_full else 0
    if (
        full_learning_mode == OpponentLearningMode.LIVE
        and warmup_hands > 0
    ):
        effective_warmup = 0

    a0 = make_agent(name0, full_learning_mode=full_learning_mode)
    a1 = make_agent(name1, full_learning_mode=full_learning_mode)

    hand_log = None
    if hand_log_path is not None:
        log_path = _hand_log_path(
            hand_log_path, name0, name1, multi_pairing=multi_pairing_log,
        )
        hand_log = HandLog(
            log_path,
            (DISPLAY[name0], DISPLAY[name1]),
            stack_size=STACK_SIZE,
            big_blind=BIG_BLIND,
            n_hands=n_hands,
            pairing_label=f"{DISPLAY[name0]} vs {DISPLAY[name1]}",
            warmup_hands=effective_warmup,
        )

    sim = run_simulation(
        a0, a1,
        n_hands=n_hands,
        stack_size=STACK_SIZE,
        big_blind=BIG_BLIND,
        show_progress=show_progress,
        hand_log=hand_log,
        warmup_hands=effective_warmup,
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
        warmup_hands=effective_warmup,
    )


def _pairing_worker(
    args: tuple[str, str, int, str | None, bool, int, str],
) -> PairingResult:
    """ProcessPoolExecutor entry point — receives names, builds agents locally."""
    name0, name1, n_hands, hand_log_path, multi_pairing_log, warmup_hands, mode_str = args
    return run_pairing(
        name0, name1, n_hands,
        show_progress=False,
        hand_log_path=hand_log_path,
        multi_pairing_log=multi_pairing_log,
        warmup_hands=warmup_hands,
        full_learning_mode=parse_learning_mode(mode_str),
    )


def build_pairings(focus_full: bool, focus_ehs: bool = False) -> list[tuple[str, str]]:
    """All unordered pairs, optionally restricted to a single agent's matchups."""
    pairs = list(itertools.combinations(AGENTS, 2))
    if focus_full:
        pairs = [p for p in pairs if "Full" in p]
    if focus_ehs:
        pairs = [p for p in pairs if "EHS" in p]
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
    base = (
        f"VPIP={diag['vpip']:.2f}  PFR={diag['pfr']:.2f}  AF={diag['af']:.1f}  "
        f"FTR={diag['ftr']:.2f}  |  "
        f"call_adj={diag['call_adj']:+.3f}  thr_red={diag['thr_red']:+.3f}"
    )
    if "warmup_hands_seen" in diag:
        base += (
            f"  |  warm-up: hands={diag['warmup_hands_seen']} "
            f"call_adj={diag['warmup_call_adj']:+.3f} "
            f"thr_red={diag['warmup_thr_red']:+.3f}"
        )
    return base


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

def run_benchmark(
    n_hands: int,
    focus_full: bool,
    parallel: bool,
    jobs: int | None,
    focus_ehs: bool = False,
    hand_log: str | None = None,
    warmup_hands: int = 0,
    full_learning_mode: OpponentLearningMode = OpponentLearningMode.LIVE,
) -> None:
    pairs = build_pairings(focus_full, focus_ehs)
    if focus_full:
        suffix = "  (FullAgent matchups only)"
    elif focus_ehs:
        suffix = "  (EHSAgent matchups only)"
    else:
        suffix = ""
    hands_line = f"{len(pairs)} pairings × {n_hands:,} scored hands"
    has_any_full = "Full" in {a for p in pairs for a in p}
    if (
        warmup_hands > 0
        and full_learning_mode != OpponentLearningMode.LIVE
        and has_any_full
    ):
        hands_line += f" (+ {warmup_hands:,} warm-up per Full pairing)"
    section(f"Round-Robin Benchmark — {hands_line}" + suffix)
    if "Full" in {a for p in pairs for a in p}:
        print(f"  FullAgent learning: {full_learning_mode.value}")
        if warmup_hands > 0 and full_learning_mode != OpponentLearningMode.LIVE:
            print(
                f"  Warm-up hands (not scored, Full pairings only): {warmup_hands:,}"
            )
        elif warmup_hands > 0:
            print("  (--warmup-hands ignored in live mode)")
        print()

    log_base: str | None = hand_log
    if log_base is None:
        multi_pairing_log = False
    else:
        multi_pairing_log = len(pairs) > 1
        if multi_pairing_log:
            print(f"  Hand logs → {Path(log_base)}/*-vs-*.txt (one file per pairing)")
        else:
            resolved = _hand_log_path(log_base, pairs[0][0], pairs[0][1], multi_pairing=False)
            print(f"  Hand log → {resolved}")
        print()

    results: list[PairingResult] = []

    if parallel:
        n_workers = max(1, jobs) if jobs is not None else min(os.cpu_count() or 1, len(pairs))
        print(f"  Running {len(pairs)} pairings in parallel ({n_workers} workers)...\n")
        tasks = [
            (n0, n1, n_hands, log_base, multi_pairing_log, warmup_hands,
             full_learning_mode.value)
            for n0, n1 in pairs
        ]
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
            res = run_pairing(
                name0, name1, n_hands,
                show_progress=True,
                hand_log_path=log_base,
                multi_pairing_log=multi_pairing_log,
                warmup_hands=warmup_hands,
                full_learning_mode=full_learning_mode,
            )
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
    parser.add_argument("--focus-ehs", action="store_true",
                        help="Only run EHSAgent matchups")
    parser.add_argument(
        "--hand-log",
        metavar="PATH",
        nargs="?",
        const="results/hand-logs",
        default=None,
        help="Write per-hand log to PATH (default dir: results/hand-logs). "
             "Multiple pairings get separate files.",
    )
    parser.add_argument(
        "--full-learning",
        choices=[m.value for m in OpponentLearningMode],
        default=OpponentLearningMode.LIVE.value,
        help="FullAgent opponent learning schedule (default: live)",
    )
    parser.add_argument(
        "--warmup-hands",
        type=int,
        default=0,
        metavar="N",
        help="Additive warm-up hands before scored segment (FullAgent observe modes)",
    )
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
        return

    learning_mode = parse_learning_mode(args.full_learning)
    if args.warmup_hands < 0:
        parser.error("--warmup-hands must be >= 0")
    if args.warmup_hands > 0 and learning_mode == OpponentLearningMode.LIVE:
        print("  Note: --warmup-hands ignored when --full-learning is live")

    hand_log_path = args.hand_log
    if hand_log_path == "results/hand-logs":
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        hand_log_path = f"results/hand-logs/run-{ts}"

    run_benchmark(
        n_hands=args.hands,
        focus_full=args.focus_full,
        parallel=args.parallel,
        jobs=args.jobs,
        focus_ehs=args.focus_ehs,
        hand_log=hand_log_path,
        warmup_hands=args.warmup_hands,
        full_learning_mode=learning_mode,
    )


if __name__ == "__main__":
    main()
