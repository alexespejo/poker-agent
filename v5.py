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
  python3 v5.py --focus-ehs-full      # EHS + Full vs others and each other
python3 v5.py --save-results        # write report to results/behavior-agent-performance/
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "behavior-agent-performance")

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
class SessionConfig:
    """Snapshot of parameters used for a benchmark run."""
    timestamp: str
    command: str
    n_hands: int
    focus_full: bool
    focus_ehs_full: bool
    parallel: bool
    jobs: int | None
    pairings: list[tuple[str, str]]
    stack_size: int = STACK_SIZE
    big_blind: int = BIG_BLIND
    ehs_samples: int = EHS_SAMPLES
    ehs_raise_threshold: float = EHS_RAISE_THRESHOLD
    full_samples: int = FULL_SAMPLES
    full_raise_threshold: float = FULL_RAISE_THRESHOLD
    arch_samples: int = ARCH_SAMPLES
    lag_seed: int = LAG_SEED
    agents: list[str] = field(default_factory=lambda: list(AGENTS))


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


def build_pairings(focus_full: bool, focus_ehs_full: bool) -> list[tuple[str, str]]:
    """All unordered pairs, optionally restricted to specific agent matchups."""
    pairs = list(itertools.combinations(AGENTS, 2))
    if focus_ehs_full:
        pairs = [p for p in pairs if "EHS" in p or "Full" in p]
    elif focus_full:
        pairs = [p for p in pairs if "Full" in p]
    return pairs


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _section_lines(title: str) -> list[str]:
    return ["", "=" * 60, f"  {title}", "=" * 60]


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


def _build_matrix(results: list[PairingResult]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {a: {} for a in AGENTS}
    for r in results:
        matrix[r.name0][r.name1] = r.mbb0
        matrix[r.name1][r.name0] = r.mbb1
    return matrix


def _format_session_config(config: SessionConfig) -> list[str]:
    focus = "all pairings"
    if config.focus_ehs_full:
        focus = "EHS + Full matchups only"
    elif config.focus_full:
        focus = "FullAgent matchups only"

    lines = _section_lines("Session Configuration")
    lines.extend([
        f"  timestamp:        {config.timestamp}",
        f"  command:          {config.command}",
        "",
        "  Run settings:",
        f"    hands/pairing:  {config.n_hands:,}",
        f"    focus:          {focus}",
        f"    parallel:       {config.parallel}"
        + (f"  (jobs={config.jobs})" if config.parallel and config.jobs else ""),
        f"    pairings:       {len(config.pairings)}",
    ])
    for name0, name1 in config.pairings:
        lines.append(f"      - {name0} vs {name1}")

    lines.extend([
        "",
        "  Game settings:",
        f"    stack_size:     {config.stack_size}",
        f"    big_blind:      {config.big_blind}",
        "",
        "  Agent parameters:",
        f"    EHSAgent:       n_samples={config.ehs_samples}, raise_threshold={config.ehs_raise_threshold}",
        f"    FullAgent:      n_samples={config.full_samples}, raise_threshold={config.full_raise_threshold}",
        f"    TAG/LAG:        n_samples={config.arch_samples}, LAG seed={config.lag_seed}",
    ])
    return lines


def _format_raw_table(results: list[PairingResult]) -> list[str]:
    lines = _section_lines("Raw Results")
    lines.extend([
        f"  {'Pairing':<32}{'agent0 mbb/hand':>16}{'agent1 mbb/hand':>18}",
        f"  {'-' * 63}",
    ])
    for r in results:
        label = f"{r.name0} vs {r.name1}"
        lines.append(f"  {label:<32}{r.mbb0:>+16.1f}{r.mbb1:>+18.1f}")
    return lines


def _format_action_summary(results: list[PairingResult]) -> list[str]:
    lines = _section_lines("Action Counts")
    for r in results:
        lines.append(f"\n  {r.name0} vs {r.name1}")
        lines.append(f"    {DISPLAY[r.name0]:<16} {fmt_actions(r.actions0)}")
        lines.append(f"    {DISPLAY[r.name1]:<16} {fmt_actions(r.actions1)}")
        if r.errors:
            lines.append(f"    errors: {r.errors}")
    return lines


def _format_matrix(results: list[PairingResult]) -> list[str]:
    matrix = _build_matrix(results)
    col_w = 13
    lines = _section_lines("Win-Rate Matrix (mbb/hand)")
    header = f"  {'Agent':<16}"
    for opp in AGENTS:
        header += f"{'vs ' + opp:>{col_w}}"
    header += f"{'avg':>{col_w}}"
    lines.append(header)
    lines.append(f"  {'-' * (16 + col_w * (len(AGENTS) + 1) - 2)}")

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
        lines.append(row)
    return lines


def _format_full_diagnostics(results: list[PairingResult]) -> list[str]:
    diag_results = [r for r in results if r.full_diag is not None]
    if not diag_results:
        return []

    lines = _section_lines("FullAgent Opponent-Model Diagnostics")
    lines.append("  Confirms the model reads each archetype before judging win rate.")
    lines.append("")
    for r in diag_results:
        opponent = r.name1 if r.full_side == 0 else r.name0
        label = f"FullAgent vs {DISPLAY[opponent]}"
        lines.append(f"  {label:<24} —  {fmt_diag(r.full_diag)}")
    return lines


def format_report(config: SessionConfig, results: list[PairingResult]) -> str:
    """Build a human-readable report with session params and all stats."""
    total_errors = sum(r.errors for r in results)
    lines = [
        "Milestone 5 — Round-Robin Benchmark Report",
        "",
        *_format_session_config(config),
        *_format_raw_table(results),
        *_format_action_summary(results),
        *_format_matrix(results),
        *_format_full_diagnostics(results),
        "",
        f"  Total errors across all pairings: {total_errors}",
        "",
    ]
    return "\n".join(lines)


def resolve_results_path(save_results: str) -> str:
    """Map --save-results value to an output file path."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_name = f"run-{timestamp}.txt"

    if not save_results:
        return os.path.join(RESULTS_DIR, default_name)

    path = os.path.abspath(save_results)
    if path.endswith(os.sep) or (os.path.isdir(path) and os.path.exists(path)):
        return os.path.join(path, default_name)
    if os.path.isdir(path):
        return os.path.join(path, default_name)
    if not os.path.splitext(path)[1]:
        return os.path.join(path, default_name)
    return path


def save_report(path: str, config: SessionConfig, results: list[PairingResult]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_report(config, results))
    print(f"\n  Report saved: {path}")


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
    focus_ehs_full: bool,
    parallel: bool,
    jobs: int | None,
    save_results: str | None,
) -> None:
    pairs = build_pairings(focus_full, focus_ehs_full)
    config = SessionConfig(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        command=" ".join(sys.argv),
        n_hands=n_hands,
        focus_full=focus_full,
        focus_ehs_full=focus_ehs_full,
        parallel=parallel,
        jobs=jobs,
        pairings=pairs,
    )
    focus_label = ""
    if focus_ehs_full:
        focus_label = "  (EHS + Full matchups only)"
    elif focus_full:
        focus_label = "  (FullAgent matchups only)"
    section(
        f"Round-Robin Benchmark — {len(pairs)} pairings × {n_hands:,} hands"
        + focus_label
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

    if save_results is not None:
        save_report(resolve_results_path(save_results), config, results)

    if total_errors:
        sys.exit(1)


def _print_raw_table(results: list[PairingResult]) -> None:
    section("Raw Results")
    for line in _format_raw_table(results)[4:]:
        print(line)


def _print_matrix(results: list[PairingResult]) -> None:
    section("Win-Rate Matrix (mbb/hand)")
    for line in _format_matrix(results)[4:]:
        print(line)


def _print_full_diagnostics(results: list[PairingResult]) -> None:
    lines = _format_full_diagnostics(results)
    if not lines:
        return
    section("FullAgent Opponent-Model Diagnostics")
    for line in lines[4:]:
        print(line)


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
    parser.add_argument("--focus-ehs-full", action="store_true",
                        help="Run EHS and Full against each other and all other agents")
    parser.add_argument("--save-results", nargs="?", const="", default=None, metavar="PATH",
                        help="Save a readable report (default: results/behavior-agent-performance/run-<timestamp>.txt)")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
        return

    run_benchmark(
        n_hands=args.hands,
        focus_full=args.focus_full,
        focus_ehs_full=args.focus_ehs_full,
        parallel=args.parallel,
        jobs=args.jobs,
        save_results=args.save_results,
    )


if __name__ == "__main__":
    main()
