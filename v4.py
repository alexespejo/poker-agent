"""Milestone 4 test script — hyperparameter tuning, evaluation, and plots.

Usage:
  python3 v4.py                  # full pipeline (~25 min)
  python3 v4.py --parallel --jobs 8   # faster grid + eval (multiprocessing)
  python3 v4.py --skip-grid --n-samples 500 --raise-threshold 0.15
  python3 v4.py --selfplay       # also run FullAgent vs FullAgent moonshot
  python3 v4.py --no-plots       # skip matplotlib output
  python3 v4.py --grid-hands 500 # faster grid search for testing
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))

from poker_agent.agents.ehs_agent import EHSAgent
from poker_agent.agents.full_agent import FullAgent
from poker_agent.agents.random_agent import RandomAgent
from poker_agent.agents.rule_based_agent import RuleBasedAgent
from poker_agent.game import PokerGame
from poker_agent.simulation import SimResults, run_simulation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
STACK_SIZE = 1000
BIG_BLIND = 10
ACTIONS = ("fold", "call", "check", "raise")
STREETS = ("preflop", "flop", "turn", "river")
TARGETS = {"RandomAgent": 100, "RuleBasedAgent": 50, "EHSAgent": 20}

GRID_CONFIGS = [
    (n_samples, raise_threshold)
    for n_samples in (200, 500, 1000)
    for raise_threshold in (0.10, 0.15, 0.20)
]


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


@dataclass
class GridSearchResult:
    n_samples: int
    raise_threshold: float
    mbb_per_hand: float


def _default_jobs(n_tasks: int, jobs: int | None) -> int:
    if jobs is not None:
        return max(1, jobs)
    return min(os.cpu_count() or 1, n_tasks)


def run_full_agent_simulation(
    full_agent,
    opponent,
    n_hands,
    *,
    show_progress=True,
    collect_hero_ehs=False,
):
    """Run a simulation and flush the FullAgent opponent model afterward."""
    results = run_simulation(
        full_agent,
        opponent,
        n_hands=n_hands,
        stack_size=STACK_SIZE,
        big_blind=BIG_BLIND,
        show_progress=show_progress,
        collect_hero_ehs=collect_hero_ehs,
    )
    full_agent.finalize_session()
    return results


def _make_opponent(name: str, n_samples: int, raise_threshold: float):
    if name == "RandomAgent":
        return RandomAgent()
    if name == "RuleBasedAgent":
        return RuleBasedAgent()
    if name == "EHSAgent":
        return EHSAgent(n_samples=n_samples, raise_threshold=raise_threshold)
    raise ValueError(f"Unknown opponent: {name}")


def _grid_search_worker(args: tuple[int, float, int]) -> GridSearchResult:
    n_samples, raise_threshold, hands_per_config = args
    agent = FullAgent(n_samples=n_samples, raise_threshold=raise_threshold)
    sim = run_full_agent_simulation(
        agent, RuleBasedAgent(), hands_per_config, show_progress=False
    )
    return GridSearchResult(n_samples, raise_threshold, sim.mbb_per_hand_agent0)


def _eval_worker(
    args: tuple[str, int, float, int, bool],
) -> tuple[str, SimResults]:
    name, n_samples, raise_threshold, n_hands, collect_ehs = args
    agent = FullAgent(n_samples=n_samples, raise_threshold=raise_threshold)
    opponent = _make_opponent(name, n_samples, raise_threshold)
    sim = run_full_agent_simulation(
        agent, opponent, n_hands, show_progress=False, collect_hero_ehs=collect_ehs
    )
    return name, sim


def _print_ehs_summary(results: SimResults) -> None:
    for street in STREETS:
        w = results.win_ehs_by_street.get(street, [])
        l = results.lose_ehs_by_street.get(street, [])
        w_avg = sum(w) / len(w) if w else float("nan")
        l_avg = sum(l) / len(l) if l else float("nan")
        print(f"    {street:8s}  win EHS={w_avg:.3f} (n={len(w)})  "
              f"lose EHS={l_avg:.3f} (n={len(l)})")


def grid_search(
    hands_per_config: int = 3_000,
    *,
    parallel: bool = False,
    jobs: int | None = None,
) -> tuple[tuple[int, float], list[GridSearchResult]]:
    """Search n_samples and raise_threshold; return best config and all results."""
    configs = GRID_CONFIGS
    results: list[GridSearchResult] = []
    best_config = configs[0]
    best_mbb = float("-inf")

    print(f"\n  {'n_samples':>10}  {'raise_thr':>10}  {'mbb/hand':>10}")
    print(f"  {'-' * 34}")

    if parallel:
        n_workers = _default_jobs(len(configs), jobs)
        print(f"  Running {len(configs)} configs in parallel ({n_workers} workers)...")
        tasks = [
            (n_samples, raise_threshold, hands_per_config)
            for n_samples, raise_threshold in configs
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for gs_result in pool.map(_grid_search_worker, tasks):
                results.append(gs_result)
                print(f"  {gs_result.n_samples:>10}  {gs_result.raise_threshold:>10.2f}  "
                      f"{gs_result.mbb_per_hand:>+10.1f}")
                if gs_result.mbb_per_hand > best_mbb:
                    best_mbb = gs_result.mbb_per_hand
                    best_config = (gs_result.n_samples, gs_result.raise_threshold)
    else:
        for i, (n_samples, raise_threshold) in enumerate(configs, 1):
            print(f"  Config {i}/{len(configs)}: n_samples={n_samples}, "
                  f"raise_threshold={raise_threshold}")
            agent = FullAgent(n_samples=n_samples, raise_threshold=raise_threshold)
            sim = run_full_agent_simulation(
                agent, RuleBasedAgent(), hands_per_config, show_progress=False
            )
            mbb = sim.mbb_per_hand_agent0
            results.append(GridSearchResult(n_samples, raise_threshold, mbb))
            print(f"  {n_samples:>10}  {raise_threshold:>10.2f}  {mbb:>+10.1f}")

            if mbb > best_mbb:
                best_mbb = mbb
                best_config = (n_samples, raise_threshold)

    return best_config, results


def run_evaluations(
    n_samples: int,
    raise_threshold: float,
    n_hands: int,
    *,
    parallel: bool = False,
    jobs: int | None = None,
) -> dict[str, SimResults]:
    """Run FullAgent vs each baseline opponent."""
    opponent_names = ("RandomAgent", "RuleBasedAgent", "EHSAgent")
    evals: dict[str, SimResults] = {}

    if parallel:
        n_workers = _default_jobs(len(opponent_names), jobs)
        print(f"  Running {len(opponent_names)} evaluations in parallel "
              f"({n_workers} workers)...")
        tasks = [
            (name, n_samples, raise_threshold, n_hands, name == "RuleBasedAgent")
            for name in opponent_names
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_eval_worker, t): t[0] for t in tasks}
            for future in as_completed(futures):
                name, sim = future.result()
                evals[name] = sim
                print(f"    {name}: mbb/hand {sim.mbb_per_hand_agent0:+.1f}  "
                      f"errors: {sim.errors}")
    else:
        for name in opponent_names:
            print(f"\n  FullAgent vs {name}...")
            agent = FullAgent(n_samples=n_samples, raise_threshold=raise_threshold)
            collect_ehs = name == "RuleBasedAgent"
            evals[name] = run_full_agent_simulation(
                agent,
                _make_opponent(name, n_samples, raise_threshold),
                n_hands,
                collect_hero_ehs=collect_ehs,
            )
            print(f"    mbb/hand: {evals[name].mbb_per_hand_agent0:+.1f}  "
                  f"errors: {evals[name].errors}")

    if "RuleBasedAgent" in evals:
        print("\n  EHS by street (from RuleBased evaluation):")
        _print_ehs_summary(evals["RuleBasedAgent"])

    return evals


def generate_plots(mbb_scores, evals, win_ehs, lose_ehs):
    """Save milestone 4 result plots to results/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)

    opponents = list(mbb_scores.keys())
    values = [mbb_scores[o] for o in opponents]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
    bars = ax.bar(opponents, values, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (5 if val >= 0 else -15),
            f"{val:+.0f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=10,
            fontweight="bold",
        )
    for opp, target in TARGETS.items():
        if opp in mbb_scores:
            idx = opponents.index(opp)
            ax.hlines(target, idx - 0.4, idx + 0.4, colors="orange", linestyles="--", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("mbb/hand")
    ax.set_title("FullAgent Performance vs Baselines")
    ax.set_ylim(min(min(values) - 50, -50), max(max(values) + 100, 150))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "mbb_comparison.png"), dpi=150)
    plt.close(fig)

    chip_history = evals["RandomAgent"].chip_history
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(chip_history) + 1), chip_history, color="#3498db", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Hand")
    ax.set_ylabel("Cumulative Net Chips (FullAgent)")
    ax.set_title("Chip History — FullAgent vs RandomAgent")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "chip_history_vs_random.png"), dpi=150)
    plt.close(fig)

    x = list(range(len(STREETS)))
    width = 0.35
    win_avgs = [
        sum(win_ehs.get(s, [])) / len(win_ehs[s]) if win_ehs.get(s) else 0.0
        for s in STREETS
    ]
    lose_avgs = [
        sum(lose_ehs.get(s, [])) / len(lose_ehs[s]) if lose_ehs.get(s) else 0.0
        for s in STREETS
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], win_avgs, width, label="Winning hands", color="#2ecc71")
    ax.bar([i + width / 2 for i in x], lose_avgs, width, label="Losing hands", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(STREETS)
    ax.set_ylabel("Average EHS")
    ax.set_title("Average EHS by Street — Winning vs Losing Hands")
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ehs_by_round.png"), dpi=150)
    plt.close(fig)

    action_results = {name: evals[name].action_counts_agent0 for name in evals}
    opponents = list(action_results.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = [0.0] * len(opponents)
    colors = {"fold": "#e74c3c", "call": "#f39c12", "check": "#3498db", "raise": "#2ecc71"}
    for action in ACTIONS:
        counts = [action_results[opp].get(action, 0) for opp in opponents]
        totals = [sum(action_results[opp].values()) or 1 for opp in opponents]
        pcts = [c / t * 100 for c, t in zip(counts, totals)]
        ax.bar(opponents, pcts, bottom=bottoms, label=action, color=colors[action])
        bottoms = [b + p for b, p in zip(bottoms, pcts)]
    ax.set_ylabel("Action Frequency (%)")
    ax.set_title("FullAgent Action Distribution vs Each Opponent")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "action_distribution.png"), dpi=150)
    plt.close(fig)

    for fname in (
        "mbb_comparison.png",
        "chip_history_vs_random.png",
        "ehs_by_round.png",
        "action_distribution.png",
    ):
        print(f"  Saved {os.path.join(RESULTS_DIR, fname)}")


def run_selfplay(n_hands=10_000, n_samples=500, raise_threshold=0.15, window=500):
    """Optional moonshot: FullAgent vs FullAgent convergence check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)

    agent0 = FullAgent(n_samples=n_samples, raise_threshold=raise_threshold)
    agent1 = FullAgent(n_samples=n_samples, raise_threshold=raise_threshold)
    agents = [agent0, agent1]
    game = PokerGame(stack_size=STACK_SIZE, big_blind=BIG_BLIND)

    chip_deltas = []
    net = 0
    print(f"  Running {n_hands:,}-hand self-play...")

    for hand_num in range(n_hands):
        dealer = hand_num % 2
        state = game.reset(dealer=dealer)
        done = False

        while not done:
            p = state.current_player
            legal = game.legal_actions(state)
            action, amount = agents[p].act(state, p)
            if action not in legal:
                action = legal[0]
                amount = 0
            state, rewards, done = game.step(action, amount)

        agent0.finalize_session()
        agent1.finalize_session()
        chip_deltas.append(rewards[0])
        net += rewards[0]

        if (hand_num + 1) % 2000 == 0:
            running_mbb = (net / (hand_num + 1) / BIG_BLIND) * 1000
            print(f"    Hand {hand_num + 1:,}: running mbb/hand = {running_mbb:+.1f}")

    overall_mbb = (net / n_hands / BIG_BLIND) * 1000
    rolling = []
    window_deltas = []
    cumulative = 0
    for i, delta in enumerate(chip_deltas):
        cumulative += delta
        window_deltas.append(delta)
        if len(window_deltas) > window:
            window_deltas.pop(0)
        hands_in_window = i + 1
        if hands_in_window >= window:
            rolling.append((sum(window_deltas) / window / BIG_BLIND) * 1000)
        else:
            rolling.append((cumulative / hands_in_window / BIG_BLIND) * 1000)

    last_2000_avg = sum(rolling[-2000:]) / len(rolling[-2000:])
    converged = abs(last_2000_avg) < 20

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, n_hands + 1), rolling, color="#9b59b6", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(20, color="orange", linewidth=0.8, linestyle=":")
    ax.axhline(-20, color="orange", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Hand")
    ax.set_ylabel(f"Rolling {window}-Hand mbb/hand")
    ax.set_title("FullAgent Self-Play Convergence")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "selfplay_convergence.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n  Overall mbb/hand (agent0): {overall_mbb:+.1f}")
    print(f"  Last 2,000-hand avg rolling mbb/hand: {last_2000_avg:+.1f}")
    print(f"  Convergence (|avg| < 20): {'PASS' if converged else 'FAIL'}")
    print(f"  Plot saved: {out_path}")
    return converged


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4 — tuning, evaluation, plots")
    parser.add_argument("--selfplay", action="store_true", help="Run FullAgent self-play moonshot")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--grid-hands", type=int, default=3_000, help="Hands per grid-search config")
    parser.add_argument("--eval-hands", type=int, default=10_000, help="Hands per final evaluation")
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run grid search and evaluations in parallel (multiprocessing)",
    )
    parser.add_argument(
        "--jobs", type=int, default=None,
        help="Max parallel workers (default: min(CPU count, num tasks))",
    )
    parser.add_argument(
        "--skip-grid", action="store_true",
        help="Skip hyperparameter grid search (use --n-samples / --raise-threshold)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Monte Carlo samples when --skip-grid (default: 500)",
    )
    parser.add_argument(
        "--raise-threshold", type=float, default=None,
        help="Raise threshold when --skip-grid (default: 0.15)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Hyperparameter grid search
    # ------------------------------------------------------------------
    if args.skip_grid:
        best_n_samples = args.n_samples if args.n_samples is not None else 500
        best_raise_threshold = (
            args.raise_threshold if args.raise_threshold is not None else 0.15
        )
        section("Step 1: Grid Search Skipped")
        print(f"  Using n_samples={best_n_samples}, raise_threshold={best_raise_threshold:.2f}")
    else:
        section(f"Step 1: Hyperparameter Grid Search ({args.grid_hands:,} hands per config vs RuleBased)")
        best_config, _ = grid_search(
            hands_per_config=args.grid_hands,
            parallel=args.parallel,
            jobs=args.jobs,
        )
        best_n_samples, best_raise_threshold = best_config
        print(f"\n  Best config: n_samples={best_n_samples}, "
              f"raise_threshold={best_raise_threshold:.2f}")

    # ------------------------------------------------------------------
    # 2. Final evaluations (EHS tracking included for RuleBased)
    # ------------------------------------------------------------------
    section(f"Step 2: Final Evaluation ({args.eval_hands:,} hands each)")
    evals = run_evaluations(
        best_n_samples,
        best_raise_threshold,
        args.eval_hands,
        parallel=args.parallel,
        jobs=args.jobs,
    )
    total_errors = sum(r.errors for r in evals.values())
    mbb_scores = {name: r.mbb_per_hand_agent0 for name, r in evals.items()}

    rb_results = evals["RuleBasedAgent"]
    win_ehs = rb_results.win_ehs_by_street
    lose_ehs = rb_results.lose_ehs_by_street

    # ------------------------------------------------------------------
    # 3. Generate plots
    # ------------------------------------------------------------------
    plots_ok = True
    if args.no_plots:
        print("\n  (plots skipped — --no-plots)")
    else:
        section("Step 3: Generating Plots")
        try:
            generate_plots(mbb_scores, evals, win_ehs, lose_ehs)
        except ImportError:
            plots_ok = False
            print("  matplotlib not installed — skipping plots")

    # ------------------------------------------------------------------
    # 4. Optional self-play moonshot
    # ------------------------------------------------------------------
    if args.selfplay:
        section("Step 4: Self-Play Moonshot (FullAgent vs FullAgent)")
        try:
            run_selfplay(
                n_hands=args.eval_hands,
                n_samples=best_n_samples,
                raise_threshold=best_raise_threshold,
            )
        except ImportError:
            print("  matplotlib not installed — skipping self-play plot")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    section("Final Summary")
    print(f"\n  {'Opponent':<18} {'mbb/hand':>10} {'Target':>10} {'Status':>8}")
    print(f"  {'-' * 48}")
    for opp in ("RandomAgent", "RuleBasedAgent", "EHSAgent"):
        mbb = mbb_scores[opp]
        target = TARGETS[opp]
        status = "PASS" if mbb > target else "FAIL"
        print(f"  {opp:<18} {mbb:>+10.1f} {target:>+10} {status:>8}")

    success = total_errors == 0 and (plots_ok or args.no_plots)

    print(f"""
============================
MILESTONE 4 {'COMPLETE' if success else 'INCOMPLETE'}
Best config: n_samples={best_n_samples}, raise_threshold={best_raise_threshold:.2f}
Hands run: {args.eval_hands:,} per pairing
Key metrics:
  vs RandomAgent:     {mbb_scores['RandomAgent']:+.1f} mbb/hand
  vs RuleBasedAgent:  {mbb_scores['RuleBasedAgent']:+.1f} mbb/hand
  vs EHSAgent:        {mbb_scores['EHSAgent']:+.1f} mbb/hand
Plots saved to: {RESULTS_DIR}/{'  (skipped)' if args.no_plots else ''}
============================
""")

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
