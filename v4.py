"""Milestone 4 test script — hyperparameter tuning, evaluation, and plots.

Usage:
  python3 v4.py                  # full pipeline (~25 min)
  python3 v4.py --selfplay       # also run FullAgent vs FullAgent moonshot
  python3 v4.py --no-plots       # skip matplotlib output
  python3 v4.py --grid-hands 500 # faster grid search for testing
"""

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))

from poker_agent.agents.ehs_agent import EHSAgent
from poker_agent.agents.full_agent import FullAgent
from poker_agent.agents.random_agent import RandomAgent
from poker_agent.agents.rule_based_agent import RuleBasedAgent
from poker_agent.game import PokerGame
from poker_agent.simulation import run_simulation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
STACK_SIZE = 1000
BIG_BLIND = 10
ACTIONS = ("fold", "call", "check", "raise")
STREETS = ("preflop", "flop", "turn", "river")
TARGETS = {"RandomAgent": 100, "RuleBasedAgent": 50, "EHSAgent": 20}


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


@dataclass
class GridSearchResult:
    n_samples: int
    raise_threshold: float
    mbb_per_hand: float


def run_full_agent_simulation(full_agent, opponent, n_hands, *, show_progress=True):
    """Run a simulation and flush the FullAgent opponent model afterward."""
    results = run_simulation(
        full_agent,
        opponent,
        n_hands=n_hands,
        stack_size=STACK_SIZE,
        big_blind=BIG_BLIND,
        show_progress=show_progress,
    )
    full_agent.finalize_session()
    return results


def grid_search(hands_per_config=3_000):
    """Search n_samples and raise_threshold; return best config and all results."""
    configs = [
        (n_samples, raise_threshold)
        for n_samples in (200, 500, 1000)
        for raise_threshold in (0.10, 0.15, 0.20)
    ]
    results = []
    best_config = configs[0]
    best_mbb = float("-inf")

    print(f"\n  {'n_samples':>10}  {'raise_thr':>10}  {'mbb/hand':>10}")
    print(f"  {'-' * 34}")

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


def collect_ehs_by_outcome(full_agent, opponent, n_hands):
    """Track EHS by street for winning vs losing hands (player 0)."""
    game = PokerGame(stack_size=STACK_SIZE, big_blind=BIG_BLIND)
    agents = [full_agent, opponent]
    win_ehs = defaultdict(list)
    lose_ehs = defaultdict(list)

    for hand_num in range(n_hands):
        dealer = hand_num % 2
        state = game.reset(dealer=dealer)
        hand_records = []
        done = False

        while not done:
            p = state.current_player
            legal = game.legal_actions(state)
            action, amount = agents[p].act(state, p)
            if action not in legal:
                action = legal[0]
                amount = 0

            if p == 0:
                hand_records.append((state.street, full_agent.last_decision["ehs"]))

            state, rewards, done = game.step(action, amount)

        full_agent.finalize_session()

        if rewards[0] > 0:
            target = win_ehs
        elif rewards[0] < 0:
            target = lose_ehs
        else:
            continue

        for street, ehs in hand_records:
            target[street].append(ehs)

    return win_ehs, lose_ehs


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
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Hyperparameter grid search
    # ------------------------------------------------------------------
    section(f"Step 1: Hyperparameter Grid Search ({args.grid_hands:,} hands per config vs RuleBased)")
    best_config, _ = grid_search(hands_per_config=args.grid_hands)
    best_n_samples, best_raise_threshold = best_config
    print(f"\n  Best config: n_samples={best_n_samples}, "
          f"raise_threshold={best_raise_threshold:.2f}")

    def make_full_agent():
        return FullAgent(n_samples=best_n_samples, raise_threshold=best_raise_threshold)

    # ------------------------------------------------------------------
    # 2. Final evaluations
    # ------------------------------------------------------------------
    section(f"Step 2: Final Evaluation ({args.eval_hands:,} hands each)")
    evals = {}
    pairings = [
        ("RandomAgent", RandomAgent()),
        ("RuleBasedAgent", RuleBasedAgent()),
        ("EHSAgent", EHSAgent(n_samples=best_n_samples, raise_threshold=best_raise_threshold)),
    ]

    total_errors = 0
    for name, opponent in pairings:
        print(f"\n  FullAgent vs {name}...")
        agent = make_full_agent()
        evals[name] = run_full_agent_simulation(agent, opponent, n_hands=args.eval_hands)
        total_errors += evals[name].errors
        print(f"    mbb/hand: {evals[name].mbb_per_hand_agent0:+.1f}  "
              f"errors: {evals[name].errors}")

    mbb_scores = {name: r.mbb_per_hand_agent0 for name, r in evals.items()}

    # ------------------------------------------------------------------
    # 3. EHS data for plots
    # ------------------------------------------------------------------
    section(f"Step 3: EHS Win/Loss Tracking ({args.eval_hands:,} hands vs RuleBased)")
    print("  Running logged simulation...")
    ehs_agent = make_full_agent()
    win_ehs, lose_ehs = collect_ehs_by_outcome(
        ehs_agent, RuleBasedAgent(), n_hands=args.eval_hands
    )
    for street in STREETS:
        w = win_ehs.get(street, [])
        l = lose_ehs.get(street, [])
        w_avg = sum(w) / len(w) if w else float("nan")
        l_avg = sum(l) / len(l) if l else float("nan")
        print(f"    {street:8s}  win EHS={w_avg:.3f} (n={len(w)})  "
              f"lose EHS={l_avg:.3f} (n={len(l)})")

    # ------------------------------------------------------------------
    # 4. Generate plots
    # ------------------------------------------------------------------
    plots_ok = True
    if args.no_plots:
        print("\n  (plots skipped — --no-plots)")
    else:
        section("Step 4: Generating Plots")
        try:
            generate_plots(mbb_scores, evals, win_ehs, lose_ehs)
        except ImportError:
            plots_ok = False
            print("  matplotlib not installed — skipping plots")

    # ------------------------------------------------------------------
    # 5. Optional self-play moonshot
    # ------------------------------------------------------------------
    selfplay_ok = True
    if args.selfplay:
        section("Step 5: Self-Play Moonshot (FullAgent vs FullAgent)")
        try:
            selfplay_ok = run_selfplay(
                n_hands=args.eval_hands,
                n_samples=best_n_samples,
                raise_threshold=best_raise_threshold,
            )
        except ImportError:
            selfplay_ok = False
            print("  matplotlib not installed — skipping self-play plot")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    section("Final Summary")
    print(f"\n  {'Opponent':<18} {'mbb/hand':>10} {'Target':>10} {'Status':>8}")
    print(f"  {'-' * 48}")
    target_results = {}
    for opp in ("RandomAgent", "RuleBasedAgent", "EHSAgent"):
        mbb = mbb_scores[opp]
        target = TARGETS[opp]
        status = "PASS" if mbb > target else "FAIL"
        target_results[opp] = status == "PASS"
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
