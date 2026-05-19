"""Milestone 3 — FullAgent (EHS + opponent modeling) vs RuleBasedAgent.

Usage:
  python3 v3.py                 # visual demo (30 hands) then full benchmark
  python3 v3.py --no-visual     # benchmark only, no per-hand display
  python3 v3.py --hands 50      # visual demo with 50 hands
  python3 v3.py --delay 0.4     # seconds between actions (default 0.25)
"""

import sys
import os
import time
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from poker_agent.agents.rule_based_agent import RuleBasedAgent
from poker_agent.agents.ehs_agent import EHSAgent
from poker_agent.agents.full_agent import FullAgent
from poker_agent.game import PokerGame
from poker_agent.monte_carlo import estimate_ehs  # used in benchmark sanity check
from poker_agent.simulation import run_simulation

# ── ANSI colour helpers ────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BLUE   = "\033[34m"
_MAGENTA = "\033[35m"
_WHITE  = "\033[97m"
_BG_DARK = "\033[48;5;235m"   # very dark grey background

def G(s): return f"{_GREEN}{s}{_RESET}"
def R(s): return f"{_RED}{s}{_RESET}"
def Y(s): return f"{_YELLOW}{s}{_RESET}"
def C(s): return f"{_CYAN}{s}{_RESET}"
def B(s): return f"{_BLUE}{s}{_RESET}"
def M(s): return f"{_MAGENTA}{s}{_RESET}"
def DIM(s): return f"{_DIM}{s}{_RESET}"
def BOLD(s): return f"{_BOLD}{s}{_RESET}"

# ── Card display ───────────────────────────────────────────────────────────────
_SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
_SUIT_COLORS  = {"c": _GREEN, "d": _RED, "h": _RED, "s": _BLUE}

def fmt_card(card) -> str:
    """Format a Card with unicode suit and colour."""
    rank_str = str(card.rank)
    suit_char = str(card.suit)
    sym  = _SUIT_SYMBOLS[suit_char]
    col  = _SUIT_COLORS[suit_char]
    return f"{_BOLD}{col}{rank_str}{sym}{_RESET}"

def fmt_cards(cards) -> str:
    return "  ".join(fmt_card(c) for c in cards) if cards else DIM("—")

def fmt_hidden() -> str:
    return DIM("?♠  ?♠")

# ── Layout constants ───────────────────────────────────────────────────────────
W = 64   # total display width

def hline(char="─"): print(char * W)
def dline():          print("═" * W)

def _bar(label: str, value: str, width: int = W) -> None:
    """Print a key: value line padded to width."""
    content = f"  {BOLD(label)}: {value}"
    print(content)

# ── Visual simulation ──────────────────────────────────────────────────────────

def run_visual(
    n_hands: int = 30,
    delay: float = 0.25,
    stack_size: int = 1000,
    big_blind: int = 10,
) -> None:
    """Play n_hands visually in the terminal, printing each action."""

    full_agent = FullAgent(n_samples=300, raise_threshold=0.15)
    opponent   = RuleBasedAgent()
    agents     = [full_agent, opponent]
    game       = PokerGame(stack_size=stack_size, big_blind=big_blind)

    net = [0, 0]
    agent_names = [
        f"{_CYAN}FullAgent{_RESET}",
        f"{_MAGENTA}RuleBased{_RESET}",
    ]

    print()
    dline()
    print(BOLD(f"  MILESTONE 3 — Visual Simulation ({n_hands} hands)"))
    print(f"  {C('FullAgent')} (EHS + Opponent Model)  vs  {M('RuleBased')} (Chen heuristic)")
    dline()
    time.sleep(delay)

    for hand_idx in range(n_hands):
        dealer = hand_idx % 2
        state  = game.reset(dealer=dealer)
        done   = False
        actions_this_hand: list[str] = []
        current_street = "preflop"
        hand_ehs_vals: list[float] = []

        # ── Hand header ──────────────────────────────────────────────────
        print()
        hline()
        dealer_name = agent_names[dealer]
        print(
            f"  {BOLD(f'Hand {hand_idx+1}/{n_hands}')}  │  "
            f"Dealer: {dealer_name}  │  "
            f"Pot: {Y(str(state.pot))}"
        )
        hline()
        # Hole cards
        full_cards = fmt_cards(state.hole_cards[0])
        rule_cards = fmt_hidden()
        print(f"  {C('FullAgent')}  [{full_cards}]  stack {state.stacks[0]}")
        print(f"  {M('RuleBased')}  [{rule_cards}]  stack {state.stacks[1]}")
        hline()

        prev_community: list = []

        # ── Action loop ───────────────────────────────────────────────────
        while not done:
            p     = state.current_player
            legal = game.legal_actions(state)

            # Print street header when community cards change
            if state.community_cards != prev_community:
                print()
                new_cards = state.community_cards[len(prev_community):]
                print(f"  {BOLD(state.street.upper())}: {fmt_cards(new_cards)}")
                prev_community = list(state.community_cards)
                time.sleep(delay * 1.5)

            # Get action — use act() normally so model state stays consistent
            action, amount = agents[p].act(state, p)
            if action not in legal:
                action = legal[0]; amount = 0

            if p == 0:  # FullAgent — annotate with decision info
                d        = full_agent.last_decision
                raw_ehs  = d["ehs"]
                adj_ehs  = d["adjusted_ehs"]
                mult     = d["multiplier"]
                pot_odds = d["pot_odds"]
                hand_ehs_vals.append(adj_ehs)

                ehs_str  = C(f"EHS={raw_ehs:.2f}")
                adj_str  = (G if mult >= 0 else R)(f"adj={adj_ehs:.2f}")
                mult_str = (G if mult >= 0 else R)(f"({mult:+.2f})")
                odds_str = Y(f"odds={pot_odds:.2f}")

                if action == "raise":
                    act_str = G(f"raises {amount}")
                elif action == "call":
                    act_str = Y("calls")
                elif action == "check":
                    act_str = B("checks")
                else:
                    act_str = R("folds")

                print(
                    f"    {C('FullAgent')}  {act_str:<28}"
                    f"  {ehs_str}  {adj_str} {mult_str}  {odds_str}"
                )

            else:  # RuleBasedAgent
                if action == "raise":
                    act_str = M(f"raises {amount}")
                elif action == "call":
                    act_str = M("calls")
                elif action == "check":
                    act_str = DIM("checks")
                else:
                    act_str = R("folds")
                print(f"    {M('RuleBased')}  {act_str}")

            time.sleep(delay)
            state, rewards, done = game.step(action, amount)

        # ── Hand result ────────────────────────────────────────────────────
        net[0] += rewards[0]; net[1] += rewards[1]
        # Finalize this hand so model stats are current for the display below
        full_agent.finalize_session()

        rule_cards_revealed = fmt_cards(state.hole_cards[1])
        community_str = fmt_cards(state.community_cards) if state.community_cards else DIM("(none)")

        print()
        hline()
        if rewards[0] > 0:
            result_str = G(f"FullAgent wins  +{rewards[0]:,}")
        elif rewards[1] > 0:
            result_str = R(f"RuleBased wins  +{rewards[1]:,}")
        else:
            result_str = Y("Split pot")

        print(f"  {BOLD('RESULT')}: {result_str}")
        if state.community_cards:
            print(f"  Board:  {community_str}")
        print(f"  {M('RuleBased')} had: [{rule_cards_revealed}]")

        # Running stats
        hands_done = hand_idx + 1
        running_mbb = (net[0] / hands_done / big_blind) * 1000
        m = full_agent.opponent_model
        mbb_col = G if running_mbb >= 0 else R
        print()
        print(
            f"  {DIM('Running')}  mbb/hand: {mbb_col(f'{running_mbb:+.0f}')}"
            f"  net: {mbb_col(f'{net[0]:+,}')}"
        )
        print(
            f"  {DIM('OpModel')}  "
            f"VPIP={C(f'{m.vpip:.2f}')}  "
            f"PFR={C(f'{m.pfr:.2f}')}  "
            f"AF={C(f'{m.aggression_factor:.2f}')}  "
            f"hands={m.hands_seen}"
        )
        hline()
        time.sleep(delay * 2)

    dline()
    final_mbb = (net[0] / n_hands / big_blind) * 1000
    col = G if final_mbb >= 0 else R
    print(f"  Visual run complete — {n_hands} hands")
    print(f"  {C('FullAgent')} final mbb/hand: {col(f'{final_mbb:+.1f}')}")
    m = full_agent.opponent_model
    print(f"  Opponent model: VPIP={m.vpip:.3f}  PFR={m.pfr:.3f}  AF={m.aggression_factor:.3f}")
    dline()


# ── Benchmark helpers ──────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print('=' * W)


def run_with_model_checkpoints(
    full_agent: FullAgent,
    opponent,
    n_hands: int,
    checkpoints: list[int],
    stack_size: int = 1000,
    big_blind: int = 10,
) -> tuple[float, list[tuple[int, float, float, float]]]:
    """Run simulation collecting opponent model stats at checkpoint hand counts."""
    game = PokerGame(stack_size=stack_size, big_blind=big_blind)
    agents = [full_agent, opponent]
    net_chips = [0, 0]
    checkpoint_records: list[tuple[int, float, float, float]] = []
    next_cp = 0
    violations = 0

    _progress_interval = max(1, n_hands // 100)

    for hand_num in range(n_hands):
        dealer = hand_num % 2
        state  = game.reset(dealer=dealer)
        done   = False

        while not done:
            p      = state.current_player
            legal  = game.legal_actions(state)
            action, amount = agents[p].act(state, p)
            if action not in legal:
                action = legal[0]; amount = 0

            # Sanity check: no call where adjusted_ehs < pot_odds
            if p == 0 and action == "call" and state.current_bet > 0:
                hole  = state.hole_cards[p]
                comm  = state.community_cards
                ehs   = estimate_ehs(hole, comm, hole + comm, full_agent.n_samples)
                mult  = full_agent.opponent_model.get_range_multiplier()
                adj   = max(0.0, min(1.0, ehs + mult))
                odds  = state.current_bet / (state.pot + state.current_bet)
                if adj < odds:
                    violations += 1

            state, rewards, done = game.step(action, amount)

        net_chips[0] += rewards[0]
        net_chips[1] += rewards[1]

        current_hand = hand_num + 1
        if current_hand % _progress_interval == 0 or current_hand == n_hands:
            pct  = current_hand / n_hands * 100
            bar_filled = int(pct / 2)
            bar  = "█" * bar_filled + "░" * (50 - bar_filled)
            running_mbb = (net_chips[0] / current_hand / big_blind) * 1000
            mbb_col = _GREEN if running_mbb >= 0 else _RED
            print(
                f"\r  [{bar}] {pct:5.1f}%  hand {current_hand:>{len(str(n_hands))}}/{n_hands}"
                f"  mbb/hand: {mbb_col}{running_mbb:+.1f}{_RESET}   ",
                end="", flush=True,
            )

        while next_cp < len(checkpoints) and current_hand >= checkpoints[next_cp]:
            full_agent.finalize_session()
            m = full_agent.opponent_model
            checkpoint_records.append((
                checkpoints[next_cp], m.vpip, m.pfr, m.aggression_factor
            ))
            next_cp += 1

    print()  # end the progress line
    full_agent.finalize_session()
    mbb = (net_chips[0] / n_hands / big_blind) * 1000
    print(f"  Pot-odds violations (adj_ehs < pot_odds on call): {violations}")
    return mbb, checkpoint_records


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 3 — FullAgent vs RuleBasedAgent")
    parser.add_argument("--no-visual", action="store_true", help="Skip the visual demo")
    parser.add_argument("--hands",  type=int,   default=30,   help="Visual demo hand count")
    parser.add_argument("--delay",  type=float, default=0.25, help="Seconds between actions")
    args = parser.parse_args()

    # ── Visual demo ─────────────────────────────────────────────────────
    if not args.no_visual:
        run_visual(n_hands=args.hands, delay=args.delay)

    # ── Convergence check ────────────────────────────────────────────────
    section("Convergence Check: Opponent Model (1,000 hands vs RuleBasedAgent)")
    conv_agent = FullAgent(n_samples=200)
    _, records = run_with_model_checkpoints(
        conv_agent, RuleBasedAgent(), n_hands=1_000, checkpoints=[100, 500, 1_000]
    )
    print(f"\n  {'Hands':>6}  {'VPIP':>6}  {'PFR':>6}  {'AF':>6}")
    print(f"  {'-' * 30}")
    for hand_num, vpip, pfr, af in records:
        print(f"  {hand_num:>6}  {vpip:>6.3f}  {pfr:>6.3f}  {af:>6.3f}")

    # ── 10k: FullAgent vs RuleBased ───────────────────────────────────────
    section("10,000-Hand Benchmark — FullAgent vs RuleBasedAgent (n_samples=500)")
    full_agent = FullAgent(n_samples=500)
    full_mbb, _ = run_with_model_checkpoints(
        full_agent, RuleBasedAgent(), n_hands=10_000, checkpoints=[]
    )
    m = full_agent.opponent_model
    print(f"\n  FullAgent mbb/hand:  {full_mbb:+.1f}")
    print(f"  Final model — VPIP={m.vpip:.3f}  PFR={m.pfr:.3f}  "
          f"AF={m.aggression_factor:.3f}  FTR={m.fold_to_raise_rate:.3f}")

    # ── 10k: EHSAgent vs RuleBased (baseline) ────────────────────────────
    section("10,000-Hand Benchmark — EHSAgent vs RuleBasedAgent (n_samples=500)")
    ehs_results = run_simulation(
        EHSAgent(n_samples=500), RuleBasedAgent(), n_hands=10_000,
        stack_size=1000, big_blind=10,
    )
    ehs_mbb = ehs_results.mbb_per_hand_agent0
    print(f"\n  EHSAgent mbb/hand:   {ehs_mbb:+.1f}")

    # ── Summary ───────────────────────────────────────────────────────────
    section("Results Summary")
    print(f"\n  {'Agent':<22} {'mbb/hand':>10}")
    print(f"  {'-' * 34}")
    print(f"  {'FullAgent (EHS+Model)':<22} {full_mbb:>+10.1f}")
    print(f"  {'EHSAgent (no model)':<22} {ehs_mbb:>+10.1f}")
    print(f"  {'Improvement':<22} {full_mbb - ehs_mbb:>+10.1f}")

    full_beats_ehs = full_mbb > ehs_mbb
    print(f"""
============================
MILESTONE 3 {'COMPLETE' if full_beats_ehs else 'INCOMPLETE'}
FullAgent mbb/hand:   {full_mbb:+.1f}
EHSAgent  mbb/hand:   {ehs_mbb:+.1f}
FullAgent > EHSAgent: {'PASS' if full_beats_ehs else 'FAIL'}
============================
""")

    if not full_beats_ehs:
        sys.exit(1)


if __name__ == "__main__":
    main()
