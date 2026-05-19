"""Interactive heads-up poker — play against an AI agent.

Usage:
  python3 play.py                  # vs FullAgent (default)
  python3 play.py --agent rule     # vs RuleBasedAgent
  python3 play.py --agent ehs      # vs EHSAgent
  python3 play.py --agent full     # vs FullAgent (EHS + opponent model)
  python3 play.py --stack 500 --bb 5
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from poker_agent.game import PokerGame
from poker_agent.hand_eval import evaluate_hand
from poker_agent.agents.human_agent import HumanAgent, _fmt_card, _fmt_cards, _hline, _dline
from poker_agent.agents.rule_based_agent import RuleBasedAgent
from poker_agent.agents.ehs_agent import EHSAgent
from poker_agent.agents.full_agent import FullAgent

# ── ANSI ──────────────────────────────────────────────────────────────────────
_R     = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_GREEN = "\033[32m"
_RED   = "\033[31m"
_YELLOW= "\033[33m"
_CYAN  = "\033[36m"
_MAG   = "\033[35m"

def _g(s): return f"{_GREEN}{s}{_R}"
def _r(s): return f"{_RED}{s}{_R}"
def _y(s): return f"{_YELLOW}{s}{_R}"
def _c(s): return f"{_CYAN}{s}{_R}"
def _m(s): return f"{_MAG}{s}{_R}"
def _b(s): return f"{_BOLD}{s}{_R}"
def _dim(s): return f"{_DIM}{s}{_R}"

W = 64
AGENT_NAMES = {
    "rule": "RuleBasedAgent",
    "ehs":  "EHSAgent",
    "full": "FullAgent",
}

# ── Setup ─────────────────────────────────────────────────────────────────────

def build_ai(name: str, n_samples: int = 300):
    if name == "rule":
        return RuleBasedAgent()
    if name == "ehs":
        return EHSAgent(n_samples=n_samples)
    return FullAgent(n_samples=n_samples)


def print_banner(ai_name: str, stack: int, bb: int) -> None:
    print()
    _dline()
    print(_b(f"  POKER  —  You vs {ai_name}"))
    print(f"  Stack: {_y(str(stack))}  │  Big blind: {_y(str(bb))}")
    print(f"  Commands:  [f] fold  [c] call/check  [r] raise  [q] quit")
    _dline()


def print_hand_header(hand_num: int, human_id: int, state, ai_name: str) -> None:
    dealer_id = state.betting_history[0][0]
    your_pos   = "Dealer (SB)" if human_id == dealer_id else "Big Blind  "
    opp_pos    = "Big Blind  " if human_id == dealer_id else "Dealer (SB)"
    print()
    _dline()
    print(
        f"  {_b(f'Hand {hand_num}')}  │  "
        f"You: {_c(your_pos)}  │  "
        f"Opp ({_m(ai_name)}): {_dim(opp_pos)}"
    )
    _hline()
    print(f"  Your cards: [ {_fmt_cards(state.hole_cards[human_id])} ]")
    _hline()


def print_hand_result(
    state,
    rewards: list[int],
    human_id: int,
    net: list[int],
    hand_num: int,
    bb: int,
) -> None:
    opp = 1 - human_id
    human_reward = rewards[human_id]
    _hline()

    # Reveal result
    folded = any(a == "fold" for _, a, _ in state.betting_history)
    if folded:
        folder = next(p for p, a, _ in state.betting_history if a == "fold")
        if folder == human_id:
            print(f"  {_r('You folded.')}  {_m('Opponent')} wins {_y(str(state.pot))}")
        else:
            print(f"  {_m('Opponent')} folded.  {_c('You')} win {_y(str(state.pot))}!")
    else:
        # Showdown — reveal both hands
        community = state.community_cards
        your_cards = state.hole_cards[human_id]
        opp_cards  = state.hole_cards[opp]
        _, your_desc = evaluate_hand(your_cards + community)
        _, opp_desc  = evaluate_hand(opp_cards  + community)
        print(f"  {_b('SHOWDOWN')}")
        print(f"  {_c('You')}:      [ {_fmt_cards(your_cards)} ]  → {your_desc}")
        print(f"  {_m('Opponent')}: [ {_fmt_cards(opp_cards)}  ]  → {opp_desc}")
        if state.community_cards:
            print(f"  Board:     {_fmt_cards(state.community_cards)}")

    # Result line
    print()
    if human_reward > 0:
        result = _g(f"You WIN   +{human_reward:,} chips")
    elif human_reward < 0:
        result = _r(f"You LOSE   {human_reward:,} chips")
    else:
        result = _y("Split pot")
    print(f"  {result}")

    # Running totals
    mbb = (net[human_id] / hand_num / bb) * 1000
    col = _GREEN if mbb >= 0 else _RED
    print(
        f"  {_dim('Running')}  net: {col}{net[human_id]:+,}{_R} chips  │  "
        f"mbb/hand: {col}{mbb:+.0f}{_R}"
    )
    _dline()


# ── Main game loop ─────────────────────────────────────────────────────────────

def play(ai_name: str, stack: int, bb: int, n_samples: int) -> None:
    human  = HumanAgent()
    ai     = build_ai(ai_name, n_samples)
    game   = PokerGame(stack_size=stack, big_blind=bb)
    # Human is always player 0; dealer alternates each hand
    human_id = 0
    agents   = {human_id: human, 1 - human_id: ai}

    net      = [0, 0]
    hand_num = 0

    print_banner(AGENT_NAMES[ai_name], stack, bb)

    while True:
        # ── Start hand ───────────────────────────────────────────────────
        hand_num += 1
        dealer  = (hand_num - 1) % 2
        state   = game.reset(dealer=dealer)
        human.reset_hand()
        done    = False

        print_hand_header(hand_num, human_id, state, AGENT_NAMES[ai_name])

        # ── Action loop ──────────────────────────────────────────────────
        while not done:
            p = state.current_player

            if p == human_id:
                try:
                    action, amount = human.act(state, p)
                except (EOFError, KeyboardInterrupt):
                    print_summary(net, hand_num - 1, bb, human_id)
                    return
                # Let human quit mid-hand
                if isinstance(action, str) and action == "quit":
                    print_summary(net, hand_num - 1, bb, human_id)
                    return
            else:
                # AI acts silently; human_agent will show it on next turn
                action, amount = agents[p].act(state, p)

            legal = game.legal_actions(state)
            if action not in legal:
                action = legal[0]; amount = 0

            state, rewards, done = game.step(action, amount)

        # ── Hand over ────────────────────────────────────────────────────
        # Show any final opponent actions the human hasn't seen yet
        opp_actions = [
            e for e in state.betting_history[human._prev_history_len:]
            if e[0] != human_id and e[1] != "blind"
        ]
        for _, action, amount in opp_actions:
            label = _dim("Opponent")
            if action == "fold":   print(f"  {label} {_r('folds')}")
            elif action == "call": print(f"  {label} {_y('calls')}")
            elif action == "check":print(f"  {label} {_dim('checks')}")
            elif action == "raise":print(f"  {label} {_g(f'raises  (+{amount})')}")

        net[0] += rewards[0]
        net[1] += rewards[1]

        print_hand_result(state, rewards, human_id, net, hand_num, bb)

        # ── Continue prompt ───────────────────────────────────────────────
        try:
            cont = input("  Press Enter for next hand, or [q] to quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cont in ("q", "quit", "exit"):
            break

    print_summary(net, hand_num, bb, human_id)


def print_summary(net: list[int], hands: int, bb: int, human_id: int) -> None:
    if hands == 0:
        print("\n  No hands played.")
        return
    mbb = (net[human_id] / hands / bb) * 1000
    col = _GREEN if mbb >= 0 else _RED
    print()
    _dline()
    print(f"  {_b('Session over')} — {hands} hand{'s' if hands != 1 else ''}")
    print(f"  Net chips:  {col}{net[human_id]:+,}{_R}")
    print(f"  mbb/hand:   {col}{mbb:+.0f}{_R}")
    _dline()
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Play heads-up poker against an AI agent.")
    parser.add_argument("--agent",   choices=["rule", "ehs", "full"], default="full",
                        help="Which AI to face (default: full)")
    parser.add_argument("--stack",   type=int, default=1000, help="Starting stack (default 1000)")
    parser.add_argument("--bb",      type=int, default=10,   help="Big blind (default 10)")
    parser.add_argument("--samples", type=int, default=300,
                        help="Monte Carlo samples for EHS/Full agents (default 300)")
    args = parser.parse_args()

    try:
        play(args.agent, args.stack, args.bb, args.samples)
    except KeyboardInterrupt:
        print("\n  Goodbye.")


if __name__ == "__main__":
    main()
