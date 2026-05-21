"""Milestone 1 test script — environment, hand evaluator, baseline agents."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from poker_agent.hand_eval import run_unit_tests
from poker_agent.game import PokerGame
from poker_agent.agents.rule_based_agent import RuleBasedAgent
from poker_agent.simulation import run_simulation


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Hand evaluator unit tests
    # ------------------------------------------------------------------
    section("Step 1: Hand Evaluator Unit Tests")
    results = run_unit_tests()
    all_pass = True
    for desc, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")
        if not passed:
            all_pass = False
    print(f"\n  Result: {'ALL PASS' if all_pass else 'SOME FAILURES'} "
          f"({sum(p for _, p in results)}/{len(results)})")

    # ------------------------------------------------------------------
    # 2. Manual trace of 5 hands (random vs random)
    # ------------------------------------------------------------------
    section("Step 2: Manual Trace — 5 Hands (RuleBased vs RuleBased)")
    game = PokerGame(stack_size=1000, big_blind=10)
    rand0 = RuleBasedAgent()
    rand1 = RuleBasedAgent()

    for hand_num in range(5):
        print(f"\n  --- Hand {hand_num + 1} (dealer=player {hand_num % 2}) ---")
        state = game.reset(dealer=hand_num % 2)
        hole_p0 = ' '.join(str(c) for c in state.hole_cards[0])
        hole_p1 = ' '.join(str(c) for c in state.hole_cards[1])
        print(f"  Hole cards: P0=[{hole_p0}]  P1=[{hole_p1}]")
        print(f"  Stacks: {state.stacks}  Pot: {state.pot}  Street: {state.street}")

        done = False
        step = 0
        while not done:
            p = state.current_player
            agents = [rand0, rand1]
            action, amount = agents[p].act(state, p)
            legal = game.legal_actions(state)
            if action not in legal:
                action = legal[0]; amount = 0
            community = ' '.join(str(c) for c in state.community_cards) or '(none)'
            print(f"    [{state.street}] P{p} acts: {action}"
                  + (f" {amount}" if action == "raise" else "")
                  + f"  | community: {community}  pot: {state.pot}")
            state, rewards, done = game.step(action, amount)
            step += 1

        community = ' '.join(str(c) for c in state.community_cards) or '(none)'
        print(f"  Hand over. Community: [{community}]  Rewards: {rewards}")

    # ------------------------------------------------------------------
    # 3. 10,000-hand simulation: Random vs Random
    # ------------------------------------------------------------------
    section("Step 3: 10,000-Hand Simulation — RuleBased vs RuleBased")
    print("  Running... (this may take a few seconds)")
    results_sim = run_simulation(RuleBasedAgent(), RuleBasedAgent(), n_hands=10_000,
                                 stack_size=1000, big_blind=10)

    print(f"\n  Hands completed : {results_sim.hands_played:,}")
    print(f"  Errors          : {results_sim.errors}")
    print(f"  mbb/hand P0     : {results_sim.mbb_per_hand_agent0:+.1f}")
    print(f"  mbb/hand P1     : {results_sim.mbb_per_hand_agent1:+.1f}")
    print(f"\n  Action distribution — Player 0 (RuleBasedAgent):")
    total0 = sum(results_sim.action_counts_agent0.values()) or 1
    for act, cnt in results_sim.action_counts_agent0.items():
        print(f"    {act:8s}: {cnt:6,}  ({cnt/total0*100:.1f}%)")
    print(f"\n  Action distribution — Player 1 (RuleBasedAgent):")
    total1 = sum(results_sim.action_counts_agent1.values()) or 1
    for act, cnt in results_sim.action_counts_agent1.items():
        print(f"    {act:8s}: {cnt:6,}  ({cnt/total1*100:.1f}%)")

    # ------------------------------------------------------------------
    # 4. Milestone verdict
    # ------------------------------------------------------------------
    success = (results_sim.errors == 0 and all_pass
               and results_sim.hands_played == 10_000)

    print(f"""
============================
MILESTONE 1 {'COMPLETE' if success else 'INCOMPLETE'}
Unit tests:  {'ALL PASS' if all_pass else 'FAILURES DETECTED'}
Hands run:   {results_sim.hands_played:,}
Errors:      {results_sim.errors}
mbb/hand P0: {results_sim.mbb_per_hand_agent0:+.1f}
============================
""")

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
