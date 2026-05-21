"""Milestone 2 test script — Monte Carlo EHS agent vs Random agent."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from poker_agent.card import Card, Rank, Suit
from poker_agent.monte_carlo import estimate_ehs
from poker_agent.agents.ehs_agent import EHSAgent
from poker_agent.agents.rule_based_agent import RuleBasedAgent
from poker_agent.game import PokerGame, GameState
from poker_agent.simulation import run_simulation
from poker_agent.stats import SessionLogger


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def make_cards(*specs: str) -> list[Card]:
    """Build cards from strings like 'Ah', '2c'."""
    rank_map = {
        '2': Rank.TWO, '3': Rank.THREE, '4': Rank.FOUR, '5': Rank.FIVE,
        '6': Rank.SIX, '7': Rank.SEVEN, '8': Rank.EIGHT, '9': Rank.NINE,
        'T': Rank.TEN, 'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING,
        'A': Rank.ACE,
    }
    suit_map = {'c': Suit.CLUBS, 'd': Suit.DIAMONDS, 'h': Suit.HEARTS, 's': Suit.SPADES}
    return [Card(rank_map[s[:-1]], suit_map[s[-1]]) for s in specs]


def run_logged_simulation(
    ehs_agent: EHSAgent,
    opponent: RuleBasedAgent,
    n_hands: int,
    big_blind: int = 10,
    stack_size: int = 1000,
) -> tuple[SessionLogger, float]:
    """Run simulation with per-action logging; EHS agent is player 0."""
    game = PokerGame(stack_size=stack_size, big_blind=big_blind)
    logger = SessionLogger()
    agents = [ehs_agent, opponent]
    net_chips = [0, 0]

    for hand_num in range(n_hands):
        dealer = hand_num % 2
        state = game.reset(dealer=dealer)
        done = False

        while not done:
            p = state.current_player
            legal = game.legal_actions(state)

            if p == 0:
                # EHS agent — capture EHS for logging
                hole = state.hole_cards[p]
                community = state.community_cards
                dead = hole + community
                ehs_val = estimate_ehs(hole, community, dead, ehs_agent.n_samples)
                action, amount = ehs_agent._decide(
                    ehs_val,
                    state.current_bet / (state.pot + state.current_bet) if state.current_bet > 0 else 0.0,
                    state.current_bet,
                    state.pot,
                    state.stacks[p],
                    state.min_raise,
                    legal,
                )
            else:
                ehs_val = None
                action, amount = agents[p].act(state, p)

            if action not in legal:
                action = legal[0]
                amount = 0

            logger.log_action(
                hand_number=hand_num + 1,
                street=state.street,
                player=p,
                action=action,
                amount=amount,
                ehs=ehs_val,
                pot=state.pot,
                stacks=state.stacks,
            )

            state, rewards, done = game.step(action, amount)

        net_chips[0] += rewards[0]
        net_chips[1] += rewards[1]
        logger.record_hand_result(rewards)

    mbb = (net_chips[0] / n_hands / big_blind) * 1000
    return logger, mbb


def main() -> None:
    # ------------------------------------------------------------------
    # 1. EHS sanity checks
    # ------------------------------------------------------------------
    section("Step 1: EHS Sanity Checks (n_samples=2000)")

    aa = make_cards('Ah', 'As')
    ehs_aa = estimate_ehs(aa, [], aa, n_samples=2000)
    aa_ok = 0.80 <= ehs_aa <= 0.92
    print(f"  AA preflop EHS:  {ehs_aa:.3f}  (expect ~0.85)  {'✓' if aa_ok else '✗ OUT OF RANGE'}")

    trash = make_cards('7h', '2c')
    ehs_72 = estimate_ehs(trash, [], trash, n_samples=2000)
    trash_ok = 0.28 <= ehs_72 <= 0.45
    print(f"  72o preflop EHS: {ehs_72:.3f}  (expect ~0.35)  {'✓' if trash_ok else '✗ OUT OF RANGE'}")

    kk = make_cards('Kh', 'Ks')
    ehs_kk = estimate_ehs(kk, [], kk, n_samples=2000)
    print(f"  KK preflop EHS:  {ehs_kk:.3f}  (expect ~0.73)")

    # Postflop example: flush draw on flop
    hero = make_cards('Ah', '2h')
    board = make_cards('Kh', '7h', '3c')
    dead = hero + board
    ehs_fd = estimate_ehs(hero, board, dead, n_samples=2000)
    print(f"  Ah2h on Kh7h3c:  {ehs_fd:.3f}  (nut flush draw, expect 0.40–0.60)")

    sanity_ok = aa_ok and trash_ok

    # ------------------------------------------------------------------
    # 2. Pot-odds violation check (1,000 logged hands)
    # ------------------------------------------------------------------
    section("Step 2: Pot-Odds Violation Check (1,000 hands, n_samples=200)")
    ehs_check_agent = EHSAgent(n_samples=200)
    logger, _ = run_logged_simulation(ehs_check_agent, RuleBasedAgent(), n_hands=1_000)
    violations = logger.pot_odds_violations(player_id=0)
    violation_ok = violations == 0
    print(f"  Calls where EHS < pot_odds: {violations}  "
          f"({'✓ none — policy correct' if violation_ok else '✗ violations detected!'})")

    # ------------------------------------------------------------------
    # 3. 10,000-hand simulation: EHSAgent vs RuleBasedAgent
    # ------------------------------------------------------------------
    section("Step 3: 10,000-Hand Simulation — EHSAgent vs RuleBasedAgent (n_samples=500)")
    print("  Running... (may take ~60–90 seconds)")
    results = run_simulation(
        EHSAgent(n_samples=500),
        RuleBasedAgent(),
        n_hands=10_000,
        stack_size=1000,
        big_blind=10,
    )

    print(f"\n  Hands completed : {results.hands_played:,}")
    print(f"  Errors          : {results.errors}")
    print(f"  mbb/hand EHSAgent  (P0): {results.mbb_per_hand_agent0:+.1f}")
    print(f"  mbb/hand RuleBasedAgent(P1): {results.mbb_per_hand_agent1:+.1f}")

    print(f"\n  Action distribution — EHSAgent:")
    total = sum(results.action_counts_agent0.values()) or 1
    for act in ("fold", "call", "check", "raise"):
        cnt = results.action_counts_agent0.get(act, 0)
        print(f"    {act:8s}: {cnt:6,}  ({cnt/total*100:.1f}%)")

    ehs_positive = results.mbb_per_hand_agent0 > 0

    # ------------------------------------------------------------------
    # 4. Milestone verdict
    # ------------------------------------------------------------------
    success = sanity_ok and violation_ok and ehs_positive

    print(f"""
============================
MILESTONE 2 {'COMPLETE' if success else 'INCOMPLETE'}
EHS sanity (AA~0.85, 72o~0.35): {'PASS' if sanity_ok else 'FAIL'}
Pot-odds violations:            {violations} ({'PASS' if violation_ok else 'FAIL'})
EHSAgent mbb/hand vs Random:   {results.mbb_per_hand_agent0:+.1f} ({'PASS' if ehs_positive else 'FAIL — not positive'})
============================
""")

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
