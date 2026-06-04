"""Texas Hold'em game environment for heads-up play."""

from __future__ import annotations

from dataclasses import dataclass, field

from poker_agent.card import Card, Deck
from poker_agent.hand_eval import evaluate_hand


STREETS = ["preflop", "flop", "turn", "river"]


@dataclass
class GameState:
    """Immutable snapshot of game state passed to agents."""
    hole_cards: list[list[Card]]          # [player0_cards, player1_cards]
    community_cards: list[Card]
    pot: int
    stacks: list[int]
    current_player: int
    street: str
    current_bet: int                      # amount the current player must add to call
    min_raise: int
    betting_history: list[tuple[int, str, int]]  # (player, action, amount)
    is_done: bool
    big_blind: int
    # How much each player has put in this street (for raise tracking)
    street_investment: list[int] = field(default_factory=lambda: [0, 0])
    # Total chips wagered by each player this hand (for side-pot logic if extended)
    total_investment: list[int] = field(default_factory=lambda: [0, 0])
    # Number of non-blind actions taken so far this street (used to detect check-check)
    street_actions: int = 0
    # Monotonic hand counter (incremented on each reset())
    hand_id: int = 0


class PokerGame:
    """Heads-up Texas Hold'em simulator.

    Player 0 is always the dealer (small blind) at the start;
    dealer alternates each hand via reset(dealer=...).
    """

    def __init__(self, stack_size: int = 1000, big_blind: int = 10) -> None:
        self.stack_size = stack_size
        self.big_blind = big_blind
        self.small_blind = big_blind // 2
        self._state: GameState | None = None
        self._deck: Deck | None = None
        self._hand_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, dealer: int = 0) -> GameState:
        """Start a new hand; dealer posts small blind and acts first preflop."""
        self._hand_id += 1
        self._deck = Deck()
        self._deck.shuffle()

        stacks = [self.stack_size, self.stack_size]
        investment = [0, 0]

        # Post blinds
        sb_player = dealer
        bb_player = 1 - dealer

        sb_amount = min(self.small_blind, stacks[sb_player])
        stacks[sb_player] -= sb_amount
        investment[sb_player] += sb_amount

        bb_amount = min(self.big_blind, stacks[bb_player])
        stacks[bb_player] -= bb_amount
        investment[bb_player] += bb_amount

        pot = sb_amount + bb_amount

        # Deal hole cards
        hole_cards = [self._deck.deal(2), self._deck.deal(2)]

        # Preflop: dealer (SB) acts first in heads-up
        # current_bet for SB is the difference (BB - SB already posted)
        current_bet = bb_amount - sb_amount  # amount SB still needs to add to call

        self._state = GameState(
            hole_cards=hole_cards,
            community_cards=[],
            pot=pot,
            stacks=stacks,
            current_player=sb_player,
            street="preflop",
            current_bet=current_bet,
            min_raise=self.big_blind,
            betting_history=[
                (sb_player, "blind", sb_amount),
                (bb_player, "blind", bb_amount),
            ],
            is_done=False,
            big_blind=self.big_blind,
            street_investment=investment[:],
            total_investment=investment[:],
            hand_id=self._hand_id,
        )
        return self._state

    def legal_actions(self, state: GameState) -> list[str]:
        """Return the list of legal action strings for the current player."""
        p = state.current_player
        actions = []
        if state.current_bet > 0:
            actions.append("fold")
            if state.stacks[p] >= state.current_bet:
                actions.append("call")
            else:
                # Only option when can't call full amount is to go all-in (treated as call)
                actions.append("call")
        else:
            actions.append("check")

        # Can raise if we have chips beyond the call amount and can meet min_raise
        effective_stack = state.stacks[p] - state.current_bet
        if effective_stack >= state.min_raise:
            actions.append("raise")

        return actions

    def step(self, action: str, amount: int = 0) -> tuple[GameState, list[int], bool]:
        """Advance the game by one action.

        Returns (new_state, rewards, done).
        rewards[i] is the net chip change for player i (only meaningful when done=True).
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.is_done:
            raise RuntimeError("Hand is already over. Call reset().")

        state = self._state
        p = state.current_player
        opp = 1 - p

        legal = self.legal_actions(state)
        if action not in legal:
            raise ValueError(f"Illegal action '{action}'. Legal: {legal}")

        new_stacks = list(state.stacks)
        new_pot = state.pot
        new_investment = list(state.street_investment)
        new_total = list(state.total_investment)
        new_history = list(state.betting_history)

        if action == "fold":
            new_history.append((p, "fold", 0))
            rewards = [-state.total_investment[p], state.pot]
            # Winner collects the pot
            rewards[opp] = state.pot - state.total_investment[opp]
            rewards[p] = -state.total_investment[p]
            new_state = GameState(
                hole_cards=state.hole_cards,
                community_cards=state.community_cards,
                pot=new_pot,
                stacks=new_stacks,
                current_player=p,
                street=state.street,
                current_bet=state.current_bet,
                min_raise=state.min_raise,
                betting_history=new_history,
                is_done=True,
                big_blind=self.big_blind,
                street_investment=new_investment,
                total_investment=new_total,
                hand_id=state.hand_id,
            )
            self._state = new_state
            return new_state, self._compute_rewards(state, winner=opp), True

        if action == "check":
            new_history.append((p, "check", 0))
            return self._after_action(state, p, new_stacks, new_pot, new_investment,
                                      new_total, new_history, added=0, is_aggressive=False,
                                      new_street_actions=state.street_actions + 1)

        if action == "call":
            call_amount = min(state.current_bet, new_stacks[p])
            new_stacks[p] -= call_amount
            new_pot += call_amount
            new_investment[p] += call_amount
            new_total[p] += call_amount
            new_history.append((p, "call", call_amount))
            return self._after_action(state, p, new_stacks, new_pot, new_investment,
                                      new_total, new_history, added=call_amount, is_aggressive=False,
                                      new_street_actions=state.street_actions + 1)

        if action == "raise":
            # amount is the TOTAL raise size (how much more than call the raiser adds)
            # Validate and clamp
            min_r = state.min_raise
            max_r = new_stacks[p]  # all-in max
            raise_extra = max(min_r, min(amount, max_r))
            total_added = state.current_bet + raise_extra  # call + raise on top
            total_added = min(total_added, new_stacks[p])  # cap at stack (all-in)

            new_stacks[p] -= total_added
            new_pot += total_added
            new_investment[p] += total_added
            new_total[p] += total_added

            # New bet to call for opponent = what raiser invested this action - opp invested
            new_current_bet = new_investment[p] - new_investment[opp]
            new_min_raise = raise_extra  # min re-raise = size of this raise

            new_history.append((p, "raise", total_added))

            new_state = GameState(
                hole_cards=state.hole_cards,
                community_cards=state.community_cards,
                pot=new_pot,
                stacks=new_stacks,
                current_player=opp,
                street=state.street,
                current_bet=new_current_bet,
                min_raise=new_min_raise,
                betting_history=new_history,
                is_done=False,
                big_blind=self.big_blind,
                street_investment=new_investment,
                total_investment=new_total,
                street_actions=1,  # raiser acted; opponent must still respond
                hand_id=state.hand_id,
            )
            self._state = new_state
            return new_state, [0, 0], False

        raise ValueError(f"Unknown action: {action}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _after_action(
        self,
        state: GameState,
        acting_player: int,
        new_stacks: list[int],
        new_pot: int,
        new_investment: list[int],
        new_total: list[int],
        new_history: list[tuple[int, str, int]],
        added: int,
        is_aggressive: bool,
        new_street_actions: int = 0,
    ) -> tuple[GameState, list[int], bool]:
        """Decide whether to advance the street or continue betting.

        The street ends when:
          1. Both players' street investments are equal, AND
          2. Both players have acted at least once this street (street_actions >= 2).
        This correctly handles check-check (both players must check, not just one).
        """
        opp = 1 - acting_player

        investments_equal = new_investment[0] == new_investment[1]
        both_acted = new_street_actions >= 2
        street_done = investments_equal and both_acted

        if street_done:
            return self._advance_street(state, new_stacks, new_pot, new_investment,
                                        new_total, new_history)

        # Pass to opponent with updated street_actions counter
        new_state = GameState(
            hole_cards=state.hole_cards,
            community_cards=state.community_cards,
            pot=new_pot,
            stacks=new_stacks,
            current_player=opp,
            street=state.street,
            current_bet=new_investment[acting_player] - new_investment[opp],
            min_raise=self.big_blind,
            betting_history=new_history,
            is_done=False,
            big_blind=self.big_blind,
            street_investment=new_investment,
            total_investment=new_total,
            street_actions=new_street_actions,
            hand_id=state.hand_id,
        )
        self._state = new_state
        return new_state, [0, 0], False

    def _advance_street(
        self,
        state: GameState,
        new_stacks: list[int],
        new_pot: int,
        new_investment: list[int],
        new_total: list[int],
        new_history: list[tuple[int, str, int]],
    ) -> tuple[GameState, list[int], bool]:
        """Move to the next street or go to showdown."""
        current_idx = STREETS.index(state.street)

        if current_idx == 3:  # river — go to showdown
            return self._showdown(state, new_stacks, new_pot, new_total, new_history)

        next_street = STREETS[current_idx + 1]

        # Deal community cards
        new_community = list(state.community_cards)
        if next_street == "flop":
            new_community.extend(self._deck.deal(3))
        else:
            new_community.extend(self._deck.deal(1))

        # Check all-in: if either player has 0 chips, run out the board
        if new_stacks[0] == 0 or new_stacks[1] == 0:
            # Keep dealing until river, then showdown
            while len(new_community) < 5:
                new_community.extend(self._deck.deal(1))
            temp_state = GameState(
                hole_cards=state.hole_cards,
                community_cards=new_community,
                pot=new_pot,
                stacks=new_stacks,
                current_player=0,
                street="river",
                current_bet=0,
                min_raise=self.big_blind,
                betting_history=new_history,
                is_done=False,
                big_blind=self.big_blind,
                street_investment=[0, 0],
                total_investment=new_total,
                hand_id=state.hand_id,
            )
            return self._showdown(temp_state, new_stacks, new_pot, new_total, new_history)

        # Postflop: non-dealer (player 1 when dealer=0) acts first
        # We determine first-to-act by finding who was NOT the preflop dealer
        # The dealer is whoever posted the first blind in betting_history
        dealer = state.betting_history[0][0]
        first_to_act = 1 - dealer

        new_state = GameState(
            hole_cards=state.hole_cards,
            community_cards=new_community,
            pot=new_pot,
            stacks=new_stacks,
            current_player=first_to_act,
            street=next_street,
            current_bet=0,
            min_raise=self.big_blind,
            betting_history=new_history,
            is_done=False,
            big_blind=self.big_blind,
            street_investment=[0, 0],
            total_investment=new_total,
            street_actions=0,
            hand_id=state.hand_id,
        )
        self._state = new_state
        return new_state, [0, 0], False

    def _showdown(
        self,
        state: GameState,
        new_stacks: list[int],
        new_pot: int,
        new_total: list[int],
        new_history: list[tuple[int, str, int]],
    ) -> tuple[GameState, list[int], bool]:
        """Evaluate hands and award the pot."""
        community = state.community_cards
        score0, _ = evaluate_hand(state.hole_cards[0] + community)
        score1, _ = evaluate_hand(state.hole_cards[1] + community)

        rewards = self._compute_rewards_from_scores(new_total, new_pot, score0, score1)

        new_state = GameState(
            hole_cards=state.hole_cards,
            community_cards=community,
            pot=new_pot,
            stacks=new_stacks,
            current_player=state.current_player,
            street="river",
            current_bet=0,
            min_raise=self.big_blind,
            betting_history=new_history,
            is_done=True,
            big_blind=self.big_blind,
            street_investment=[0, 0],
            total_investment=new_total,
            hand_id=state.hand_id,
        )
        self._state = new_state
        return new_state, rewards, True

    def _compute_rewards(self, state: GameState, winner: int) -> list[int]:
        """Compute net chip change when winner wins by fold."""
        loser = 1 - winner
        rewards = [0, 0]
        rewards[winner] = state.total_investment[loser]   # gains loser's chips
        rewards[loser] = -state.total_investment[loser]   # loses nothing extra (`already lost)
        # Actually: winner gets back their own investment + opponent's
        # Net = pot received - own investment = opp's investment
        rewards[winner] = state.total_investment[loser]
        rewards[loser] = -state.total_investment[loser]
        return rewards

    def _compute_rewards_from_scores(
        self,
        total_investment: list[int],
        pot: int,
        score0: int,
        score1: int,
    ) -> list[int]:
        """Award pot based on hand scores; handle ties."""
        if score0 > score1:
            winner = 0
        elif score1 > score0:
            winner = 1
        else:
            winner = -1  # tie

        if winner == -1:
            # Split pot
            each = pot // 2
            remainder = pot % 2
            return [
                each + remainder - total_investment[0],
                each - total_investment[1],
            ]
        else:
            loser = 1 - winner
            return [
                pot - total_investment[0] if winner == 0 else -total_investment[0],
                pot - total_investment[1] if winner == 1 else -total_investment[1],
            ]
