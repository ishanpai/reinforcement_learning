import random
from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    player_sum: int
    dealer_sum: int


@dataclass
class Action:
    action: str


@dataclass
class Reward:
    reward: float


def init_game() -> State:
    player_card = np.abs(hit(0))  # Get just the card value
    dealer_card = np.abs(hit(0))  # Get just the card value
    return State(
        player_sum=player_card,
        dealer_sum=dealer_card,
    )


def is_bust(sum: int) -> bool:
    return sum > 21 or sum < 1


def hit(sum: int) -> tuple[int, int]:
    card_number = random.randint(1, 10)
    card_color = random.choices(["red", "black"], weights=[1 / 3, 2 / 3])[0]
    if card_color == "red":
        card_number = -card_number
    return sum + card_number


def step(s: State, a: Action) -> tuple[State, Reward]:
    if a.action == "hit":
        new_sum = hit(s.player_sum)
        if is_bust(new_sum):
            return State(new_sum, s.dealer_sum), Reward(-1)
        else:
            return State(new_sum, s.dealer_sum), Reward(0)
    elif a.action == "stick":
        dealer_sum = s.dealer_sum
        while dealer_sum < 17:
            dealer_sum = hit(dealer_sum)
            if is_bust(dealer_sum):
                return State(s.player_sum, dealer_sum), Reward(1)
        if s.player_sum > dealer_sum:
            return State(s.player_sum, dealer_sum), Reward(1)
        elif s.player_sum == dealer_sum:
            return State(s.player_sum, dealer_sum), Reward(0)
        else:
            return State(s.player_sum, dealer_sum), Reward(-1)
    else:
        raise ValueError(f"Invalid action: {a.action}")

