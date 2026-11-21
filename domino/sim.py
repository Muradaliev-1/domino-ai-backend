"""Basit simülasyon yardımcıları (test ve baseline için)."""
import random
from .engine import Match, MatchConfig, Round


def play_round_random(rnd: Round, rng: random.Random) -> None:
    while not rnd.is_over:
        rnd.apply(rng.choice(rnd.legal_moves()))


def play_match_random(seed, rng: random.Random, config: MatchConfig = None, max_rounds: int = 500) -> Match:
    m = Match(seed, config)
    while m.winner is None and m.round_no < max_rounds:
        r = m.new_round()
        play_round_random(r, rng)
        m.apply_result(r.result)
    return m
