"""
Değerlendirme düzeneği (Faz 2, 07 - Deney Tasarimi). Yer değiştirmeli eşleştirme:
her dağıtım iki kez oynanır; 1. oyunda A takımı 0'da (koltuk 0, 2), 2. oyunda takım 1'de (koltuk 1, 3).
Eller, başlayan ve (maçta) her turun dağıtımı iki oyunda aynıdır; şans payı büyük ölçüde düşer.

L1 (D-07): tek tur, 1|1 sahibi 1|1 ile açar (maçın ilk turu gibi). Metrik: A'nın puan farkı (kazanırsa +puan,
          kaybederse -puan, eşitlikte 0) ve tur kazanma oranı (eşitlik 0,5).
L2 (D-07): tam maç (101). Metrik: maç kazanma oranı (tur sınırına takılan maç 0,5 sayılır ve ayrıca raporlanır).

Ajanlara yalnız round.observation(oyuncu), yasal hamleler ve Match.public_state() verilir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .agents import Agent, make_agent
from .engine import OPENING_TILE, IllegalMove, Match, MatchConfig, Round, deal_hands, team_of

AgentFactory = Callable[[str], Agent]   # tohum (str) -> yeni ajan


def factory(name: str):
    """Ajan adı -> fabrika. Karışık takım (E-04x, partner bağımlılığı): 'X+Y' -> (X, Y) fabrika çifti; takımın
    küçük numaralı koltuğuna X, diğerine Y oturur (takım 0: koltuk 0 = X, 2 = Y; takım 1: 1 = X, 3 = Y)."""
    if "+" in name:
        x, y = name.split("+")
        return (factory(x), factory(y))
    return lambda seed: make_agent(name, seed)


def play_round(rnd: Round, agents: Sequence[Agent], match_state: Optional[dict] = None) -> None:
    while not rnd.is_over:
        p = rnd.current_player
        legal = rnd.legal_moves()
        if getattr(agents[p], "is_oracle", False):
            # BİLİNÇLİ İSTİSNA (D-04): yalnız is_oracle işaretli ajan tam durumun KOPYASINI alır
            mv = agents[p].act_oracle(rnd.clone(), list(legal), match_state)
        else:
            mv = agents[p].act(rnd.observation(p), list(legal), match_state)
        if mv not in legal:
            raise IllegalMove(f"ajan P{p} yasal olmayan hamle döndürdü: {mv}")
        rnd.apply(mv)


def seat_agents(fa: AgentFactory, fb: AgentFactory, a_team: int, seed: str) -> List[Agent]:
    """A takımı a_team'de. Her koltuğa kendi tohumuyla yeni ajan. Fabrika çifti (karışık takım) ise
    takımın ilk koltuğuna (s // 2 == 0) ilk eleman, ikincisine ikinci eleman."""
    def member(f, s):
        return f[s // 2] if isinstance(f, tuple) else f
    return [member(fa if team_of(s) == a_team else fb, s)(f"{seed}:s{s}") for s in range(4)]


def l1_deal(seed, i: int) -> Round:
    hands = deal_hands(f"L1:{seed}", i)
    starter = next(p for p, h in enumerate(hands) if OPENING_TILE in h)
    return Round(hands, starter, forced_opening=OPENING_TILE)


def a_view(res, a_team: int):
    """Turu A takımı açısından: (puan farkı, kazanma skoru 1/0,5/0)."""
    if res.winner_team is None:
        return 0, 0.5
    if res.winner_team == a_team:
        return res.points, 1.0
    return -res.points, 0.0


@dataclass
class Stat:
    n: int
    mean: float
    se: float

    @property
    def ci95(self):
        return (self.mean - 1.96 * self.se, self.mean + 1.96 * self.se)


def stat(xs: Sequence[float]) -> Stat:
    """Ortalama ve standart hata (örneklem std / sqrt(n)). Eşleştirilmiş çiftlerin ortalamaları üzerinden çağrılır."""
    n = len(xs)
    if n == 0:
        raise ValueError("boş örnek")
    m = sum(xs) / n
    if n == 1:
        return Stat(1, m, float("nan"))
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return Stat(n, m, math.sqrt(var / n))


@dataclass
class PairedResult:
    level: str
    seed: str
    n_pairs: int
    diff: Optional[Stat]          # L1: A'nın ortalama puan farkı (çift ortalaması)
    win: Stat                     # A'nın kazanma oranı (çift ortalaması)
    unfinished: int = 0           # L2: tur sınırına takılan maç sayısı
    games: List[tuple] = field(default_factory=list)   # (çift, a_team, değer, kazanma) ham sonuçlar


def l1_pair_games(fa: AgentFactory, fb: AgentFactory, seed, i: int) -> List[tuple]:
    """L1 çiftinin iki oyunu: [(i, a_team, puan farkı, kazanma), ...]. Her çift kendi tohumlarıyla bağımsızdır,
    bu yüzden çiftler hangi sırada ya da hangi süreçte oynanırsa oynansın sonuç aynıdır (paired_parallel)."""
    out = []
    for a_team in (0, 1):
        rnd = l1_deal(seed, i)
        play_round(rnd, seat_agents(fa, fb, a_team, f"L1:{seed}:{i}:{a_team}"))
        d, w = a_view(rnd.result, a_team)
        out.append((i, a_team, d, w))
    return out


def l1_result(games: Sequence[tuple], n_pairs: int, seed) -> PairedResult:
    """Ham oyunlardan (çift sırasına göre, her çiftte a_team 0 sonra 1) L1 sonucu."""
    diffs, wins = [], []
    for k in range(n_pairs):
        (i0, t0, d0, w0), (i1, t1, d1, w1) = games[2 * k], games[2 * k + 1]
        assert i0 == i1 == k and (t0, t1) == (0, 1), "oyun sırası bozuk"
        diffs.append((d0 + d1) / 2)
        wins.append((w0 + w1) / 2)
    return PairedResult("L1", str(seed), n_pairs, stat(diffs), stat(wins), 0, list(games))


def paired_l1(fa: AgentFactory, fb: AgentFactory, n_pairs: int, seed=0) -> PairedResult:
    games = []
    for i in range(n_pairs):
        games.extend(l1_pair_games(fa, fb, seed, i))
    return l1_result(games, n_pairs, seed)


def play_match(seed, agents: Sequence[Agent], config: Optional[MatchConfig] = None, max_rounds: int = 300) -> Match:
    m = Match(seed, config)
    while m.winner is None and m.round_no < max_rounds:
        r = m.new_round()
        play_round(r, agents, m.public_state())
        m.apply_result(r.result)
    return m


def l2_pair_games(fa: AgentFactory, fb: AgentFactory, seed, i: int,
                  config: Optional[MatchConfig] = None, max_rounds: int = 300) -> List[tuple]:
    """L2 çiftinin iki maçı: [(i, a_team, tur sayısı, kazanma, bitmedi mi), ...]."""
    out = []
    for a_team in (0, 1):
        m = play_match(f"L2:{seed}:{i}", seat_agents(fa, fb, a_team, f"L2:{seed}:{i}:{a_team}"),
                       config, max_rounds)
        if m.winner is None:
            out.append((i, a_team, m.round_no, 0.5, True))
        else:
            out.append((i, a_team, m.round_no, 1.0 if m.winner == a_team else 0.0, False))
    return out


def l2_result(games: Sequence[tuple], n_pairs: int, seed) -> PairedResult:
    wins, unfinished = [], 0
    for k in range(n_pairs):
        g0, g1 = games[2 * k], games[2 * k + 1]
        assert g0[0] == g1[0] == k and (g0[1], g1[1]) == (0, 1), "oyun sırası bozuk"
        wins.append((g0[3] + g1[3]) / 2)
        unfinished += g0[4] + g1[4]
    return PairedResult("L2", str(seed), n_pairs, None, stat(wins), unfinished, [g[:4] for g in games])


def paired_l2(fa: AgentFactory, fb: AgentFactory, n_pairs: int, seed=0,
              config: Optional[MatchConfig] = None, max_rounds: int = 300) -> PairedResult:
    games = []
    for i in range(n_pairs):
        games.extend(l2_pair_games(fa, fb, seed, i, config, max_rounds))
    return l2_result(games, n_pairs, seed)


# ── çok süreçli koşu ─────────────────────────────────────────
# Fabrikalar (lambda) süreçler arası taşınamaz, bu yüzden ajan ADLARI gönderilir. Çiftler bitişik parçalara
# bölünür, her parça ayrı süreçte oynanır, sonuçlar çift sırasına göre birleştirilir: tek süreçliyle BİREBİR aynı.

def _chunk(level: str, a_name: str, b_name: str, seed, start: int, stop: int, max_rounds: int) -> List[tuple]:
    fa, fb = factory(a_name), factory(b_name)
    out = []
    for i in range(start, stop):
        if level == "L1":
            out.extend(l1_pair_games(fa, fb, seed, i))
        else:
            out.extend(l2_pair_games(fa, fb, seed, i, None, max_rounds))
    return out


def chunks(n_pairs: int, workers: int) -> List[Tuple[int, int]]:
    """[0, n_pairs) aralığını bitişik parçalara böl (iş dengesi için süreç başına ~4 parça)."""
    k = max(1, min(n_pairs, 4 * workers))
    bounds = [n_pairs * j // k for j in range(k + 1)]
    return [(bounds[j], bounds[j + 1]) for j in range(k) if bounds[j] < bounds[j + 1]]


def paired_parallel(level: str, a_name: str, b_name: str, n_pairs: int, seed=0, workers: int = 1,
                    max_rounds: int = 300) -> PairedResult:
    """paired_l1 / paired_l2'nin ad tabanlı, çok süreçli karşılığı. workers <= 1 ise süreç açılmaz."""
    if level not in ("L1", "L2"):
        raise ValueError(level)
    if workers <= 1:
        games = _chunk(level, a_name, b_name, seed, 0, n_pairs, max_rounds)
    else:
        from concurrent.futures import ProcessPoolExecutor
        parts = chunks(n_pairs, workers)
        with ProcessPoolExecutor(max_workers=min(workers, len(parts))) as ex:
            futs = [ex.submit(_chunk, level, a_name, b_name, seed, s, e, max_rounds) for s, e in parts]
            games = [g for f in futs for g in f.result()]     # parça sırası = çift sırası
    if level == "L1":
        return l1_result(games, n_pairs, seed)
    return l2_result(games, n_pairs, seed)
