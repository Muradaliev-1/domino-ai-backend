"""
Bilgi katmanı. YALNIZCA Observation alır (Round kabul etmez): gizli ellere erişim yapısal olarak yok.

K1 Exact : kesin çıkarımlar (sayım, pas kanıtı, kontrol, kapatma)
K2 Belief: kesin olasılıklar ve örnekleyici (belief.py), isteğe bağlı hesaplanır
K3 Soft  : davranış sayaçları (yorumu yok: yorumu uzman kural ya da öğrenen model yapar)
Durumsuz: her çağrıda olay günlüğünden baştan hesaplar (doğruluk önce).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from .belief import Belief
from .engine import (ALL_TILES, Move, Observation, Tile, hand_value, legal_moves_for,
                     missing_from_events, new_ends_after, norm)


@dataclass(frozen=True)
class Exact:
    me: int
    played: FrozenSet[Tile]
    played_by: Tuple[FrozenSet[Tile], ...]       # kim hangi taşı oynadı
    unseen: FrozenSet[Tile]                      # ne masada ne bende: diğer 3 elde
    hand_sizes: Tuple[int, ...]
    missing: Tuple[FrozenSet[int], ...]          # pas kanıtı: oyuncu başına "elinde yok" sayılar
    played_with: Tuple[int, ...]                 # [sayı] o sayıyı içeren oynanmış taş adedi
    mine_with: Tuple[int, ...]                   # [sayı] o sayıyı içeren benim taşlarım
    unseen_with: Tuple[int, ...]                 # [sayı] o sayıyı içeren görünmeyen taşlar
    control: Tuple[bool, ...]                    # sayının kalan TÜM taşları bende (en az bir taşım var)
    dead: Tuple[bool, ...]                       # sayının hiç taşı kalmadı (ne bende ne görünmeyen)


@dataclass(frozen=True)
class Soft:
    """K3 sayaçları. 'Açtı': taşı koyduktan sonra ucu o sayıya çevirdi. 'Üstüne oynadı': o sayı açıkken
    onun üstüne oynadı. Açılış taşı: her iki sayısı için 'açtı'."""
    opened: Tuple[Tuple[int, ...], ...]          # [oyuncu][sayı]
    covered: Tuple[Tuple[int, ...], ...]         # [oyuncu][sayı]
    plays_with: Tuple[Tuple[int, ...], ...]      # [oyuncu][sayı] oynadığı taşlarda sayı sıklığı
    first_play_double: Tuple[Optional[bool], ...]
    n_passes: Tuple[int, ...]


@dataclass(frozen=True)
class CloseOutcome:
    p_win: float
    p_tie: float
    p_loss: float
    exp_delta: float      # beklenen (bizim takım puanı - rakip takım puanı), eşitlik 0 sayılır
    n: int


def compute_exact(obs: Observation) -> Exact:
    played_by = [set() for _ in range(4)]
    for e in obs.events:
        if e.kind == "play":
            played_by[e.player].add(norm(e.tile))
    played = set().union(*played_by)
    mine = set(norm(t) for t in obs.hand)
    unseen = set(ALL_TILES) - played - mine
    played_with = tuple(sum(1 for t in played if n in t) for n in range(7))
    mine_with = tuple(sum(1 for t in mine if n in t) for n in range(7))
    unseen_with = tuple(sum(1 for t in unseen if n in t) for n in range(7))
    return Exact(
        me=obs.player,
        played=frozenset(played),
        played_by=tuple(frozenset(s) for s in played_by),
        unseen=frozenset(unseen),
        hand_sizes=tuple(obs.hand_sizes),
        missing=tuple(frozenset(s) for s in missing_from_events(obs.events)),
        played_with=played_with,
        mine_with=mine_with,
        unseen_with=unseen_with,
        control=tuple(unseen_with[n] == 0 and mine_with[n] > 0 for n in range(7)),
        dead=tuple(unseen_with[n] == 0 and mine_with[n] == 0 for n in range(7)),
    )


def compute_soft(obs: Observation) -> Soft:
    opened = [[0] * 7 for _ in range(4)]
    covered = [[0] * 7 for _ in range(4)]
    plays_with = [[0] * 7 for _ in range(4)]
    first_double: List[Optional[bool]] = [None] * 4
    n_passes = [0] * 4
    for e in obs.events:
        p = e.player
        if e.kind == "pass":
            n_passes[p] += 1
            continue
        a, b = e.tile
        for n in {a, b}:
            plays_with[p][n] += 1
        if first_double[p] is None:
            first_double[p] = (a == b)
        if e.left_end == -1:                             # açılış taşı
            for n in {a, b}:
                opened[p][n] += 1
        else:
            matched = e.left_end if e.side == "L" else e.right_end
            covered[p][matched] += 1
            opened[p][a + b - matched] += 1
    return Soft(tuple(map(tuple, opened)), tuple(map(tuple, covered)),
                tuple(map(tuple, plays_with)), tuple(first_double), tuple(n_passes))


class Info:
    def __init__(self, obs: Observation):
        if not isinstance(obs, Observation):
            raise TypeError("Info yalnızca Observation alır (gizli bilgi sızıntısını önlemek için)")
        self.obs = obs
        self.exact = compute_exact(obs)
        self.soft = compute_soft(obs)
        self._belief: Optional[Belief] = None

    # -- K2
    @property
    def belief(self) -> Belief:
        if self._belief is None:
            me = self.obs.player
            others = tuple((me + i) % 4 for i in (1, 2, 3))
            self._belief = Belief(
                unseen=self.exact.unseen,
                caps=[self.exact.hand_sizes[q] for q in others],
                forbidden=[self.exact.missing[q] for q in others],
                players=others,
            )
        return self._belief

    def possible(self) -> Dict[Tile, FrozenSet[int]]:
        return self.belief.possible()

    def marginals(self) -> Dict[Tile, Dict[int, float]]:
        return self.belief.marginals()

    # -- kapatma (K1 kesin + K2 olasılık)
    def certain_closing_moves(self) -> List[Move]:
        """Şimdi oynarsam yolu KESİN kapatan hamleler: iki yeni uçta görünmeyen taş yok ve elimde
        de kalmadı (kimse oynayamaz). Elimi bitiren hamle kapatma değil, bitirmedir."""
        o = self.obs
        if o.current_player != o.player or o.left_end < 0:
            return []
        out = []
        for mv in legal_moves_for(o.hand, o.left_end, o.right_end):
            if mv.is_pass:
                continue
            nl, nr = new_ends_after(o.left_end, o.right_end, mv)
            rest = [t for t in o.hand if t != mv.tile]
            if not rest:
                continue
            if (self.exact.unseen_with[nl] == 0 and self.exact.unseen_with[nr] == 0
                    and not any(nl in t or nr in t for t in rest)):
                out.append(mv)
        return out

    def close_outcome(self, move: Move, rng, n_samples: int = 500) -> CloseOutcome:
        """Kesin kapatan `move` oynanırsa tur sonucunun dağılımı (eller o andan sonra değişmez)."""
        if move not in self.certain_closing_moves():
            raise ValueError("hamle kesin kapatan bir hamle değil")
        me = self.obs.player
        rest = [t for t in self.obs.hand if t != move.tile]
        win = tie = loss = 0
        delta = 0.0
        for _ in range(n_samples):
            hands = dict(self.belief.sample(rng))
            hands[me] = frozenset(rest)
            totals = [0, 0]
            for p, h in hands.items():
                totals[p % 2] += hand_value(h)
            mine_t, opp_t = totals[me % 2], totals[1 - me % 2]
            if mine_t < opp_t:
                win += 1
                delta += opp_t
            elif mine_t > opp_t:
                loss += 1
                delta -= mine_t
            else:
                tie += 1
        n = n_samples
        return CloseOutcome(win / n, tie / n, loss / n, delta / n, n)


def derive(obs: Observation) -> Info:
    return Info(obs)
