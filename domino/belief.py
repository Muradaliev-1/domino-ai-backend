"""
K2: Kesin inanç (belief). Gizli taşların, kısıtlara uyan TÜM dağılımları üzerinde tam sayım,
tam marjinaller ve tam tekdüze örnekleme.

Problem: görünmeyen taşlar (hiçbirimizin görmediği), 3 diğer oyuncuya atanır.
  - oyuncu i tam n_i taş alır (el boyutu kesin)
  - oyuncu i, pas kanıtı olan sayıları (forbidden[i]) taşıyan taşı alamaz
Taşın izinli oyuncu kümesi (maske) yalnızca sayılarına bağlıdır, bu yüzden taşlar en çok 7 maske
grubuna ayrılır. Sayım, gruplar üzerinde küçük bir dinamik programdır (durum: kalan kapasiteler,
en çok 8x8x8). Yaklaşık yok: sayım, marjinal ve örnekleme tamdır.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

Tile = Tuple[int, int]


class InconsistentObservation(Exception):
    """Gözlemle uyumlu hiçbir dağılım yok (olmaması gerekir: bug ya da bozuk gözlem)."""


@lru_cache(maxsize=None)
def _count(gs: Tuple[int, ...], idx: int, caps: Tuple[int, int, int]) -> int:
    """gs[m]: izinli-oyuncu maskesi m olan, henüz atanmamış taş adedi. caps: kalan kapasiteler."""
    while idx < 8 and gs[idx] == 0:
        idx += 1
    if sum(caps) != sum(gs[idx:]):
        return 0
    if idx == 8:
        return 1
    allowed = [i for i in range(3) if (idx >> i) & 1]
    if not allowed:
        return 0                                          # hiçbir oyuncuya atanamayan taş
    g = gs[idx]

    def splits(k: int, remaining: int):
        cap = caps[allowed[k]]
        if k == len(allowed) - 1:
            if remaining <= cap:
                yield (remaining,)
            return
        for x in range(min(remaining, cap) + 1):
            for rest in splits(k + 1, remaining - x):
                yield (x,) + rest

    total = 0
    for xs in splits(0, g):
        new_caps = list(caps)
        mult = factorial(g)
        for i, x in zip(allowed, xs):
            new_caps[i] -= x
            mult //= factorial(x)
        total += mult * _count(gs, idx + 1, tuple(new_caps))
    return total


class Belief:
    def __init__(self, unseen: Iterable[Tile], caps: Sequence[int],
                 forbidden: Sequence[Iterable[int]], players: Sequence[int] = (0, 1, 2)):
        if len(caps) != 3 or len(forbidden) != 3 or len(players) != 3:
            raise ValueError("3 oyuncu bekleniyor")
        self.unseen: Tuple[Tile, ...] = tuple(sorted(unseen))
        self.caps: Tuple[int, int, int] = tuple(caps)            # type: ignore[assignment]
        self.forbidden: Tuple[FrozenSet[int], ...] = tuple(frozenset(f) for f in forbidden)
        self.players: Tuple[int, ...] = tuple(players)
        self._mask: Dict[Tile, int] = {}
        gs = [0] * 8
        for t in self.unseen:
            m = 0
            for i, f in enumerate(self.forbidden):
                if t[0] not in f and t[1] not in f:
                    m |= 1 << i
            self._mask[t] = m
            gs[m] += 1
        self._gs: Tuple[int, ...] = tuple(gs)

    # -- sayım
    def total(self) -> int:
        """Kısıtlara uyan toplam dağılım sayısı."""
        return _count(self._gs, 0, self.caps)

    def joint_count(self, tile: Tile, i: int) -> int:
        """Taş `tile` oyuncu i'nin elindeyken kalan dağılım sayısı (i: 0..2 sırası)."""
        m = self._mask[tile]
        if not (m >> i) & 1 or self.caps[i] == 0:
            return 0
        gs = list(self._gs)
        gs[m] -= 1
        caps = list(self.caps)
        caps[i] -= 1
        return _count(tuple(gs), 0, tuple(caps))

    def require_consistent(self) -> int:
        n = self.total()
        if n == 0:
            raise InconsistentObservation("gözlemle uyumlu dağılım yok")
        return n

    # -- türevler
    def possible(self) -> Dict[Tile, FrozenSet[int]]:
        """Kısıt yayılımı tam: taş t, oyuncu q'da bulunabilir iff en az bir tutarlı dağılımda q'da."""
        self.require_consistent()
        return {t: frozenset(self.players[i] for i in range(3) if self.joint_count(t, i) > 0)
                for t in self.unseen}

    def marginals(self) -> Dict[Tile, Dict[int, float]]:
        """P(taş t, oyuncu q'nun elinde), tekdüze dağılım üzerinde."""
        n = self.require_consistent()
        return {t: {self.players[i]: self.joint_count(t, i) / n for i in range(3)}
                for t in self.unseen}

    def sample(self, rng) -> Dict[int, FrozenSet[Tile]]:
        """Tutarlı dağılımlar üzerinde TAM tekdüze örnek (tam tamsayı ağırlıklarla)."""
        self.require_consistent()
        gs = list(self._gs)
        caps = list(self.caps)
        hands: List[List[Tile]] = [[], [], []]
        for t in self.unseen:
            m = self._mask[t]
            options = []
            for i in range(3):
                if not (m >> i) & 1 or caps[i] == 0:
                    continue
                gs2 = list(gs)
                gs2[m] -= 1
                caps2 = list(caps)
                caps2[i] -= 1
                w = _count(tuple(gs2), 0, tuple(caps2))
                if w:
                    options.append((i, w))
            r = rng.randrange(sum(w for _, w in options))
            for i, w in options:
                if r < w:
                    break
                r -= w
            hands[i].append(t)
            gs[m] -= 1
            caps[i] -= 1
        return {self.players[i]: frozenset(hands[i]) for i in range(3)}
