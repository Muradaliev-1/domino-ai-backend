"""
!!! ORACLE (TAM BİLGİLİ) AJAN: GİZLİ ELLERİ GÖRÜR. Bilerek etiketlenmiş tek istisna (CLAUDE.md, D-04). !!!
Yalnız üst sınır / karşılaştırma referansı. Gizli bilgisiz hiçbir ajanla karıştırılmaz, demoda kullanılmaz.

Tam bilgili tur üzerinde UCT (MCTS). Eski MCTS'ten (legacy/) farklar:
- B-02 düzeltmesi: her düğümün değeri, o düğüme HAMLEYİ YAPAN oyuncunun takımı açısından tutulur. Rakip düğümlerinde
  seçim rakibin lehine yapılır (rakip "yardım etmez").
- B-01 burada bir hata değil, tanımın kendisi: ajan gerçek durumu görür. Arena ona yalnız `act_oracle` ile, durumun
  bir KOPYASINI verir; normal `act` çağrılırsa hata verir.
- Rollout politikası tekdüze rastgele (elle yazılmış sezgi yok, D-22), tohumlu ve yeniden üretilebilir.

Ödül [0, 1] (takım açısından):
  'points': 0,5 + puan farkı / (2·168)   (168 = tüm taşların pip toplamı, puan farkının üst sınırı). Varsayılan.
  'win'   : kazanırsa 1, eşitlik 0,5, kaybederse 0.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional

from .agents import Agent
from .engine import ALL_TILES, Move, Round, RoundResult, team_of

MAX_POINTS = sum(a + b for a, b in ALL_TILES)   # 168


def team_diff(res: RoundResult, team: int) -> int:
    """Tur sonucunun takım açısından puan farkı (kazandıysa +puan, kaybettiyse -puan, eşitlik 0)."""
    if res.winner_team is None:
        return 0
    return res.points if res.winner_team == team else -res.points


def reward(res: RoundResult, team: int, mode: str) -> float:
    if mode == "win":
        if res.winner_team is None:
            return 0.5
        return 1.0 if res.winner_team == team else 0.0
    return 0.5 + team_diff(res, team) / (2 * MAX_POINTS)


class _Node:
    __slots__ = ("move", "parent", "team", "children", "untried", "visits", "value")

    def __init__(self, move: Optional[Move], parent: Optional["_Node"], team: Optional[int], untried: List[Move]):
        self.move = move
        self.parent = parent
        self.team = team              # bu düğüme gelen hamleyi yapan oyuncunun takımı (kök: None)
        self.children: List[_Node] = []
        self.untried = untried
        self.visits = 0
        self.value = 0.0              # self.team açısından ödül toplamı


class OracleMCTS(Agent):
    name = "oracle"
    is_oracle = True

    def __init__(self, seed=0, iterations: int = 200, c: float = 0.7, mode: str = "points"):
        if mode not in ("points", "win"):
            raise ValueError(f"bilinmeyen ödül: {mode}")
        self.rng = random.Random(seed)
        self.iterations = iterations
        self.c = c
        self.mode = mode
        self.last_iterations = 0
        self.last_root: Optional[_Node] = None

    def act(self, obs, legal, match=None):
        raise TypeError("OracleMCTS yalnız act_oracle ile (tam durumla) çağrılır; Observation ile oynayamaz")

    def _ucb(self, parent: _Node, ch: _Node) -> float:
        return ch.value / ch.visits + self.c * math.sqrt(math.log(parent.visits) / ch.visits)

    def act_oracle(self, rnd: Round, legal, match=None) -> Move:
        legal = list(legal)
        self.last_iterations = 0
        self.last_root = None
        if len(legal) == 1:
            return legal[0]
        rng = self.rng
        root = _Node(None, None, None, list(legal))
        for _ in range(self.iterations):
            state = rnd.clone()
            node = root
            # seçim
            while not node.untried and node.children:
                node = max(node.children, key=lambda ch: self._ucb(node, ch))
                state.apply(node.move)
            # genişletme
            if node.untried:
                mv = node.untried.pop(rng.randrange(len(node.untried)))
                mover = state.current_player
                state.apply(mv)
                child = _Node(mv, node, team_of(mover), [] if state.is_over else state.legal_moves())
                node.children.append(child)
                node = child
            # rollout
            while not state.is_over:
                moves = state.legal_moves()
                state.apply(moves[rng.randrange(len(moves))])
            # geri yayılım: her düğüm kendi hamlecisinin takımı açısından
            res = state.result
            while node is not None:
                node.visits += 1
                if node.team is not None:
                    node.value += reward(res, node.team, self.mode)
                node = node.parent
        self.last_iterations = self.iterations
        self.last_root = root
        best = max(root.children, key=lambda ch: (ch.visits, ch.value / ch.visits))
        return best.move
