"""
Gizli bilgisiz arama ajanları (Faz 3): PIMC ve SO-ISMCTS. İkisi de YALNIZ Observation görür.

Determinizasyon: gözlemden (kendi el, masa, olay günlüğü, el boyutları, pas kanıtı) K2 tam tekdüze örnekleyicisiyle
(belief.py) diğer üç elin bir örneği çekilir ve bu örnekle tam bilgili bir Round kurulur. Gerçek eller hiçbir
yerde kullanılmaz: `determinize` yalnız Observation ve örnek alır.

Ödül ve perspektif oracle.py ile aynı (B-02 düzeltmesi): her düğümün değeri, o düğüme HAMLEYİ YAPAN oyuncunun takımı
açısından tutulur; rakip düğümlerinde seçim rakibin lehine yapılır. Rollout tekdüze rastgele (D-22: elle sezgi yok).

PIMCAgent (Perfect Information Monte Carlo): D determinizasyon, her birinde ayrı tam bilgili UCT (N/D iterasyon);
kök hamlelerinin ziyaret sayıları toplanır, en çok ziyaret edilen seçilir (eşitlikte ortalama değer).
Bilinen zayıflıkları (strategy fusion, non-locality) literatürde belgeli; burada ölçülen şey bu.

ISMCTSAgent (SO-ISMCTS, tek gözlemci): tek ağaç; her iterasyonda yeni determinizasyon. Düğümün çocukları hamleye göre
anahtarlı; o determinizasyonda yasal olmayan çocuklar atlanır. Seçimde kullanılabilirlik sayımlı UCB:
  değer/ziyaret + c·sqrt(ln(kullanılabilirlik)/ziyaret), kullanılabilirlik = çocuğun seçime aday olduğu iterasyon sayısı.
"""
from __future__ import annotations

import bisect
import math
import random
from typing import Dict, List, Mapping, Optional, Sequence

from .agents import Agent
from .engine import Move, Observation, Round, legal_moves_for, norm, team_of
from .info import Info
from .oracle import reward


def determinize(obs: Observation, rng, info: Optional[Info] = None) -> Round:
    """Gözlemle tutarlı, tekdüze örneklenmiş tam bilgili bir tur. Yalnız Observation kullanır."""
    if not isinstance(obs, Observation):
        raise TypeError("determinize yalnız Observation alır")
    info = info or Info(obs)
    sample = info.belief.sample(rng)                  # {oyuncu: frozenset(taşlar)}, diğer üç oyuncu
    return round_from_sample(obs, sample)


def round_from_sample(obs: Observation, sample: Mapping[int, frozenset]) -> Round:
    """Gözlem + diğer üç elin bir örneğinden tam bilgili tur. Gerçek eller kullanılmaz."""
    hands: List[set] = [set() for _ in range(4)]
    hands[obs.player] = set(obs.hand)
    for q, h in sample.items():
        hands[q] = set(h)
    r = Round.__new__(Round)
    played_by: List[list] = [[] for _ in range(4)]
    last_tile_player = None
    for e in obs.events:
        if e.kind == "play":
            played_by[e.player].append(e.tile)
            last_tile_player = e.player
    r._hands = hands
    r.initial_hands = tuple(tuple(sorted(set(hands[p]) | set(played_by[p]))) for p in range(4))
    r.forced_opening = None                            # yalnız boş masada önemli; kökte yasal hamleler verilir
    r.starter = obs.events[0].player if obs.events else obs.current_player
    r.board = list(obs.board)
    r.current_player = obs.current_player
    r.consecutive_passes = obs.consecutive_passes
    r.events = list(obs.events)
    r.last_tile_player = last_tile_player
    r.result = None
    return r


# ── K3/Bayes ağırlıklı determinizasyon (RQ3, D-25) ──────────────────────────
# Tekdüze K2 örneği yalnız sert kısıtları (el boyutu, pas kanıtı) kullanır. Diğer oyuncuların OYNADIĞI hamleler de
# ellerine dair kanıttır: P(eller | geçmiş) ∝ P(eller) · Π_t π(a_t | o andaki el, o andaki uçlar). Önsel (dağıtım)
# tekdüze olduğundan ve oynanmış taşlar public olduğundan, tutarlı ŞİMDİKİ eller üzerinde tekdüze önselle başlayıp
# her örneği gözlenen hamlelerin olasılığıyla ağırlıklandırmak tam posteriordur. π bir OYUNCU MODELİdir (D-22: S-xx
# kuralı değil):
#   'rand' : tekdüze rastgele oyuncu, π(a) = 1/|yasal|  (bilgisiz model: çok seçeneği olan elde belli bir hamle
#            daha az olası)
#   'pip'  : yumuşak açgözlü, π(a) ∝ exp(pip(a)/τ)  (τ = PIP_TAU, ayarlanmadı; açgözlü baseline'ın yumuşatılmışı)
# Paslar: modelden bağımsız olarak π(pas) = 1 ancak ve ancak yasal hamle yoksa, yani pas kanıtı zaten K2'nin sert
# kısıtıdır (ağırlık 1). Kendi hamlelerim: elim biliniyor, sabit çarpan (atlanır). Açılış taşı (masa boşken) atlanır:
# zorunlu açılış (R-04) gözlemde yok, bilgisiz sayılır. Eşit uçlarda kayıtlı 'R' aynalı yerleşimdir, kanonik 'L'
# hamlesi olarak değerlendirilir (D-18).
PLAYER_MODELS = ("rand", "pip")
PIP_TAU = 3.0


def _pips(m: Move) -> int:
    return m.tile[0] + m.tile[1]


def move_logprob(model: str, legal: Sequence[Move], move: Move, tau: float = PIP_TAU) -> float:
    """log π(move | yasal hamleler) oyuncu modeli altında."""
    if move not in legal:
        raise ValueError(f"hamle yasal değil: {move}")
    if model == "rand":
        return -math.log(len(legal))
    if model == "pip":
        xs = [_pips(m) / tau for m in legal]
        mx = max(xs)
        return _pips(move) / tau - (mx + math.log(sum(math.exp(x - mx) for x in xs)))
    raise ValueError(f"bilinmeyen oyuncu modeli: {model}")


def deal_loglik(obs: Observation, sample: Mapping[int, frozenset], model: str, tau: float = PIP_TAU) -> float:
    """log P(diğer oyuncuların gözlenen hamleleri | şimdiki elleri = sample), oyuncu modeli altında.
    Yalnız Observation ve örnek kullanır. Olaylar sondan başa yürünür: oyuncunun o andaki eli = şimdiki eli +
    o olaydan itibaren oynadığı taşlar."""
    if not isinstance(obs, Observation):
        raise TypeError("deal_loglik yalnız Observation alır")
    me = obs.player
    cur = {q: set(norm(t) for t in h) for q, h in sample.items()}
    ll = 0.0
    for e in reversed(obs.events):
        if e.kind != "play" or e.player == me:
            continue
        t = norm(e.tile)
        cur[e.player].add(t)
        if e.left_end < 0:
            continue                                   # açılış: bilgisiz (zorunlu olabilir)
        side = "L" if e.left_end == e.right_end else e.side
        ll += move_logprob(model, legal_moves_for(cur[e.player], e.left_end, e.right_end), Move(t, side), tau)
    return ll


class WeightedDeterminizer:
    """Önem örneklemesi + yeniden örnekleme (SIR): `pool` tekdüze K2 örneği çekilir, her biri deal_loglik ile
    ağırlıklandırılır, `draw` ağırlıkla orantılı (iadeli) bir örnekten tur kurar. pool sonsuza giderken çekilen
    eller posteriordan gelir. `ess`: etkin örnek sayısı (Σw)²/Σw², ağırlık yozlaşmasının göstergesi."""

    def __init__(self, obs: Observation, info: Info, model: str, pool: int, rng, tau: float = PIP_TAU):
        if model not in PLAYER_MODELS:
            raise ValueError(f"bilinmeyen oyuncu modeli: {model}")
        self.obs = obs
        self.samples = [info.belief.sample(rng) for _ in range(max(1, pool))]
        self.logw = [deal_loglik(obs, smp, model, tau) for smp in self.samples]
        mx = max(self.logw)
        w = [math.exp(x - mx) for x in self.logw]
        self.weights = w
        self.cum = []
        acc = 0.0
        for x in w:
            acc += x
            self.cum.append(acc)
        self.ess = acc * acc / sum(x * x for x in w)

    def draw_sample(self, rng) -> Mapping[int, frozenset]:
        i = bisect.bisect_right(self.cum, rng.random() * self.cum[-1])
        return self.samples[min(i, len(self.samples) - 1)]

    def draw(self, rng) -> Round:
        return round_from_sample(self.obs, self.draw_sample(rng))


def _rollout(state: Round, rng) -> None:
    while not state.is_over:
        moves = state.legal_moves()
        state.apply(moves[rng.randrange(len(moves))])


class _UNode:
    __slots__ = ("move", "parent", "team", "children", "untried", "visits", "value")

    def __init__(self, move, parent, team, untried):
        self.move = move
        self.parent = parent
        self.team = team
        self.children: List["_UNode"] = []
        self.untried = untried
        self.visits = 0
        self.value = 0.0


def uct_root(state: Round, root_moves: Sequence[Move], iterations: int, c: float, mode: str, rng) -> List[_UNode]:
    """Tam bilgili (determinize edilmiş) durum üzerinde UCT; kök çocuklarını döndürür. oracle.py ile aynı mantık."""
    root = _UNode(None, None, None, list(root_moves))
    for _ in range(iterations):
        s = state.clone()
        node = root
        while not node.untried and node.children:
            lp = math.log(node.visits)
            node = max(node.children, key=lambda ch: ch.value / ch.visits + c * math.sqrt(lp / ch.visits))
            s.apply(node.move)
        if node.untried:
            mv = node.untried.pop(rng.randrange(len(node.untried)))
            mover = s.current_player
            s.apply(mv)
            child = _UNode(mv, node, team_of(mover), [] if s.is_over else s.legal_moves())
            node.children.append(child)
            node = child
        _rollout(s, rng)
        res = s.result
        while node is not None:
            node.visits += 1
            if node.team is not None:
                node.value += reward(res, node.team, mode)
            node = node.parent
    return root.children


class PIMCAgent(Agent):
    name = "pimc"

    def __init__(self, seed=0, iterations: int = 200, determinizations: int = 10, c: float = 0.7,
                 mode: str = "points", model: Optional[str] = None, pool: Optional[int] = None,
                 tau: float = PIP_TAU):
        if mode not in ("points", "win"):
            raise ValueError(f"bilinmeyen ödül: {mode}")
        if model is not None and model not in PLAYER_MODELS:
            raise ValueError(f"bilinmeyen oyuncu modeli: {model}")
        self.model, self.pool, self.tau = model, pool, tau     # None: tekdüze K2 (D-23); aksi: ağırlıklı (D-25)
        self.last_ess = None
        self.rng = random.Random(seed)
        self.iterations = iterations
        self.determinizations = max(1, min(determinizations, iterations))
        self.c = c
        self.mode = mode
        self.last_iterations = 0
        self.last_stats: Dict[Move, list] = {}

    def act(self, obs, legal, match=None):
        legal = list(legal)
        self.last_iterations = 0
        self.last_stats = {}
        if len(legal) == 1:
            return legal[0]
        info = Info(obs)
        d = self.determinizations
        per = [self.iterations // d + (1 if k < self.iterations % d else 0) for k in range(d)]
        wd = None
        if self.model is not None:
            wd = WeightedDeterminizer(obs, info, self.model, self.pool or self.iterations, self.rng, self.tau)
            self.last_ess = wd.ess
        stats: Dict[Move, list] = {m: [0, 0.0] for m in legal}       # [ziyaret, değer toplamı]
        for k in range(d):
            if per[k] == 0:
                continue
            state = wd.draw(self.rng) if wd is not None else determinize(obs, self.rng, info)
            for ch in uct_root(state, legal, per[k], self.c, self.mode, self.rng):
                stats[ch.move][0] += ch.visits
                stats[ch.move][1] += ch.value
        self.last_iterations = sum(per)
        self.last_stats = stats
        return max(legal, key=lambda m: (stats[m][0], stats[m][1] / stats[m][0] if stats[m][0] else 0.0))


class _ISNode:
    __slots__ = ("move", "parent", "team", "children", "visits", "value", "avail")

    def __init__(self, move, parent, team):
        self.move = move
        self.parent = parent
        self.team = team                      # bu düğüme gelen hamleyi yapanın takımı (kök: None)
        self.children: Dict[Move, "_ISNode"] = {}
        self.visits = 0
        self.value = 0.0
        self.avail = 0                        # seçimde aday (o determinizasyonda yasal) olduğu iterasyon sayısı


class ISMCTSAgent(Agent):
    name = "ismcts"

    def __init__(self, seed=0, iterations: int = 200, c: float = 0.7, mode: str = "points",
                 model: Optional[str] = None, pool: Optional[int] = None, tau: float = PIP_TAU):
        if mode not in ("points", "win"):
            raise ValueError(f"bilinmeyen ödül: {mode}")
        if model is not None and model not in PLAYER_MODELS:
            raise ValueError(f"bilinmeyen oyuncu modeli: {model}")
        self.model, self.pool, self.tau = model, pool, tau     # None: tekdüze K2 (D-23); aksi: ağırlıklı (D-25)
        self.last_ess = None
        self.rng = random.Random(seed)
        self.iterations = iterations
        self.c = c
        self.mode = mode
        self.last_iterations = 0
        self.last_root: Optional[_ISNode] = None

    def _ucb(self, ch: _ISNode) -> float:
        return ch.value / ch.visits + self.c * math.sqrt(math.log(ch.avail) / ch.visits)

    def act(self, obs, legal, match=None):
        legal = list(legal)
        self.last_iterations = 0
        self.last_root = None
        if len(legal) == 1:
            return legal[0]
        rng = self.rng
        info = Info(obs)
        wd = None
        if self.model is not None:
            wd = WeightedDeterminizer(obs, info, self.model, self.pool or self.iterations, rng, self.tau)
            self.last_ess = wd.ess
        root = _ISNode(None, None, None)
        for _ in range(self.iterations):
            s = wd.draw(rng) if wd is not None else determinize(obs, rng, info)
            node = root
            moves = legal                                      # kökte yasal hamleler kendi elimden: kesin
            while True:
                untried = [m for m in moves if m not in node.children]
                if untried:
                    break
                cands = [node.children[m] for m in moves]
                for ch in cands:
                    ch.avail += 1
                node = max(cands, key=self._ucb)
                s.apply(node.move)
                if s.is_over:
                    break
                moves = s.legal_moves()
            if not s.is_over and untried:
                for m in moves:                               # genişletilen düğüm ve kardeşleri bu turda aday
                    if m in node.children:
                        node.children[m].avail += 1
                mv = untried[rng.randrange(len(untried))]
                mover = s.current_player
                s.apply(mv)
                child = _ISNode(mv, node, team_of(mover))
                child.avail = 1
                node.children[mv] = child
                node = child
            _rollout(s, rng)
            res = s.result
            while node is not None:
                node.visits += 1
                if node.team is not None:
                    node.value += reward(res, node.team, self.mode)
                node = node.parent
        self.last_iterations = self.iterations
        self.last_root = root
        best = max(root.children.values(), key=lambda ch: (ch.visits, ch.value / ch.visits))
        return best.move
