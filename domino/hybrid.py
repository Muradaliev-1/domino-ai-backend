"""
Öğrenme + arama melezi (Faz 6, RQ2/RQ3, D-36): SO-ISMCTS (`search.ISMCTSAgent`, ağırlıklı determinizasyon D-25)
+ Faz 5 DMC ağı (`dmc.py`, varsayılan `results/models/E-04r_l1.npz`). İki mekanizma, ayrı ayrı açılıp kapanır:

1) KÖK ÖNCÜLÜ (PUCT, `temp` > 0): kendi kararımızda (kök) ağın Q değerlerinden öncül P(a) = softmax(Q(a)/temp)
   (yalnız yasal indeksler, D-18 kanonik). Kökte seçim UCB yerine
       Q̄(a) + c_puct · P(a) · sqrt(N_kök) / (1 + n(a)),   Q̄ ziyaretsiz çocukta FPU = 0,5 (nötr ödül),
   yani "önce her hamleyi bir kez dene" kuralı kökte kalkar; öncülü düşük hamleler az denenir. Kök altı aynen
   ISMCTS (kullanılabilirlik sayımlı UCB). Ağ kökte GERÇEK gözlemimizi görür (tek çağrı).
2) YAPRAK DEĞERİ (`lam` > 0): genişletilen düğümün durumu s (bir determinizasyon) bitmemişse, sırası gelen oyuncu
   p için ağın değeri V = max_{a yasal} Q(gözlem_p(s), a) (p'nin takımı açısından tur puanı farkı / 100) ISMCTS
   ödül ölçeğine çevrilir: 0,5 + 100·V / (2·168) ([0, 1]'e kırpılır), diğer takım için 1 − bu. Yaprak değeri
       lam · ağ + (1 − lam) · rastgele rollout sonucu      (lam = 1: rollout yok; lam = 0: saf ISMCTS)
   Bitmiş durumda gerçek tur sonucu kullanılır (ağ çağrılmaz).

SIZINTI YOK: ajan yalnız `Observation` alır (Round gelirse TypeError). Yaprakta ağa giden gözlem_p(s), gözlemden
örneklenmiş determinizasyondan (`round_from_sample`) türetilir: p'nin oradaki eli ÖRNEKTİR, gerçek el değildir.
Gerçek gizli eller değişince karar ve kök istatistikleri değişmez (T-HY3). Maç durumu (L2) public'tir ve her
oyuncunun gözlemi için takım göreli kodlanır (`features._match_block`).
Kural yok (D-22): S-xx'ler kodlanmadı; ağ ve arama ne biliyorsa o.
Hız (Q-34 a, E-06b): yaprak ve kök Q değerleri varsayılan olarak `fastenc` hızlı F3 kodlayıcısıyla hesaplanır
(`fast=True`); referans yol (`fast=False`, `dmc.q_values`) ile aynı tohumda bit bit aynı karar ve kök istatistiği
(T-HY9). Ajan adı değişmedi: karar aynı olduğu için aynı ajandır.
SEYREK YAPRAK DEĞERİ (Q-34 a, E-06c, D-41, `every` = k ≥ 1): ağ yalnız i mod k = 0 olan iterasyonların (i 0'dan)
bitmemiş yaprağında çağrılır; öteki iterasyonlarda yaprak değeri saf rollout (lam = 0 yolu). Bu KARARI DEĞİŞTİRİR
(yeni ajan, ad `:k=K`); k = 1 varsayılan ve eski ajanla bit bit aynı (T-HY10).
`temp = 0` ve `lam = 0` iken ajan aynı tohumla `ISMCTSAgent` ile bit bit aynı karar ve kök istatistiğini verir (T-HY1).

Ad: 'hyb[-rand|-pip]@N[:c=X][:T=t][:lam=l][:cp=X][:k=K][:net=yol.npz]' (make_agent). Varsayılanlar: c 0,7 (D-23),
T 0 (öncül kapalı), lam 0, cp = c, k 1, net E-04r_l1.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional

import numpy as np

from .dmc import REWARD_SCALE, Weights, initial_match_state, legal_indices, load_cached, q_values
from .engine import Move, Observation, Round, team_of
from .fastenc import features_f3
from .info import Info
from .oracle import MAX_POINTS, reward
from .search import PIP_TAU, ISMCTSAgent, WeightedDeterminizer, _ISNode, _rollout, determinize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NET = "results/models/E-04r_l1.npz"
FPU = 0.5                     # ziyaretsiz kök çocuğunun değer tahmini (nötr ödül)


def net_value_to_reward(v: float) -> float:
    """Ağ değeri (takım açısından tur puanı farkı / REWARD_SCALE) → ISMCTS 'points' ödülü (oracle.reward ölçeği)."""
    r = 0.5 + v * REWARD_SCALE / (2 * MAX_POINTS)
    return min(1.0, max(0.0, r))


def fast_ok(w: Weights) -> bool:
    """Hızlı kodlayıcı (fastenc, Q-34 a) yalnız dizisiz F3 ağında kullanılabilir."""
    return w.level == "F3" and not w.seq_cfg


def q_fast(w: Weights, obs: Observation, match: Optional[dict] = None) -> np.ndarray:
    """dmc.q_values ile bit bit aynı (T-HY9), özellikler fastenc.features_f3 ile."""
    if match is None and w.meta.get("l1_match") == "start":
        match = initial_match_state()
    return w.forward(features_f3(obs, match))


def state_value(w: Weights, s: Round, match: Optional[dict] = None, fast: bool = False) -> float:
    """Bitmemiş (determinize edilmiş) durumda 0. TAKIM açısından ağ değeri, ödül ölçeğinde. Ağa yalnız sırası gelen
    oyuncunun bu durumdaki gözlemi gider (determinizasyonda eli örnektir). fast: q_fast (aynı değer, daha hızlı)."""
    p = s.current_player
    obs = s.observation(p)
    q = q_fast(w, obs, match) if fast else q_values(w, obs, match)
    v = max(float(q[i]) for i in legal_indices(obs, s.legal_moves()))
    r = net_value_to_reward(v)
    return r if team_of(p) == 0 else 1.0 - r


def root_prior(w: Weights, obs: Observation, legal, temp: float, match: Optional[dict] = None,
               fast: bool = False) -> Dict[Move, float]:
    """P(a) = softmax(Q(a)/temp) yasal hamleler üzerinde (yalnız gerçek gözlemimiz)."""
    if not isinstance(obs, Observation):
        raise TypeError("root_prior yalnız Observation alır")
    q = q_fast(w, obs, match) if fast else q_values(w, obs, match)
    z = np.array([float(q[i]) for i in legal_indices(obs, legal)], dtype=np.float64) / temp
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return {m: float(x) for m, x in zip(legal, p)}


class HybridAgent(ISMCTSAgent):
    name = "hyb"

    def __init__(self, seed=0, iterations: int = 200, c: float = 0.7, model: Optional[str] = None,
                 pool: Optional[int] = None, tau: float = PIP_TAU, weights: Optional[Weights] = None,
                 path: str = DEFAULT_NET, temp: float = 0.0, lam: float = 0.0, c_puct: Optional[float] = None,
                 fast: bool = True, every: int = 1):
        super().__init__(seed, iterations=iterations, c=c, mode="points", model=model, pool=pool, tau=tau)
        if temp < 0 or not (0.0 <= lam <= 1.0):
            raise ValueError(f"geçersiz melez ayarı: temp={temp}, lam={lam}")
        if int(every) != every or every < 1:
            raise ValueError(f"geçersiz melez ayarı: k={every} (pozitif tam sayı olmalı)")
        self.every = int(every)                           # ağ yalnız i mod every = 0 iterasyonlarında (E-06c)
        if weights is None:
            weights = load_cached(path if os.path.isabs(path) else os.path.join(ROOT, path))
        self.w = weights
        self.fast = bool(fast) and fast_ok(weights)      # False: referans yol (dmc.q_values), karar aynı (T-HY9)
        self.temp, self.lam = float(temp), float(lam)
        self.c_puct = c if c_puct is None else float(c_puct)
        self.last_prior: Optional[Dict[Move, float]] = None
        self.net_calls = 0

    def act(self, obs, legal, match=None):
        if not isinstance(obs, Observation):
            raise TypeError("HybridAgent yalnız Observation alır (gizli bilgi sızıntısını önlemek için)")
        legal = list(legal)
        self.last_iterations = 0
        self.last_root = None
        self.last_prior = None
        if len(legal) == 1:
            return legal[0]
        rng = self.rng
        info = Info(obs)
        wd = None
        if self.model is not None:
            wd = WeightedDeterminizer(obs, info, self.model, self.pool or self.iterations, rng, self.tau)
            self.last_ess = wd.ess
        prior = None
        if self.temp > 0:
            prior = root_prior(self.w, obs, legal, self.temp, match, self.fast)
            self.net_calls += 1
            self.last_prior = prior
        lam = self.lam
        every = self.every
        root = _ISNode(None, None, None)
        for it in range(self.iterations):
            s = wd.draw(rng) if wd is not None else determinize(obs, rng, info)
            node = root
            moves = legal
            pick = None                                        # kökte PUCT'un seçtiği genişletilmemiş hamle
            untried = []
            while True:
                if prior is not None and node is root:
                    sq = math.sqrt(max(root.visits, 1))

                    def score(m):
                        ch = root.children.get(m)
                        n = ch.visits if ch is not None else 0
                        qb = ch.value / ch.visits if n else FPU
                        return qb + self.c_puct * prior[m] * sq / (1 + n)
                    m = max(moves, key=score)
                    if m not in root.children:
                        pick = m
                        break
                    node = root.children[m]
                    s.apply(m)
                    if s.is_over:
                        break
                    moves = s.legal_moves()
                    continue
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
            if not s.is_over and (pick is not None or untried):
                if pick is not None:
                    mv = pick
                else:
                    for m in moves:                           # genişletilen düğüm ve kardeşleri bu turda aday
                        if m in node.children:
                            node.children[m].avail += 1
                    mv = untried[rng.randrange(len(untried))]
                mover = s.current_player
                s.apply(mv)
                child = _ISNode(mv, node, team_of(mover))
                child.avail = 1
                node.children[mv] = child
                node = child
            # yaprak değeri
            res = None
            r0 = None                                          # 0. takım açısından değer (karışımda)
            if s.is_over or lam == 0.0 or it % every:
                _rollout(s, rng)
                res = s.result
            else:
                vnet = state_value(self.w, s, match, self.fast)
                self.net_calls += 1
                if lam == 1.0:
                    r0 = vnet
                else:
                    _rollout(s, rng)
                    r0 = lam * vnet + (1.0 - lam) * reward(s.result, 0, "points")
            while node is not None:
                node.visits += 1
                if node.team is not None:
                    if res is not None:
                        node.value += reward(res, node.team, "points")
                    else:
                        node.value += r0 if node.team == 0 else 1.0 - r0
                node = node.parent
        self.last_iterations = self.iterations
        self.last_root = root
        best = max(root.children.values(), key=lambda ch: (ch.visits, ch.value / ch.visits))
        return best.move


def make_hybrid(name: str, seed) -> HybridAgent:
    """'hyb[-model]@N[:c=..][:T=..][:lam=..][:cp=..][:k=..][:net=..]' → HybridAgent. 'net=' en sonda olmalı: adın geri
    kalanı yol olarak alınır (Windows mutlak yolu 'C:\\...' iki nokta içerebilir)."""
    name, sep, net = name.partition(":net=")
    head, *opts = name.split(":")
    base, _, n = head.partition("@")
    if base != "hyb" and not base.startswith("hyb-"):
        raise ValueError(f"melez ajan adı değil: {name}")
    kw = {}
    model = base[4:] or None
    if model is not None:
        kw["model"] = model
    if n:
        kw["iterations"] = int(n)
    keys = {"c": ("c", float), "T": ("temp", float), "lam": ("lam", float), "cp": ("c_puct", float),
            "k": ("every", int)}
    for o in opts:
        k, eq, v = o.partition("=")
        if not eq or k not in keys:
            raise ValueError(f"bilinmeyen melez seçeneği: {o} ({name})")
        kw[keys[k][0]] = keys[k][1](v)
    if sep:
        kw["path"] = net
    return HybridAgent(seed, **kw)
