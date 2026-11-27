"""
Maç değeri tablosu (MET, "match equity table"; Q-33, D-35). Public maç durumundan maçı kazanma olasılığı.

Model: maç, tur sonuçlarının bir Markov zinciridir. Tur sonucu (kazanan takım, tur puanı, eşitlik havuzu, sonraki turu
başlatan takım) yalnız turu KİMİN ve NASIL başlattığına bağlı bir dağılımdan gelir: zorunlu açılış (R-19, 1|1) ya da
serbest başlangıç (R-14); maç durumundan bağımsız sayılır (VARSAYIM: politika maç durumunu gördüğü için tam doğru
değil; kalibrasyon testi bunu ölçer). Dağılım, başlatan takıma GÖRELİ tutulur (w_rel = 0: başlatan takım kazandı),
yani iki takım için simetriktir. Dağılım durumdan bağımsız olduğu için seyrek durum sorunu yoktur: tablo dağılımdan
dinamik programlamayla (değer yinelemesi) TAM çözülür; maç kuralları `step` ile `Match.apply_result`'in birebir
aynısıdır (T-MET1).

Durum (`MState`, mutlak takımlar): skorlar, bekleyenler (R-17, eşitlik payı dahil, R-11), kilit (R-16/R-18),
sonraki turu başlatan takım (None = 1|1 açılışı), maç bittiyse kazanan. Hepsi public (D-09/D-11); el yok, sızıntı yok.

Sıkıştırma (kesin): bekleyen >= hedef ise "kilit açılırsa/puan yazılırsa maç biter" demektir, bu yüzden bekleyen
hedefte kırpılır. Yapısal gerçekler: kilidi açık takımın bekleyeni yalnız son kesin sonuçtan beri gelen eşitlik
paylarıdır (t); kilitli takımınki a + t (a: kendi küçük kazançları). İki takım da açıksa bekleyenler eşit (t).
Bloklar: V11[σ, s0, s1, t], V10[σ, s0, t, a] (0 açık, 1 kilitli), V01 = 1 − V10 aynası, V00[σ, p0, p1] (σ = 2: açılış).
Değer = 0. takımın maçı kazanma olasılığı.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .engine import Match, MatchConfig, team_of

OPEN = 2                      # σ indeksi: R-19 açılışı (başlatan takım 1|1'e göre, 1/2 olasılıkla)


@dataclass(frozen=True)
class MState:
    score: Tuple[int, int]
    pending: Tuple[int, int]
    unlocked: Tuple[bool, bool]
    start: Optional[int]               # sonraki turu başlatan TAKIM; None = 1|1 açılışı
    winner: Optional[int] = None


def initial_state() -> MState:
    return MState((0, 0), (0, 0), (False, False), None, None)


def state_of(m: Match) -> MState:
    """Maçın (turlar arası) public durumu."""
    ns = m.next_start()
    return MState(tuple(m.score), tuple(m.pending), tuple(m.unlocked), None if ns is None else team_of(ns), m.winner)


def step(st: MState, winner: Optional[int], points: int, pot: int, next_team: int, cfg: MatchConfig) -> MState:
    """Bir tur sonucunun maç durumuna etkisi: `Match.apply_result` ile aynı kurallar (R-11, R-14, R-16..R-19).
    winner None = kapanmada eşitlik (pot iki bekleyene). next_team = son taşı koyanın takımı."""
    sc, pe, un = list(st.score), list(st.pending), list(st.unlocked)
    if winner is None:
        return MState(tuple(sc), (pe[0] + pot, pe[1] + pot), tuple(un), next_team, None)
    w, l = winner, 1 - winner
    pe[l] = 0
    if un[w] or points >= cfg.threshold:
        sc[w] += pe[w] + points
        pe[w] = 0
        un[w] = True
    else:
        pe[w] += points
    win = w if sc[w] >= cfg.target else None
    start = None if not any(un) else next_team
    return MState(tuple(sc), tuple(pe), tuple(un), start, win)


# ── tur sonucu dağılımı ──────────────────────────────────────

class OutcomeDist:
    """Başlatan takıma göreli tur sonucu dağılımı, iki tür: 'forced' (1|1 açılışı) ve 'free'.
    dec[kind]: (2 w_rel, 2 ns_rel, K+1) olasılık; tie[kind]: (2 ns_rel, P+1). Toplam (dec + tie) = 1."""

    KINDS = ("forced", "free")

    def __init__(self, counts: Dict[str, Dict[str, int]]):
        self.counts = counts
        K = P = 1
        for kind in self.KINDS:
            for key in counts[kind]:
                f = key.split(",")
                if f[0] == "d":
                    K = max(K, int(f[3]))
                else:
                    P = max(P, int(f[2]))
        self.K, self.P = K, P
        self.dec, self.tie, self.n = {}, {}, {}
        for kind in self.KINDS:
            d = np.zeros((2, 2, K + 1))
            t = np.zeros((2, P + 1))
            for key, c in counts[kind].items():
                f = key.split(",")
                if f[0] == "d":
                    d[int(f[1]), int(f[2]), int(f[3])] += c
                else:
                    t[int(f[1]), int(f[2])] += c
            n = d.sum() + t.sum()
            if n <= 0:
                raise ValueError(f"boş dağılım: {kind}")
            if d[:, :, 0].sum() or t[:, 0].sum():
                raise ValueError("kesin sonuç puanı ve eşitlik havuzu >= 1 olmalı")
            self.dec[kind], self.tie[kind], self.n[kind] = d / n, t / n, int(n)

    @staticmethod
    def key(winner_rel: Optional[int], points: int, pot: int, ns_rel: int) -> str:
        return f"t,{ns_rel},{pot}" if winner_rel is None else f"d,{winner_rel},{ns_rel},{points}"

    @classmethod
    def from_rounds(cls, rounds: Iterable[dict]) -> "OutcomeDist":
        """rounds: {'forced', 'starter_team', 'winner', 'points', 'pot', 'next_team'} kayıtları."""
        counts: Dict[str, Dict[str, int]] = {k: {} for k in cls.KINDS}
        for r in rounds:
            st = r["starter_team"]
            w = None if r["winner"] is None else int(r["winner"] != st)
            k = cls.key(w, r["points"], r["pot"], int(r["next_team"] != st))
            c = counts["forced" if r["forced"] else "free"]
            c[k] = c.get(k, 0) + 1
        return cls(counts)

    def absolute(self, sigma: int):
        """σ ∈ {0, 1} (başlatan takım, serbest) ya da OPEN (zorunlu açılış, başlatan takım 1/2 olasılıkla).
        Döner: Pd (2 w, 2 ns, K+1), Pt (2 ns, P+1), mutlak takımlarla."""
        def ab(kind, s):
            d, t = self.dec[kind], self.tie[kind]
            o = [s, 1 - s]                      # mutlak takım → göreli indeks: takım s ise 0
            Pd = np.empty_like(d)
            Pt = np.empty_like(t)
            for w in (0, 1):
                for ns in (0, 1):
                    Pd[w, ns] = d[int(w != s), int(ns != s)]
                Pt[w] = t[int(w != s)]
            return Pd, Pt
        if sigma == OPEN:
            a, b = ab("forced", 0), ab("forced", 1)
            return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        return ab("free", sigma)

    def to_json(self) -> dict:
        return {"counts": self.counts}

    @classmethod
    def from_json(cls, d: dict) -> "OutcomeDist":
        return cls(d["counts"])


# ── tablo (vektörel DP) ─────────────────────────────────────

class MET:
    def __init__(self, dist: OutcomeDist, config: Optional[MatchConfig] = None, tol: float = 1e-12,
                 max_sweeps: int = 5000):
        self.dist = dist
        self.cfg = config or MatchConfig()
        self.sweeps, self.residual = self._solve(tol, max_sweeps)

    # -- çözüm
    def _solve(self, tol, max_sweeps):
        T, th = self.cfg.target, self.cfg.threshold
        P = {s: self.dist.absolute(s) for s in (0, 1, OPEN)}
        V11 = np.full((2, T, T, T + 1), 0.5)
        V10 = np.full((2, T, T + 1, T + 1), 0.5)
        V00 = np.full((3, T + 1, T + 1), 0.5)
        ar_s = np.arange(T)
        ar_t = np.arange(T + 1)
        K = self.dist.K
        ks = np.arange(K + 1)
        big = ks >= th
        res = float("inf")
        for sweep in range(1, max_sweeps + 1):
            V01_00 = 1.0 - V10[::-1, :, 0, 0]          # V01[ns][s1, 0, 0] = 1 − V10[1−ns][s1, 0, 0]
            n11 = np.zeros_like(V11)
            n10 = np.zeros_like(V10)
            n00 = np.zeros_like(V00)
            for sg in (0, 1, OPEN):
                Pd, Pt = P[sg]
                for w in (0, 1):
                    for ns in (0, 1):
                        p = Pd[w, ns]
                        if not p.any():
                            continue
                        win_v = 1.0 if w == 0 else 0.0
                        if sg != OPEN:
                            # V11: kazanan açık, s_w + t + k
                            base = V11[ns, :, :, 0]                       # [s0, s1]
                            if w == 0:
                                ext = np.concatenate([base, np.full((T + K + 1, T), win_v)], 0)
                                H = sum(p[k] * ext[k:k + 2 * T + 1] for k in range(1, K + 1) if p[k])
                                n11[sg] += H[ar_s[:, None, None] + ar_t[None, None, :], ar_s[None, :, None]]
                            else:
                                ext = np.concatenate([base, np.full((T, T + K + 1), win_v)], 1)
                                H = sum(p[k] * ext[:, k:k + 2 * T + 1] for k in range(1, K + 1) if p[k])
                                n11[sg] += H[ar_s[:, None, None], ar_s[None, :, None] + ar_t[None, None, :]]
                            # V10: 0 açık, 1 kilitli
                            if w == 0:
                                ext = np.concatenate([V10[ns, :, 0, 0], np.full(T + K + 1, 1.0)])
                                H = sum(p[k] * ext[k:k + 2 * T + 1] for k in range(1, K + 1) if p[k])
                                n10[sg] += H[ar_s[:, None, None] + ar_t[None, :, None]]
                            else:
                                # y = p1 = min(t + a, T); k >= eşik: kilit açılır, s1 = y + k; yoksa a' = y + k
                                ext = np.concatenate([V11[ns, :, :, 0].T, np.zeros((T + K + 1, T))], 0)  # [s1, s0]
                                G = sum(p[k] * ext[k:k + T + 1] for k in range(K + 1) if p[k] and big[k])
                                G = G if not np.isscalar(G) else np.zeros((T + 1, T))
                                for k in range(1, th):
                                    if k <= K and p[k]:
                                        G = G + p[k] * V10[ns][:, 0, np.minimum(ar_t + k, T)].T   # [y, s0]
                                ya = np.minimum(ar_t[:, None] + ar_t[None, :], T)                  # [t, a]
                                n10[sg] += G[ya[None, :, :], ar_s[:, None, None]]
                        # V00: ikisi de kilitli; y = kazananın bekleyeni
                        if w == 0:
                            ext = np.concatenate([V10[ns, :, 0, 0], np.full(T + K + 1, 1.0)])
                            un_ = V00[OPEN, :, 0]
                        else:
                            ext = np.concatenate([V01_00[ns], np.zeros(T + K + 1)])
                            un_ = V00[OPEN, 0, :]
                        G = np.zeros(T + 1)
                        for k in range(1, K + 1):
                            if p[k]:
                                G += p[k] * (ext[k:k + T + 1] if big[k] else un_[np.minimum(ar_t + k, T)])
                        n00[sg] += G[:, None] if w == 0 else G[None, :]
                # eşitlik: tüm bekleyenler + pot, sonraki başlatan ns
                for ns in (0, 1):
                    for pot in np.nonzero(Pt[ns])[0]:
                        q = Pt[ns, pot]
                        idx = np.minimum(ar_t + pot, T)
                        if sg != OPEN:
                            n11[sg] += q * V11[ns][:, :, idx]
                            n10[sg] += q * V10[ns][:, idx, :]
                        n00[sg] += q * V00[ns][np.ix_(idx, idx)]
            res = max(np.abs(n11 - V11).max(), np.abs(n10 - V10).max(), np.abs(n00 - V00).max())
            V11, V10, V00 = n11, n10, n00
            if res < tol:
                break
        self.V11, self.V10, self.V00 = V11, V10, V00
        return sweep, float(res)

    # -- sorgu
    slope = 1.0                       # kalibrasyon: logit eğimi (kayma yok → takım simetrisi korunur, D-35)

    def p0(self, st: MState) -> float:
        """0. takımın maçı kazanma olasılığı (kalibre: sigmoid(eğim · logit(DP değeri)); biten maç 1/0)."""
        v = self.raw_p0(st)
        if self.slope == 1.0 or st.winner is not None:
            return v
        return calibrate(v, self.slope)

    def raw_p0(self, st: MState) -> float:
        """DP değeri (kalibrasyonsuz)."""
        if st.winner is not None:
            return 1.0 if st.winner == 0 else 0.0
        T = self.cfg.target
        (s0, s1), (u0, u1) = st.score, st.unlocked
        p0, p1 = min(st.pending[0], T), min(st.pending[1], T)
        sg = OPEN if st.start is None else st.start
        if (u0 or u1) and sg == OPEN:
            raise ValueError("kilidi açık takım varken 1|1 açılışı olmaz (R-19)")
        if u0 and u1:
            if p0 != p1:
                raise ValueError("iki takım açıkken bekleyenler eşit olmalı (yalnız eşitlik payı)")
            return float(self.V11[sg, s0, s1, p0])
        if u0:
            if s1 or p1 < p0:
                raise ValueError("tutarsız durum")
            return float(self.V10[sg, s0, p0, p1 - p0])
        if u1:
            if s0 or p0 < p1:
                raise ValueError("tutarsız durum")
            return 1.0 - float(self.V10[1 - sg, s1, p1, p0 - p1])
        if s0 or s1:
            raise ValueError("kilitli takımın skoru 0 olmalı")
        return float(self.V00[sg, p0, p1])

    def value(self, st: MState, team: int) -> float:
        """`team` takımının maçı kazanma olasılığı."""
        v = self.p0(st)
        return v if team == 0 else 1.0 - v

    def value_of_match(self, m: Match, team: int) -> float:
        return self.value(state_of(m), team)


LOGIT_EPS = 1e-9


def calibrate(p: float, slope: float) -> float:
    """sigmoid(eğim · logit(p)); p, [ε, 1 − ε]'a kırpılır. 0,5 sabit, p ↔ 1 − p simetrik, tekdüze."""
    import math
    p = min(max(p, LOGIT_EPS), 1.0 - LOGIT_EPS)
    z = slope * math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def fit_slope(pairs, lo: float = 0.2, hi: float = 5.0, iters: int = 200) -> float:
    """En çok olabilirlikle tek parametreli logit eğimi: pairs = [(DP değeri, sonuç 0/1)]. Log kayıp eğimde dışbükey;
    türevin işaretine göre ikiye bölme."""
    import math
    xs = [math.log(min(max(p, LOGIT_EPS), 1 - LOGIT_EPS) / (1 - min(max(p, LOGIT_EPS), 1 - LOGIT_EPS))) for p, _ in pairs]
    ys = [y for _, y in pairs]

    def grad(b):                                  # d(−log olabilirlik)/db = Σ (σ(bx) − y)·x
        g = 0.0
        for x, y in zip(xs, ys):
            z = b * x
            s = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
            g += (s - y) * x
        return g
    for _ in range(iters):
        mid = (lo + hi) / 2
        if grad(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ── yavaş referans çözücü (testler için, küçük yapılandırma) ────────────

def reference_solve(dist: OutcomeDist, cfg: MatchConfig, tol: float = 1e-13, max_sweeps: int = 100000):
    """Başlangıçtan erişilebilir tüm durumlar, `step` ile genel değer yinelemesi (bekleyen hedefte kırpılır).
    Döner: {MState: P(0. takım kazanır)}."""
    T = cfg.target
    clamp = lambda s: MState(s.score, (min(s.pending[0], T), min(s.pending[1], T)), s.unlocked, s.start, s.winner)
    trans: Dict[MState, List[Tuple[float, MState]]] = {}
    todo = [initial_state()]
    seen = {todo[0]}
    while todo:
        st = todo.pop()
        out = []
        if st.winner is None:
            sigs = [(0, 0.5), (1, 0.5)] if st.start is None else [(st.start, 1.0)]
            for s, ws in sigs:
                kind = "forced" if st.start is None else "free"
                d, t = dist.dec[kind], dist.tie[kind]
                for wr in (0, 1):
                    for nr in (0, 1):
                        for k in np.nonzero(d[wr, nr])[0]:
                            w, ns = (s if wr == 0 else 1 - s), (s if nr == 0 else 1 - s)
                            out.append((ws * d[wr, nr, k], clamp(step(st, w, int(k), 0, ns, cfg))))
                for nr in (0, 1):
                    for pot in np.nonzero(t[nr])[0]:
                        ns = s if nr == 0 else 1 - s
                        out.append((ws * t[nr, pot], clamp(step(st, None, 0, int(pot), ns, cfg))))
        trans[st] = out
        for _, n in out:
            if n not in seen:
                seen.add(n)
                todo.append(n)
    V = {s: (0.5 if s.winner is None else float(s.winner == 0)) for s in trans}
    for _ in range(max_sweeps):
        res = 0.0
        for s, out in trans.items():
            if s.winner is not None:
                continue
            v = sum(q * V[n] for q, n in out)
            res = max(res, abs(v - V[s]))
            V[s] = v
        if res < tol:
            break
    return V


# ── E-04r_l1 self-play verisi (kaynak) ve yükleme ─────────────

def selfplay_match_log(w, prefix: str, i: int, epsilon: float, max_rounds: int = 300):
    """Paylaşımlı politikayla bir eğitim tarzı maç (dmc.selfplay_match ile aynı aktör) oynar, tur sonuçlarını kaydeder.
    Döner: (turlar: her biri öncesi durum + sonuç, maçın kazananı)."""
    import random
    from .dmc import _round_decisions
    m = Match(f"{prefix}:{i}")
    rounds = []
    while m.winner is None and m.round_no < max_rounds:
        before = state_of(m)
        r = m.new_round()
        rng = random.Random(f"{prefix}:{i}:{m.round_no}:eps")
        _round_decisions(r, w, rng, epsilon, m.public_state())
        res = r.result
        rounds.append({"before": before, "forced": r.forced_opening is not None, "starter_team": team_of(r.starter),
                       "winner": res.winner_team, "points": res.points, "pot": res.pot_add,
                       "next_team": team_of(res.last_tile_player)})
        m.apply_result(res)
    return rounds, m.winner


def selfplay_log_chunk(model_path: str, prefix: str, start: int, stop: int, epsilon: float):
    from .dmc import load_cached
    w = load_cached(model_path)
    return [selfplay_match_log(w, prefix, i, epsilon) for i in range(start, stop)]


_CACHE: Dict[str, Tuple[float, MET]] = {}


def load(path: str, disk_cache: bool = True) -> MET:
    """JSON (dağılım) → MET. Çözüm süreç içinde (yol + değişiklik zamanı) ve diskte (`checkpoints/met/`, JSON
    içeriğinin sha256'sı ile adlandırılır, yani dağılım değişirse yeniden çözülür) önbelleğe alınır."""
    import hashlib
    mt = os.path.getmtime(path)
    hit = _CACHE.get(path)
    if hit is None or hit[0] != mt:
        raw = open(path, "rb").read()
        d = json.loads(raw.decode("utf-8"))
        cfg = MatchConfig(**d["config"]) if "config" in d else None
        dist = OutcomeDist.from_json(d)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cpath = os.path.join(root, "checkpoints", "met", hashlib.sha256(raw).hexdigest()[:16] + ".npz")
        met = None
        if disk_cache and os.path.exists(cpath):
            try:
                z = np.load(cpath)
                met = MET.__new__(MET)
                met.dist, met.cfg = dist, cfg or MatchConfig()
                met.V11, met.V10, met.V00 = z["V11"], z["V10"], z["V00"]
                met.sweeps, met.residual = int(z["sweeps"]), float(z["residual"])
            except Exception:          # bozuk/yarım önbellek: yeniden çöz
                met = None
        if met is None:
            met = MET(dist, cfg)
            if disk_cache:
                os.makedirs(os.path.dirname(cpath), exist_ok=True)
                tmp = cpath + f".{os.getpid()}.tmp.npz"
                np.savez(tmp, V11=met.V11, V10=met.V10, V00=met.V00, sweeps=met.sweeps, residual=met.residual)
                os.replace(tmp, cpath)
        met.slope = float(d.get("calibration", {}).get("slope", 1.0))
        _CACHE[path] = (mt, met)
    return _CACHE[path][1]
