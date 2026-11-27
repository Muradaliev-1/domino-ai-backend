"""
Davranış sondaları (E-11, D-39): uzman stratejileri (S-xx, 14 - Uzman Stratejileri) eğitilmiş ajanlarda kendiliğinden
ortaya çıkıyor mu? S-xx burada YALNIZ ÖLÇÜM HİPOTEZİDİR: hiçbir ajana kural olarak verilmez (D-22).

Sonda = (uygulanabilirlik koşulu, stratejiyle uyumlu hamle kümesi C). İkisi de YALNIZ Observation (+ Info, + public maç
durumu) ile hesaplanır: gizli el görülmez (Round verilirse TypeError). Bir durum bir sonda için ancak C, yasal hamlelerin
boş olmayan ÖZ alt kümesiyse sayılır (aksi halde seçim stratejiden bilgi taşımaz). Şans düzeyi |C| / |yasal|.

Ölçüm: aynı durumu (aynı gözlemi) her ajana sorarız (karşı-olgusal sorgu: oyunu etkilemez, ajan her durumda yeni ve
kendi tohumlu kurulur). Ajanın seçtiği hamle C'de mi → uyum (1/0). Fark = uyum − şans. Durumlar aynı maçtan geldiği
için bağımsız değil: güven aralığı maç kümeleriyle (CR1 küme-sağlam standart hata).

Sondalar (kimlik, kısa tanım):
  S-01   serbest açılış: en kalabalık sayıyı (elde o sayıyı içeren taş adedi en büyük) içeren taşla aç
  S-02   serbest açılış: çift taşla aç
  S-04   hamleden sonra elde en çok farklı sayı kalsın (her sayıdan bulundur, pas riski)
  S-06   bir uçta kontrol ettiğim sayı (kalan tüm taşları bende) var: o ucu kapatma, diğer uca oyna
  S-07   sonraki rakibin (seat+1) pas geçtiği bir sayıyı aç
  S-07p  karşılaştırma: önceki rakibin (seat+3) pas geçtiği bir sayıyı aç (14: ikincil, zincirli)
  S-08   partnerin pas geçtiği sayı açık uçta: o uçların sayısını azalt (kapat)
  S-09   partnerin açtığı (son değiştirdiği) uç, partnerin pas geçmediği sayıysa: o ucu kapatma
  S-11w  kesin kapatan hamle var ve kazanma olasılığı >= 0,6: kapat
  S-11l  kesin kapatan her hamlede kaybetme olasılığı >= 0,6: kapatma
  S-14   elde <= 3 taş ve 0|0 oynanabilir: 0|0'ı oyna (tek kalırsa 10 sayılır, R-12)
  S-19   açtığın sayıdan elde taş kalsın (açılan sayı güç sinyali); alt grup: partnerin <= 2 taşı var
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .engine import Move, Observation

PROBES: Tuple[str, ...] = ("S-01", "S-02", "S-04", "S-06", "S-07", "S-07p", "S-08", "S-09",
                           "S-11w", "S-11l", "S-14", "S-19")
CLOSE_P = 0.6            # S-11 eşiği (önceden yazıldı)
CLOSE_SAMPLES = 500      # Info.close_outcome örnek sayısı
S14_MAX_HAND = 3
SMALL_PARTNER = 2        # S-19 alt grubu: partnerin taş sayısı <= 2


def opened_numbers(obs: Observation, mv: Move) -> Tuple[int, ...]:
    """Hamlenin açtığı (yeni uç yaptığı) sayı(lar). Açılışta taşın iki sayısı."""
    a, b = mv.tile
    if obs.left_end < 0:
        return (a, b) if a != b else (a,)
    matched = obs.left_end if mv.side == "L" else obs.right_end
    return (a + b - matched,)


def ends_after(obs: Observation, mv: Move) -> Tuple[int, int]:
    a, b = mv.tile
    if obs.left_end < 0:
        return (a, b)
    if mv.side == "L":
        return (a + b - obs.left_end, obs.right_end)
    return (obs.left_end, a + b - obs.right_end)


def end_owners(obs: Observation) -> Tuple[Optional[int], Optional[int]]:
    """Şu anki sol/sağ ucu en son kim belirledi (olay günlüğünden; açılışta iki uç da açanın)."""
    left = right = None
    for e in obs.events:
        if e.kind != "play":
            continue
        if e.left_end < 0:
            left = right = e.player
        elif e.side == "L":
            left = e.player
        else:
            right = e.player
    return left, right


def _numbers(tiles) -> set:
    return {n for t in tiles for n in t}


@dataclass
class ProbeHit:
    compliant: FrozenSet[int]          # yasal listede uyumlu hamlelerin indisleri
    n_legal: int
    tags: Dict[str, object] = field(default_factory=dict)

    @property
    def chance(self) -> float:
        return len(self.compliant) / self.n_legal


def probe_sets(obs: Observation, legal: Sequence[Move], match: Optional[dict] = None,
               rng: Optional[random.Random] = None) -> Dict[str, ProbeHit]:
    """Bu durumda uygulanabilen sondalar → uyumlu hamle indisleri. YALNIZ Observation (+ Info, public maç durumu).
    rng yalnız S-11'in kapatma sonucu örneklemesi için (Info.close_outcome); verilmezse S-11 atlanır."""
    if not isinstance(obs, Observation):
        raise TypeError("sondalar yalnızca Observation alır (gizli bilgi sızıntısını önlemek için)")
    legal = list(legal)
    if len(legal) < 2 or any(m.is_pass for m in legal):
        return {}
    from .info import Info
    info = Info(obs)
    ex = info.exact
    me = obs.player
    partner, nxt, prv = (me + 2) % 4, (me + 1) % 4, (me + 3) % 4
    hand = list(obs.hand)
    lv, rv = obs.left_end, obs.right_end
    opening = lv < 0
    idx = range(len(legal))
    cand: Dict[str, Tuple[set, dict]] = {}

    if opening:
        top = max(ex.mine_with)
        strong = {n for n in range(7) if ex.mine_with[n] == top}
        cand["S-01"] = ({i for i in idx if set(legal[i].tile) & strong}, {})
        cand["S-02"] = ({i for i in idx if legal[i].tile[0] == legal[i].tile[1]}, {})
    else:
        # S-04: hamleden sonra elde kalan farklı sayı adedi en büyük olanlar
        cov = [len(_numbers([t for t in hand if t != legal[i].tile])) for i in idx]
        best = max(cov)
        cand["S-04"] = ({i for i in idx if cov[i] == best}, {})
        # S-06: kontrol ettiğim sayı bir uçta (uçlar farklıyken): o uca oynama
        if lv != rv:
            ctrl_sides = {s for s, v in (("L", lv), ("R", rv)) if ex.control[v]}
            if len(ctrl_sides) == 1:
                cand["S-06"] = ({i for i in idx if legal[i].side not in ctrl_sides}, {})
        # S-07 / S-07p: rakibin pas geçtiği sayıyı aç
        for pid, q in (("S-07", nxt), ("S-07p", prv)):
            miss = ex.missing[q]
            if miss:
                cand[pid] = ({i for i in idx if set(opened_numbers(obs, legal[i])) & miss}, {})
        # S-08: partnerin pas geçtiği sayının açık olduğu uçların sayısını azalt
        pm = ex.missing[partner]
        pre = (lv in pm) + (rv in pm)
        if pre:
            # bağlam etiketleri (E-11b, Q-35 b): x = uçlarda açık olan, partnerin pas geçtiği sayı(lar)
            xs = {v for v in (lv, rv) if v in pm}
            tags = {"x_control": all(ex.control[x] for x in xs),            # x'in kalan tüm taşları bende (S-06 ile çatışma)
                    "next_missing_x": any(x in ex.missing[nxt] for x in xs),  # sonraki rakip de x'ten yok (S-07 ile çatışma)
                    "my_x_ge2": sum(1 for t in hand if set(t) & xs) >= 2,     # elimde en az 2 x taşı
                    "partner_small": obs.hand_sizes[partner] <= SMALL_PARTNER,
                    "my_small": len(hand) <= 2}
            tags["clean"] = not tags["x_control"] and not tags["next_missing_x"]
            cand["S-08"] = ({i for i in idx if sum(v in pm for v in ends_after(obs, legal[i])) < pre}, tags)
        # S-09: partnerin açtığı ucu (pas geçmediği sayıysa) kapatma
        if lv != rv:
            lo, ro = end_owners(obs)
            psides = {s for s, v, o in (("L", lv, lo), ("R", rv, ro)) if o == partner and v not in pm}
            if len(psides) == 1:
                cand["S-09"] = ({i for i in idx if legal[i].side not in psides}, {})
        # S-11: kesin kapatma ve beklenen sonuç (K1 + K2)
        if rng is not None:
            closing = info.certain_closing_moves()
            if closing:
                outs = {mv: info.close_outcome(mv, rng, CLOSE_SAMPLES) for mv in closing}
                good = {mv for mv, o in outs.items() if o.p_win >= CLOSE_P}
                tags = {"p_win": max(o.p_win for o in outs.values()), "p_loss": min(o.p_loss for o in outs.values()),
                        "unlocked_any": bool(any(match.get("unlocked", ()))) if match else None}
                if good:
                    cand["S-11w"] = ({i for i in idx if legal[i] in good}, tags)
                if all(o.p_loss >= CLOSE_P for o in outs.values()):
                    cand["S-11l"] = ({i for i in idx if legal[i] not in outs}, tags)
        # S-19: açtığın sayıdan elde taş kalsın
        comp = set()
        for i in idx:
            rest = [t for t in hand if t != legal[i].tile]
            if set(opened_numbers(obs, legal[i])) & _numbers(rest):
                comp.add(i)
        cand["S-19"] = (comp, {"partner_small": obs.hand_sizes[partner] <= SMALL_PARTNER})
    # S-14: 0|0 (açılışta da olabilir, ama açılışta el 7 taş: koşul el boyutuyla dışlanır)
    if len(hand) <= S14_MAX_HAND:
        cand["S-14"] = ({i for i in idx if legal[i].tile == (0, 0)}, {})

    out = {}
    for pid, (c, tags) in cand.items():
        if 0 < len(c) < len(legal):
            out[pid] = ProbeHit(frozenset(c), len(legal), dict(tags))
    return out


# ── istatistik ───────────────────────────────────────────────

@dataclass
class ClusterStat:
    n: int
    clusters: int
    mean: float
    se: float

    @property
    def ci95(self) -> Tuple[float, float]:
        return (self.mean - 1.96 * self.se, self.mean + 1.96 * self.se)

    def ci(self, z: float) -> Tuple[float, float]:
        return (self.mean - z * self.se, self.mean + z * self.se)


def cluster_stat(xs: Sequence[float], groups: Sequence[object]) -> ClusterStat:
    """Ortalama ve küme-sağlam (CR1) standart hata: aynı maçtan gelen durumlar bağımsız sayılmaz.
    se^2 = G/(G-1) * sum_g (sum_{i in g} (x_i - m))^2 / n^2."""
    n = len(xs)
    if n == 0 or n != len(groups):
        raise ValueError("boş ya da uzunlukları farklı")
    m = sum(xs) / n
    acc: Dict[object, float] = {}
    for x, g in zip(xs, groups):
        acc[g] = acc.get(g, 0.0) + (x - m)
    G = len(acc)
    if G < 2:
        return ClusterStat(n, G, m, float("nan"))
    var = G / (G - 1) * sum(v * v for v in acc.values()) / (n * n)
    return ClusterStat(n, G, m, math.sqrt(var))


def verdict(st: ClusterStat, z: float = 1.96) -> str:
    """Önceden yazılan hüküm: GA tamamen 0'ın üstünde 'uyumlu', altında 'ters', aksi 'ayrılamıyor'."""
    lo, hi = st.ci(z)
    return "uyumlu" if lo > 0 else ("ters" if hi < 0 else "ayrılamıyor")


# ── durum toplama ve sorgu ───────────────────────────────────

def _query(name: str, seed: str, rnd, obs: Observation, legal: List[Move], match: Optional[dict]) -> Move:
    """Ajanı bu durum için yeni ve kendi tohumuyla kur, kararını al. Sızıntısız ajanlar yalnız Observation görür;
    yalnız is_oracle işaretli ajan (D-04, etiketli üst referans) durumun KOPYASINI alır (arena ile aynı kural)."""
    from .agents import make_agent
    ag = make_agent(name, seed)
    if getattr(ag, "is_oracle", False):
        return ag.act_oracle(rnd.clone(), list(legal), match)
    return ag.act(obs, list(legal), match)


def collect_match(table: str, gen: Tuple[str, str], seed, k: int, query_agents: Sequence[str],
                  max_rounds: int = 300, only: Optional[Sequence[str]] = None) -> List[dict]:
    """Tablo `table`'ın k. maçı: gen[0] takım 0'da, gen[1] takım 1'de (k tekse yer değiştirir) tam maç oynar. Her
    karar durumunda uygulanabilen sondalar hesaplanır ve durum query_agents'ın her birine sorulur. Döner: kayıt listesi.
    Tamamen tohumlu: aynı (table, seed, k) aynı kayıtları verir (süreç ve sıradan bağımsız).
    only: yalnız bu sondalar (E-11b): oyun aynen oynanır, yalnız bu sondaların durumları sorulur ve kaydedilir; sorgu
    tohumları durum kimliğine bağlı olduğundan aynı durumda aynı kararlar çıkar (E-11 ile yeniden üretim kapısı)."""
    from .agents import make_agent
    from .engine import Match, team_of
    a_team = k % 2
    names = [gen[0] if team_of(s) == a_team else gen[1] for s in range(4)]
    mseed = f"PROBE:{seed}:{table}:{k}"
    agents = [make_agent(names[s], f"{mseed}:s{s}") for s in range(4)]
    m = Match(mseed)
    recs: List[dict] = []
    while m.winner is None and m.round_no < max_rounds:
        rnd = m.new_round()
        ms = m.public_state()
        d = 0
        while not rnd.is_over:
            p = rnd.current_player
            legal = rnd.legal_moves()
            obs = rnd.observation(p)
            sid = f"{mseed}:r{m.round_no}:d{d}"
            need_rng = only is None or any(x.startswith("S-11") for x in only)
            hits = probe_sets(obs, legal, ms, random.Random(f"close:{sid}") if need_rng else None)
            if only is not None:
                hits = {pid: h for pid, h in hits.items() if pid in only}
            if hits:
                choice = {}
                for qn in query_agents:
                    mv = _query(qn, f"q:{sid}:{qn}", rnd, obs, legal, ms)
                    if mv not in legal:
                        raise ValueError(f"{qn} yasal olmayan hamle: {mv}")
                    choice[qn] = legal.index(mv)
                recs.append({
                    "table": table, "match": k, "round": m.round_no, "d": d, "player": p,
                    "n_legal": len(legal),
                    "probes": {pid: {"c": sorted(h.compliant), **h.tags} for pid, h in hits.items()},
                    "choice": choice,
                })
            ag = agents[p]
            mv = ag.act_oracle(rnd.clone(), list(legal), ms) if getattr(ag, "is_oracle", False) \
                else ag.act(obs, list(legal), ms)
            rnd.apply(mv)
            d += 1
        m.apply_result(rnd.result)
    return recs


def summarize(recs: Sequence[dict], agents: Sequence[str], greedy: Optional[str] = "greedy",
              min_n: int = 100, z_strict: float = 3.2) -> Dict[str, Dict[str, dict]]:
    """[sonda][ajan] → n, uyum, şans, fark (uyum − şans) ve küme-sağlam GA; hüküm (95%) ve sıkı hüküm (z_strict,
    çoklu karşılaştırma için Bonferroni); açgözlüden fark (aynı durumlarda eşleştirilmiş). Kümeler: (tablo, maç)."""
    out: Dict[str, Dict[str, dict]] = {}
    for pid in PROBES:
        rows = [r for r in recs if pid in r["probes"]]
        if not rows:
            continue
        groups = [(r["table"], r["match"]) for r in rows]
        chance = [len(r["probes"][pid]["c"]) / r["n_legal"] for r in rows]
        res: Dict[str, dict] = {}
        for a in agents:
            comp = [1.0 if r["choice"][a] in r["probes"][pid]["c"] else 0.0 for r in rows]
            lift = cluster_stat([c - s for c, s in zip(comp, chance)], groups)
            d = {"n": len(rows), "clusters": lift.clusters, "compliance": sum(comp) / len(rows),
                 "chance": sum(chance) / len(rows), "lift": lift.mean, "lift_ci95": list(lift.ci95),
                 "verdict": verdict(lift) if len(rows) >= min_n else "yetersiz veri",
                 "verdict_strict": verdict(lift, z_strict) if len(rows) >= min_n else "yetersiz veri"}
            if greedy and a != greedy and greedy in agents:
                gcomp = [1.0 if r["choice"][greedy] in r["probes"][pid]["c"] else 0.0 for r in rows]
                gd = cluster_stat([c - g for c, g in zip(comp, gcomp)], groups)
                d["minus_greedy"] = gd.mean
                d["minus_greedy_ci95"] = list(gd.ci95)
            res[a] = d
        out[pid] = res
    return out


def subgroup(recs: Sequence[dict], pid: str, tag: str, agents: Sequence[str]) -> Dict[str, dict]:
    """Betimleyici: sonda `pid`'de etiket tag True/False alt gruplarında fark (uyum − şans) ve GA."""
    out: Dict[str, dict] = {}
    for val in (True, False):
        rows = [r for r in recs if pid in r["probes"] and r["probes"][pid].get(tag) == val]
        if len(rows) < 2:
            continue
        groups = [(r["table"], r["match"]) for r in rows]
        chance = [len(r["probes"][pid]["c"]) / r["n_legal"] for r in rows]
        out[str(val)] = {}
        for a in agents:
            comp = [1.0 if r["choice"][a] in r["probes"][pid]["c"] else 0.0 for r in rows]
            st = cluster_stat([c - s for c, s in zip(comp, chance)], groups)
            out[str(val)][a] = {"n": len(rows), "compliance": sum(comp) / len(rows), "chance": sum(chance) / len(rows),
                                "lift": st.mean, "lift_ci95": list(st.ci95)}
    return out
