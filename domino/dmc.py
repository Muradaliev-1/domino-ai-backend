"""
DouZero tarzı Deep Monte Carlo (DMC) ajanı, Faz 5 (RQ2, D-28). Bu modül YALNIZ numpy gerektirir (aktör, ajan,
self-play); öğrenici (PyTorch) `domino/dmc_train.py`'dadır.

Ağ: MLP, girdi F-kademesi vektörü (`features.encode`, varsayılan F3 = 765), çıktı 57 Q değeri (aksiyon uzayı D-24).
Girdi mimarisi ablation'ı (E-04a, D-30): isteğe bağlı DİZİ KODLAYICI (`meta["seq"]`: LSTM ya da küçük transformer)
olay dizisini (`features.encode_events`) sabit boyutlu bir vektöre çevirir; bu vektör F-vektörüne eklenip MLP'ye girer.
Kademe "H" = yalnız kendi el (28), elle çıkarılmış özellik yok (kontrol kolu: ağ çıkarımı diziden kendisi yapmalı).
Q(s, a) ≈ hamleyi yapan oyuncunun TAKIMI açısından tur sonucunun beklentisi (L1: işaretli puan / REWARD_SCALE).
Seçim yalnız yasal hamlelerin indeksleri arasından (maske); eşit uçlarda aynalı 'R' kanonik indekse eşlenir (D-18),
seçilen indeks kendisine eşlenen İLK yasal hamleye çevrilir, yani her zaman `legal`'in bir elemanı döner.

Sızıntı yok: ajan ve aktör ağa yalnız `encode(Observation, ...)` ve `encode_events(Observation)` verir; kodlayıcılar
Round alırsa TypeError (T-F2). Aktör tam turu (Round) yalnız oyunu YÜRÜTMEK ve sonucu (ödül) okumak için tutar; ödül
oyun bittikten sonra okunur, gözlem vektörüne girmez (Monte Carlo hedefi, DouZero ile aynı).
Kural yok (D-22): ağ S-xx'leri kendisi bulmalı.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agents import Agent
from .engine import OPENING_TILE, Match, MatchConfig, Move, Observation, Round, deal_hands, team_of
from .features import EVENT_WIDTH, N_ACTIONS, _tiles, encode, encode_events, move_index

REWARD_SCALE = 100.0          # L1 hedefi: işaretli tur puanı / 100 (tur puanları çoğunlukla 0..~120)

VEC_LEVELS = ("H", "F0", "F1", "F2", "F3")
SEQ_DEFAULTS = {
    "lstm": {"kind": "lstm", "hidden": 128},
    "transformer": {"kind": "transformer", "d": 64, "layers": 2, "heads": 4, "ff": 128, "max_len": 128},
}
_DIM_CACHE: Dict[str, int] = {}


def vec_dim(level: str) -> int:
    """F-vektörü boyutu; "H" = yalnız kendi el (28)."""
    if level == "H":
        return 28
    if level not in _DIM_CACHE:
        from .features import dim
        _DIM_CACHE[level] = dim(level)
    return _DIM_CACHE[level]


def seq_out_dim(seq: Optional[dict]) -> int:
    if not seq:
        return 0
    if seq["kind"] == "lstm":
        return int(seq["hidden"])
    if seq["kind"] == "transformer":
        return 2 * int(seq["d"])                    # [ortalama havuz, son olay]
    raise ValueError(f"bilinmeyen dizi kodlayıcı: {seq['kind']}")


# ── dizi kodlayıcılar (numpy, torch modülleriyle birebir: T-DMC8) ─────────

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _layernorm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def lstm_embed(S: Dict[str, np.ndarray], toks: np.ndarray, hidden: int) -> np.ndarray:
    """Tek katmanlı LSTM (torch kapı sırası i, f, g, o), dizinin SON gizli durumu. Boş dizi → sıfır."""
    H = hidden
    h = np.zeros(H, np.float32)
    if len(toks) == 0:
        return h
    c = np.zeros(H, np.float32)
    Z = toks @ S["W_ih"].T + S["b_ih"] + S["b_hh"]
    Whh = S["W_hh"]
    for z in Z:
        z = z + Whh @ h
        i, f, g, o = _sigmoid(z[:H]), _sigmoid(z[H:2 * H]), np.tanh(z[2 * H:3 * H]), _sigmoid(z[3 * H:])
        c = f * c + i * g
        h = o * np.tanh(c)
    return h.astype(np.float32)


def transformer_embed(S: Dict[str, np.ndarray], toks: np.ndarray, cfg: dict) -> np.ndarray:
    """Ön-LN transformer kodlayıcı (çift yönlü dikkat, öğrenilmiş konum), çıktı [ortalama havuz, son olay].
    Boş dizi → sıfır. Dizi max_len'i aşarsa hata (bir turda en çok 28 + 3·27 + 4 = 113 olay)."""
    d, L, nh = int(cfg["d"]), int(cfg["layers"]), int(cfg["heads"])
    T = len(toks)
    if T == 0:
        return np.zeros(2 * d, np.float32)
    if T > int(cfg["max_len"]):
        raise ValueError("olay dizisi max_len'den uzun")
    x = toks @ S["emb.w"].T + S["emb.b"] + S["pos"][:T]
    dh = d // nh
    for l in range(L):
        y = _layernorm(x, S[f"l{l}.ln1.g"], S[f"l{l}.ln1.b"])
        qkv = y @ S[f"l{l}.qkv.w"].T + S[f"l{l}.qkv.b"]
        q, k, v = (qkv[:, j * d:(j + 1) * d].reshape(T, nh, dh).transpose(1, 0, 2) for j in range(3))
        a = q @ k.transpose(0, 2, 1) / np.sqrt(dh)
        a = np.exp(a - a.max(-1, keepdims=True))
        a = a / a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(T, d)
        x = x + o @ S[f"l{l}.out.w"].T + S[f"l{l}.out.b"]
        y = _layernorm(x, S[f"l{l}.ln2.g"], S[f"l{l}.ln2.b"])
        x = x + np.maximum(y @ S[f"l{l}.ff1.w"].T + S[f"l{l}.ff1.b"], 0.0) @ S[f"l{l}.ff2.w"].T + S[f"l{l}.ff2.b"]
    x = _layernorm(x, S["lnf.g"], S["lnf.b"])
    return np.concatenate([x.mean(0), x[-1]]).astype(np.float32)


# ── ağırlıklar (numpy) ───────────────────────────────────────

class Weights:
    """Ağırlıklar: layers = [(W (out,in), b (out,)), ...] MLP, gizli katmanlarda ReLU, son katman doğrusal (57).
    İsteğe bağlı dizi kodlayıcı: meta["seq"] (yapılandırma) + seq (ad → dizi). Varsa MLP girdisi
    [F-vektörü, kodlayıcı(olay dizisi)]."""

    def __init__(self, layers: Sequence[Tuple[np.ndarray, np.ndarray]], meta: Optional[dict] = None,
                 seq: Optional[Dict[str, np.ndarray]] = None):
        self.layers = [(np.asarray(W, dtype=np.float32), np.asarray(b, dtype=np.float32)) for W, b in layers]
        self.meta = dict(meta or {})
        if self.layers[-1][0].shape[0] != N_ACTIONS:
            raise ValueError("son katman 57 çıkışlı olmalı")
        self.level = self.meta.get("level", "F3")
        self.seq_cfg: Optional[dict] = self.meta.get("seq") or None
        self.seq = {k: np.asarray(v, dtype=np.float32) for k, v in (seq or {}).items()}
        if bool(self.seq_cfg) != bool(self.seq):
            raise ValueError("dizi kodlayıcı yapılandırması ve ağırlıkları birlikte verilmeli")

    @property
    def in_dim(self) -> int:
        return self.layers[0][0].shape[1]

    @property
    def vec_dim(self) -> int:
        return self.in_dim - seq_out_dim(self.seq_cfg)

    def seq_embed(self, toks) -> np.ndarray:
        toks = np.asarray(toks, dtype=np.float32).reshape(-1, EVENT_WIDTH)
        if self.seq_cfg["kind"] == "lstm":
            return lstm_embed(self.seq, toks, int(self.seq_cfg["hidden"]))
        return transformer_embed(self.seq, toks, self.seq_cfg)

    def forward(self, x: np.ndarray, toks=None) -> np.ndarray:
        """x: (d,) ya da (n, d) → (57,) ya da (n, 57). Dizi kodlayıcı varsa toks: (T, 50) ya da n tanelik liste."""
        h = np.asarray(x, dtype=np.float32)
        if self.seq_cfg:
            if toks is None:
                raise ValueError("bu ağ olay dizisi ister")
            e = self.seq_embed(toks) if h.ndim == 1 else np.stack([self.seq_embed(t) for t in toks])
            h = np.concatenate([h, e], axis=-1)
        last = len(self.layers) - 1
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < last:
                h = np.maximum(h, 0.0)
        return h

    def save(self, path: str) -> None:
        arrs = {}
        for i, (W, b) in enumerate(self.layers):
            arrs[f"W{i}"], arrs[f"b{i}"] = W, b
        for k, v in self.seq.items():
            arrs[f"S_{k}"] = v
        arrs["meta"] = np.frombuffer(json.dumps(self.meta).encode("utf-8"), dtype=np.uint8)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "wb") as f:
            np.savez(f, **arrs)

    @classmethod
    def load(cls, path: str) -> "Weights":
        z = np.load(path)
        n = sum(1 for k in z.files if k.startswith("W"))
        meta = json.loads(bytes(z["meta"]).decode("utf-8")) if "meta" in z.files else {}
        seq = {k[2:]: z[k] for k in z.files if k.startswith("S_")}
        return cls([(z[f"W{i}"], z[f"b{i}"]) for i in range(n)], meta, seq)


def random_seq_params(seq: dict, rng) -> Dict[str, np.ndarray]:
    """Dizi kodlayıcı için rastgele (testler) ağırlıklar; torch modülüyle aynı adlar ve şekiller."""
    def n(*shape, s=0.2):
        return rng.normal(0, s, shape).astype(np.float32)
    if seq["kind"] == "lstm":
        H = int(seq["hidden"])
        return {"W_ih": n(4 * H, EVENT_WIDTH), "W_hh": n(4 * H, H), "b_ih": n(4 * H), "b_hh": n(4 * H)}
    d, ff = int(seq["d"]), int(seq["ff"])
    S = {"emb.w": n(d, EVENT_WIDTH), "emb.b": n(d), "pos": n(int(seq["max_len"]), d),
         "lnf.g": 1 + n(d), "lnf.b": n(d)}
    for l in range(int(seq["layers"])):
        S.update({f"l{l}.ln1.g": 1 + n(d), f"l{l}.ln1.b": n(d), f"l{l}.qkv.w": n(3 * d, d), f"l{l}.qkv.b": n(3 * d),
                  f"l{l}.out.w": n(d, d), f"l{l}.out.b": n(d), f"l{l}.ln2.g": 1 + n(d), f"l{l}.ln2.b": n(d),
                  f"l{l}.ff1.w": n(ff, d), f"l{l}.ff1.b": n(ff), f"l{l}.ff2.w": n(d, ff), f"l{l}.ff2.b": n(d)})
    return S


def random_weights(level: str = "F3", hidden: Sequence[int] = (512, 512, 512, 512), seed=0,
                   seq: Optional[dict] = None) -> Weights:
    """Testler ve başlangıç için: He ilklendirmesi, deterministik. seq: SEQ_DEFAULTS'taki gibi bir yapılandırma."""
    import zlib
    rng = np.random.default_rng(seed if isinstance(seed, int) else zlib.crc32(str(seed).encode("utf-8")))
    sizes = [vec_dim(level) + seq_out_dim(seq)] + list(hidden) + [N_ACTIONS]
    layers = [(rng.normal(0, np.sqrt(2.0 / a), (b, a)).astype(np.float32), np.zeros(b, np.float32))
              for a, b in zip(sizes[:-1], sizes[1:])]
    meta = {"level": level, "hidden": list(hidden)}
    if seq:
        meta["seq"] = dict(seq)
    return Weights(layers, meta, random_seq_params(seq, rng) if seq else None)


_CACHE: Dict[str, Tuple[float, Weights]] = {}


def load_cached(path: str) -> Weights:
    """Arena her tur yeni ajan kurar; ağırlıklar yol + değişiklik zamanıyla önbelleğe alınır."""
    mt = os.path.getmtime(path)
    hit = _CACHE.get(path)
    if hit is None or hit[0] != mt:
        _CACHE[path] = (mt, Weights.load(path))
    return _CACHE[path][1]


# ── karar ────────────────────────────────────────────────────

def features_of(obs: Observation, level: str, match: Optional[dict] = None) -> np.ndarray:
    if level == "H":                                     # kontrol kolu: yalnız kendi el
        if not isinstance(obs, Observation):
            raise TypeError("kodlayıcı yalnız Observation alır (gizli bilgi sızıntısını önlemek için)")
        return np.asarray(_tiles(obs.hand), dtype=np.float32)
    return np.asarray(encode(obs, level, match), dtype=np.float32)


def events_of(obs: Observation) -> np.ndarray:
    """Olay dizisi (T, 50), eskiden yeniye (`features.encode_events`, yalnız Observation)."""
    return np.asarray(encode_events(obs), dtype=np.float32).reshape(-1, EVENT_WIDTH)


def q_values(w: Weights, obs: Observation, match: Optional[dict] = None) -> np.ndarray:
    """Ağın tek gözlem için 57 Q değeri (dizi kodlayıcı varsa olay dizisi de gözlemden). Maç durumu verilmemişse
    (arena L1) ve ağ maçla eğitildiyse (meta l1_match = "start"), L1 turu maçın ilk turu olarak kodlanır; eski
    (tur self-play'iyle eğitilmiş) ağlarda maç bloğu eskisi gibi sıfır kalır."""
    if match is None and w.meta.get("l1_match") == "start":
        match = initial_match_state()
    return w.forward(features_of(obs, w.level, match), events_of(obs) if w.seq_cfg else None)


def legal_indices(obs: Observation, legal: Sequence[Move]) -> List[int]:
    return [move_index(m, obs.left_end, obs.right_end) for m in legal]     # boş masada uçlar -1: eşleme yok


def choose(q: np.ndarray, idx: Sequence[int]) -> int:
    """Yasal indeksler arasında en büyük Q; eşitlikte legal sırasındaki ilk. Döner: legal içindeki konum."""
    best, best_q = 0, -np.inf
    for k, i in enumerate(idx):
        if q[i] > best_q:
            best, best_q = k, q[i]
    return best


class DMCAgent(Agent):
    """Öğrenilmiş ajan. Ad: 'dmc:<yol.npz>' (make_agent). Yalnız Observation + yasal hamle + public maç durumu."""
    name = "dmc"

    def __init__(self, seed=0, weights: Optional[Weights] = None, path: Optional[str] = None, epsilon: float = 0.0):
        self.w = weights if weights is not None else load_cached(path)
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.last_q: Optional[np.ndarray] = None

    def act(self, obs, legal, match=None):
        if not isinstance(obs, Observation):
            raise TypeError("DMCAgent yalnız Observation alır (gizli bilgi sızıntısını önlemek için)")
        if len(legal) == 1:
            return legal[0]
        q = q_values(self.w, obs, match)
        self.last_q = q
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return legal[self.rng.randrange(len(legal))]
        return legal[choose(q, legal_indices(obs, legal))]


# ── self-play aktörü (L1) ────────────────────────────────────

def l1_reward(res, team: int) -> float:
    """Turun `team` açısından L1 sonucu: kazanırsa +puan, kaybederse -puan, eşitlikte 0; ölçekli."""
    if res.winner_team is None:
        return 0.0
    return (res.points if res.winner_team == team else -res.points) / REWARD_SCALE


def train_deal(seed, i: int) -> Round:
    """Eğitim dağıtımı. Arena L1 ile aynı açılış kuralı (1|1 sahibi 1|1 ile açar) ama ayrı tohum uzayı
    ('DMC:' öneki; arena 'L1:' kullanır), yani eğitim ve değerlendirme dağıtımları ayrı."""
    hands = deal_hands(f"DMC:{seed}", i)
    starter = next(p for p, h in enumerate(hands) if OPENING_TILE in h)
    return Round(hands, starter, forced_opening=OPENING_TILE)


def _round_decisions(r: Round, w: Weights, rng: random.Random, epsilon: float, match: Optional[dict] = None):
    """Bir turu dört koltukta aynı ağla (paylaşımlı politika) ε-açgözlü oynar. Yalnız seçenekli kararlar (yasal
    >= 2) kaydedilir. Ağa yalnız gözlem + public maç durumu (`match`, tur başındaki `Match.public_state()`) gider.
    Döner: (özellik vektörleri, seçilen aksiyon indeksi, hamleyi yapanın takımı, olay dizileri)."""
    xs: List[np.ndarray] = []
    seqs: List[np.ndarray] = []
    acts: List[int] = []
    teams: List[int] = []
    use_seq = bool(w.seq_cfg)
    while not r.is_over:
        p = r.current_player
        legal = r.legal_moves()
        if len(legal) == 1:
            r.apply(legal[0])
            continue
        obs = r.observation(p)
        x = features_of(obs, w.level, match)
        toks = events_of(obs) if use_seq else np.zeros((0, EVENT_WIDTH), np.float32)
        idx = legal_indices(obs, legal)
        if rng.random() < epsilon:
            k = rng.randrange(len(legal))
        else:
            k = choose(w.forward(x, toks if use_seq else None), idx)
        xs.append(x)
        seqs.append(toks)
        acts.append(idx[k])
        teams.append(team_of(p))
        r.apply(legal[k])
    return xs, acts, teams, seqs


def selfplay_round(w: Weights, seed, i: int, epsilon: float):
    """Dört koltukta aynı ağ (paylaşımlı politika) ε-açgözlü bir L1 turu oynar (maç durumu yok). Yalnız seçenekli
    kararlar (yasal >= 2) örnek olur. Döner: (özellik vektörleri, seçilen aksiyon indeksi, MC hedefi, olay
    dizileri); dizi kodlayıcısız ağda olay dizileri boş (0, 50)."""
    r = train_deal(seed, i)
    rng = random.Random(f"DMC:{seed}:{i}:eps")
    xs, acts, teams, seqs = _round_decisions(r, w, rng, epsilon)
    ys = [l1_reward(r.result, t) for t in teams]
    return xs, acts, ys, seqs


# ── self-play aktörü, tam maç (L2 ince ayarı, D-33) ───────────

REWARDS = ("L1", "win", "score", "met")
MATCH_PREFIX = "DMCM"         # eğitim maçlarının tohum uzayı (arena L2 'L2:' kullanır)
# MET ödülü (D-35, Q-33): tur başına maç kazanma olasılığı değişimi; tablo E-04r_l1 self-play dağılımından
MET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "met",
                        "MET-E04r_l1.json")


def initial_match_state(config: Optional[MatchConfig] = None) -> dict:
    """Maçın ilk turundaki public maç durumu (arena `play_match`'in 1. turda ajana verdiği ile aynı). L1 turu,
    maçın ilk turudur (1|1 açılışı); maçla eğitilen ağlar (meta l1_match = "start") L1'de bunu görür."""
    cfg = config or MatchConfig()
    return {"score": (0, 0), "pending": (0, 0), "unlocked": (False, False), "round_no": 1, "target": cfg.target}


def match_reward(m: Match, team: int, kind: str) -> float:
    """Biten maçın `team` açısından sonucu. 'win': kazanırsa +1, kaybederse -1, bitmemişse 0.
    'score': (bizim skor - onların skoru) / hedef (101)."""
    if kind == "win":
        if m.winner is None:
            return 0.0
        return 1.0 if m.winner == team else -1.0
    if kind == "score":
        return (m.score[team] - m.score[1 - team]) / float(m.cfg.target)
    raise ValueError(f"bilinmeyen maç ödülü: {kind}")


def train_match(seed, i: int) -> Match:
    return Match(f"{MATCH_PREFIX}:{seed}:{i}")


def selfplay_match(w: Weights, seed, i: int, epsilon: float, reward: str = "win", max_rounds: int = 300):
    """Paylaşımlı politikayla tam bir eğitim maçı (101, tüm maç kuralları) oynar. Her karar o turun başındaki
    public maç durumunu görür. Hedef: reward = 'L1' ise kararın turunun L1 sonucu (kontrol kolu), 'win'/'score'
    ise maçın sonucu (`match_reward`), hamleyi yapan takım açısından. 'met' ise kararın turunun
    maç değeri değişimi: MET(tur sonrası) − MET(tur öncesi), hamleyi yapan takım açısından (tablo yalnız public maç
    durumu kullanır, `domino/met.py`). Döner: (xs, acts, ys, seqs, tur sayısı)."""
    if reward not in REWARDS:
        raise ValueError(f"bilinmeyen ödül: {reward}")
    if reward == "met":
        from . import met as met_mod
        table = met_mod.load(MET_PATH)
    m = train_match(seed, i)
    xs: List[np.ndarray] = []
    seqs: List[np.ndarray] = []
    acts: List[int] = []
    teams: List[int] = []
    ys: List[float] = []
    while m.winner is None and m.round_no < max_rounds:
        if reward == "met":
            st0 = met_mod.state_of(m)
        r = m.new_round()
        rng = random.Random(f"{MATCH_PREFIX}:{seed}:{i}:{m.round_no}:eps")
        rx, ra, rt, rs = _round_decisions(r, w, rng, epsilon, m.public_state())
        xs += rx
        acts += ra
        teams += rt
        seqs += rs
        if reward == "L1":
            ys += [l1_reward(r.result, t) for t in rt]
        m.apply_result(r.result)
        if reward == "met":
            st1 = met_mod.state_of(m)
            ys += [table.value(st1, t) - table.value(st0, t) for t in rt]
    if reward in ("win", "score"):
        ys = [match_reward(m, t, reward) for t in teams]
    return xs, acts, ys, seqs, m.round_no


def _pack(w: Weights, X, A, Y, S):
    d = w.vec_dim
    Ls = np.asarray([len(t) for t in S], dtype=np.int64)
    T = np.concatenate(S).astype(np.uint8) if S and Ls.sum() else np.zeros((0, EVENT_WIDTH), np.uint8)
    return (np.stack(X).astype(np.float32) if X else np.zeros((0, d), np.float32),
            np.asarray(A, dtype=np.int64), np.asarray(Y, dtype=np.float32), Ls, T)


def selfplay_chunk(w: Weights, seed, start: int, stop: int, epsilon: float):
    """[start, stop) turlarını oynar, dizileri birleştirir (çok süreçli aktör birimi).
    Döner: (X (n, d), A (n,), Y (n,), Ls (n,) olay dizisi uzunlukları, T (ΣLs, 50) uint8 olaylar art arda).
    Olay parçaları 0/1 olduğu için uint8 taşınır (süreçler arası aktarım 4 kat küçük)."""
    X: List[np.ndarray] = []
    A: List[int] = []
    Y: List[float] = []
    S: List[np.ndarray] = []
    for i in range(start, stop):
        xs, acts, ys, seqs = selfplay_round(w, seed, i, epsilon)
        X += xs
        A += acts
        Y += ys
        S += seqs
    return _pack(w, X, A, Y, S)


def selfplay_match_chunk(w: Weights, seed, start: int, stop: int, epsilon: float, reward: str = "win"):
    """[start, stop) maçlarını oynar. Döner: selfplay_chunk'taki 5'li + oynanan tur sayısı."""
    X: List[np.ndarray] = []
    A: List[int] = []
    Y: List[float] = []
    S: List[np.ndarray] = []
    n_rounds = 0
    for i in range(start, stop):
        xs, acts, ys, seqs, nr = selfplay_match(w, seed, i, epsilon, reward)
        X += xs
        A += acts
        Y += ys
        S += seqs
        n_rounds += nr
    return _pack(w, X, A, Y, S) + (n_rounds,)
