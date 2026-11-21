"""
Özellik kodlayıcı F0..F3 ve aksiyon maskesi (Faz 5 hazırlığı, D-24). YALNIZ Observation (ve public maç durumu) alır:
Round verilirse TypeError. Gizli ellere erişim yapısal olarak yok; T-F2 farklı gizli ellerle aynı gözlemin aynı
vektörü verdiğini doğrular.

Kademeler (her biri bir öncekinin önekidir, ablation için):
  F0  ham gözlem: kendi el, masa, kim ne oynadı, pas kanıtı, el boyutları, uçlar, sıra, son 4 olay, açan, maç durumu
  F1  F0 + K1 kesin çıkarım: el/masa/görünmeyen sayı histogramları, kontrol, ölü sayı, possible, kesin kapatma
  F2  F1 + K2 inanç: taş×oyuncu marjinalleri, beklenen el değerleri (R-12 dahil, tam), belirsizlik (log sayım)
  F3  F2 + K3 sayaçlar: açtı, üstüne oynadı, oynadığı sayılar, ilk hamle çift mi, pas sayısı

Ayrıca `encode_events(obs)`: olay günlüğünün tamamı, eskiden yeniye, olay başına sabit genişlikte parça (dizi modeli,
LSTM/transformer girdisi). F-vektörü kesin özetleri taşır; dizi yalnız sıra/zamanlama bilgisini ekler.

Oyuncular GÖRELİ sırayla kodlanır: 0 = ben, 1 = sonraki (seat+1, sağdaki rakip), 2 = partner, 3 = önceki (seat+3).
Stratejiler (S-xx) kural olarak kodlanmaz (D-22); bu bloklar yalnız uzmanın baktığı BİLGİYİ ajana verir.

Aksiyon uzayı (57): indeks = 2·taş_indeksi + (0: 'L', 1: 'R'), pas = 56. taş_indeksi = ALL_TILES sırası.
Maske Round.legal_moves ile birebir (T-M1): boş masada açılış 'R' ile, uçlar eşitken yalnız kanonik 'L' (D-18).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .engine import (ALL_TILES, PASS, ZERO_ZERO, ZERO_ZERO_SINGLE_VALUE, Move, Observation, Tile,
                     legal_moves_for, norm, team_of)
from .info import Info

TILE_INDEX: Dict[Tile, int] = {t: i for i, t in enumerate(ALL_TILES)}
N_TILES = 28
N_ACTIONS = 57
PASS_INDEX = 56
LEVELS = ("F0", "F1", "F2", "F3")
RECENT_EVENTS = 4
EVENT_WIDTH = 4 + 1 + N_TILES + 2 + 15             # oyuncu (göreli), pas, taş, taraf, olaydan ÖNCEKİ uçlar
START_COUNT = 399072960                           # 21!/(7!)^3: başlangıçtaki dağılım sayısı (T-B1)
HAND_VALUE_SCALE = 100.0                          # el değeri ölçeği (7 taşın en büyüğü 73)


def rel(me: int, q: int) -> int:
    return (q - me) % 4


def seat_of(me: int, r: int) -> int:
    return (me + r) % 4


def tile_value(t: Tile, hand_size: int) -> int:
    """Taşın el değerine katkısı. R-12: el tek taşsa ve o taş 0|0 ise 10."""
    if hand_size == 1 and norm(t) == ZERO_ZERO:
        return ZERO_ZERO_SINGLE_VALUE
    return t[0] + t[1]


# ── aksiyon maskesi ──────────────────────────────────────────

def move_index(move: Move, left_end: int = -1, right_end: int = -2) -> int:
    """Hamle → aksiyon indeksi. Uçlar eşitken 'R' aynalı yerleşimdir ve kanonik 'L' indeksine eşlenir (D-18)."""
    if move.is_pass:
        return PASS_INDEX
    side = move.side
    if side == "R" and left_end >= 0 and left_end == right_end:
        side = "L"
    return 2 * TILE_INDEX[norm(move.tile)] + (0 if side == "L" else 1)


def index_move(i: int) -> Move:
    if i == PASS_INDEX:
        return PASS
    return Move(ALL_TILES[i // 2], "L" if i % 2 == 0 else "R")


def mask_from_legal(legal: Sequence[Move]) -> List[int]:
    m = [0] * N_ACTIONS
    for mv in legal:
        m[move_index(mv)] = 1
    return m


def legal_from_obs(obs: Observation, forced_opening: Optional[Tile] = None) -> List[Move]:
    """Gözlemden yasal hamleler (Round.legal_moves ile aynı kural). forced_opening public bilgidir (R-04: maç başında
    1|1 sahibi 1|1 ile açar); arena/Round bunu bilir, gözlem nesnesinde yoktur."""
    if not isinstance(obs, Observation):
        raise TypeError("legal_from_obs yalnız Observation alır")
    if obs.current_player != obs.player:
        return []
    hand = sorted(obs.hand)
    if not obs.board:
        if forced_opening is not None:
            return [Move(norm(forced_opening), "R")]
        return [Move(t, "R") for t in hand]
    return legal_moves_for(obs.hand, obs.left_end, obs.right_end)


def action_mask(obs: Observation, forced_opening: Optional[Tile] = None) -> List[int]:
    return mask_from_legal(legal_from_obs(obs, forced_opening))


# ── özellik blokları ─────────────────────────────────────────

def _onehot(n: int, k: int) -> List[float]:
    v = [0.0] * n
    if 0 <= k < n:
        v[k] = 1.0
    return v


def _tiles(ts) -> List[float]:
    v = [0.0] * N_TILES
    for t in ts:
        v[TILE_INDEX[norm(t)]] = 1.0
    return v


def _ends(lv: int, rv: int) -> List[float]:
    """[masa boş] + sol uç one-hot 7 + sağ uç one-hot 7 (boş masada uçlar -1)."""
    return [1.0 if lv < 0 else 0.0] + _onehot(7, lv) + _onehot(7, rv)


def event_token(me: int, e) -> List[float]:
    """Tek olay, gözlemciye göre: [oyuncu göreli one-hot 4, pas, taş 28, taraf L/R, olaydan önceki uçlar 15].
    Pasın anlamı (hangi sayılar yok) olaydan önceki uçlardadır, bu yüzden uçlar parçanın içinde."""
    play = e.kind == "play"
    return (_onehot(4, rel(me, e.player))
            + [0.0 if play else 1.0]
            + (_tiles([e.tile]) if play else [0.0] * N_TILES)
            + ([1.0 if e.side == "L" else 0.0, 1.0 if e.side == "R" else 0.0] if play else [0.0, 0.0])
            + _ends(e.left_end, e.right_end))


def encode_events(obs: Observation) -> List[List[float]]:
    """Olay günlüğünün tamamı, ESKİDEN YENİYE, olay başına EVENT_WIDTH'lik parça (değişken uzunluk). Dizi modeli
    (LSTM/transformer) girdisi; F-vektörüyle birlikte kullanılmak üzere (melez, 07 ablation). Yalnız Observation."""
    if not isinstance(obs, Observation):
        raise TypeError("encode_events yalnız Observation alır (gizli bilgi sızıntısını önlemek için)")
    return [event_token(obs.player, e) for e in obs.events]


def _match_block(me: int, match: Optional[dict]) -> List[float]:
    """Public maç durumu (D-09, D-11), takım göreli: [bilindi, skor biz/onlar, bekleyen biz/onlar, kilit biz/onlar,
    tur no]. 101'e uzaklık skorun doğrusal fonksiyonu olduğu için ayrıca verilmez."""
    if not match:
        return [0.0] * 8
    us, them = team_of(me), 1 - team_of(me)
    tgt = float(match.get("target", 101))
    sc, pe, un = match["score"], match["pending"], match["unlocked"]
    return [1.0, sc[us] / tgt, sc[them] / tgt, pe[us] / tgt, pe[them] / tgt,
            float(un[us]), float(un[them]), min(match.get("round_no", 0), 20) / 20.0]


def _f0(obs: Observation, info: Info, match: Optional[dict]) -> List[Tuple[str, List[float]]]:
    me = obs.player
    ex = info.exact
    blocks = [("hand", _tiles(obs.hand)), ("board", _tiles(ex.played))]
    for r in (1, 2, 3):
        blocks.append((f"played_by_r{r}", _tiles(ex.played_by[seat_of(me, r)])))
    for r in (1, 2, 3):
        blocks.append((f"missing_r{r}", [1.0 if n in ex.missing[seat_of(me, r)] else 0.0 for n in range(7)]))
    blocks.append(("hand_sizes", [obs.hand_sizes[seat_of(me, r)] / 7.0 for r in range(4)]))
    empty = not obs.board
    blocks.append(("ends", [1.0 if empty else 0.0] + _onehot(7, obs.left_end) + _onehot(7, obs.right_end)))
    blocks.append(("turn", [1.0 if obs.current_player == me else 0.0, obs.consecutive_passes / 3.0,
                            len(ex.played) / N_TILES]))
    recent: List[float] = []
    evs = list(obs.events[-RECENT_EVENTS:])[::-1]            # en yeni önce
    for k in range(RECENT_EVENTS):
        recent += event_token(me, evs[k]) if k < len(evs) else [0.0] * EVENT_WIDTH
    blocks.append(("recent", recent))
    blocks.append(("opener", _onehot(4, rel(me, obs.events[0].player)) if obs.events else [0.0] * 4))
    blocks.append(("match", _match_block(me, match)))
    return blocks


def _f1(obs: Observation, info: Info) -> List[Tuple[str, List[float]]]:
    me = obs.player
    ex = info.exact
    hand = [norm(t) for t in obs.hand]
    poss = info.possible()
    blocks = [
        ("mine_with", [x / 7.0 for x in ex.mine_with]),
        ("played_with", [x / 7.0 for x in ex.played_with]),
        ("unseen_with", [x / 7.0 for x in ex.unseen_with]),
        ("control", [1.0 if c else 0.0 for c in ex.control]),
        ("dead", [1.0 if d else 0.0 for d in ex.dead]),
        ("hand_summary", [sum(1 for t in hand if t[0] == t[1]) / 7.0,
                          sum(1 for n in range(7) if ex.mine_with[n] > 0) / 7.0,
                          sum(tile_value(t, len(hand)) for t in hand) / HAND_VALUE_SCALE]),
    ]
    for r in (1, 2, 3):
        q = seat_of(me, r)
        blocks.append((f"possible_r{r}", [1.0 if t in poss and q in poss[t] else 0.0 for t in ALL_TILES]))
    closing = info.certain_closing_moves()
    blocks.append(("closing", [1.0 if closing else 0.0] + [float(x) for x in mask_from_legal(closing)]))
    return blocks


def expected_hand_values(info: Info) -> Dict[int, float]:
    """Diğer üç oyuncunun beklenen el değeri, K2 tekdüze inanç altında TAM (R-12 dahil: el tek taşsa 0|0 = 10)."""
    marg = info.marginals()
    out = {}
    for q in (seat_of(info.obs.player, r) for r in (1, 2, 3)):
        size = info.obs.hand_sizes[q]
        out[q] = sum(p[q] * tile_value(t, size) for t, p in marg.items())
    return out


def _f2(obs: Observation, info: Info) -> List[Tuple[str, List[float]]]:
    me = obs.player
    marg = info.marginals()
    blocks = []
    for r in (1, 2, 3):
        q = seat_of(me, r)
        blocks.append((f"marg_r{r}", [marg[t][q] if t in marg else 0.0 for t in ALL_TILES]))
    ev = expected_hand_values(info)
    mine = sum(tile_value(t, len(obs.hand)) for t in obs.hand)
    ours = mine + ev[seat_of(me, 2)]
    theirs = ev[seat_of(me, 1)] + ev[seat_of(me, 3)]
    blocks.append(("exp_values", [ev[seat_of(me, r)] / HAND_VALUE_SCALE for r in (1, 2, 3)]
                   + [ours / HAND_VALUE_SCALE, theirs / HAND_VALUE_SCALE]))
    total = info.belief.total()
    blocks.append(("uncertainty", [math.log(total) / math.log(START_COUNT)]))
    return blocks


def _f3(obs: Observation, info: Info) -> List[Tuple[str, List[float]]]:
    me = obs.player
    s = info.soft
    blocks = []
    for name, arr in (("opened", s.opened), ("covered", s.covered), ("plays_with", s.plays_with)):
        blocks.append((name, [arr[seat_of(me, r)][n] / 7.0 for r in range(4) for n in range(7)]))
    fpd: List[float] = []
    for r in range(4):
        v = s.first_play_double[seat_of(me, r)]
        fpd += [1.0 if v is None else 0.0, 1.0 if v is False else 0.0, 1.0 if v is True else 0.0]
    blocks.append(("first_play_double", fpd))
    blocks.append(("n_passes", [s.n_passes[seat_of(me, r)] / 7.0 for r in range(4)]))
    return blocks


def encode_blocks(obs: Observation, level: str = "F3", match: Optional[dict] = None,
                  info: Optional[Info] = None) -> List[Tuple[str, List[float]]]:
    if not isinstance(obs, Observation):
        raise TypeError("kodlayıcı yalnız Observation alır (gizli bilgi sızıntısını önlemek için)")
    if level not in LEVELS:
        raise ValueError(f"bilinmeyen kademe: {level}")
    info = info or Info(obs)
    k = LEVELS.index(level)
    blocks = _f0(obs, info, match)
    if k >= 1:
        blocks += _f1(obs, info)
    if k >= 2:
        blocks += _f2(obs, info)
    if k >= 3:
        blocks += _f3(obs, info)
    return blocks


def encode(obs: Observation, level: str = "F3", match: Optional[dict] = None,
           info: Optional[Info] = None) -> List[float]:
    return [x for _, v in encode_blocks(obs, level, match, info) for x in v]


def layout(level: str = "F3") -> List[Tuple[str, int, int]]:
    """(blok adı, başlangıç, bitiş) listesi. Boyutlar gözlemden bağımsızdır (T-F1)."""
    from .engine import Round, deal_hands
    obs = Round(deal_hands(0, 1), 0).observation(0)
    out, i = [], 0
    for name, v in encode_blocks(obs, level):
        out.append((name, i, i + len(v)))
        i += len(v)
    return out


def dim(level: str = "F3") -> int:
    return layout(level)[-1][2]
