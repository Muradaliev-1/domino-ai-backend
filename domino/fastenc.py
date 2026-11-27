"""
Hızlı F3 kodlayıcı (Q-34 a, E-06b): `features.encode(obs, "F3", match)` ile BİT BİT aynı float32 vektörü, daha az iş.
Kararı değiştirmez; yalnız melezin yaprak değerini (karar başı ~180 ağ çağrısı) ucuzlatır.

Neden aynı: her öğe referansla aynı Python ifadesiyle (aynı tamsayılar, aynı bölmeler, aynı toplama sırası) hesaplanır,
sonra aynı float64 → float32 dönüşümünden geçer. Farklar yalnız tekrarlanan işin kaldırılması:
  1) olay günlüğü TEK geçişte okunur (kesin çıkarım `compute_exact` + K3 sayaçları `compute_soft` + pas kanıtı);
  2) ortak sayım (taş t, oyuncu i) yalnız t'nin izinli-oyuncu maskesine ve i'ye bağlıdır (`Belief.joint_count`),
     bu yüzden taş başına değil (maske, oyuncu) başına bir kez hesaplanır (en çok 7 × 3 DP çağrısı); referans aynı
     sayımları `possible()` için bir, `marginals()` için iki kez (F2 + beklenen el değerleri) taş başına yapar;
  3) bloklar ara listeler yerine önceden ayrılmış tek listeye yazılır.
SIZINTI YOK: yalnız Observation (Round gelirse TypeError) ve public maç durumu. Eşitlik testi T-HY9, mutantlar
`tools/mutation_check.py` "hızlı kodlayıcı".
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .belief import InconsistentObservation, _count
from .engine import ALL_TILES, Observation, legal_moves_for, new_ends_after, norm
from .features import (EVENT_WIDTH, HAND_VALUE_SCALE, N_TILES, RECENT_EVENTS, START_COUNT, TILE_INDEX,
                       _match_block, move_index, tile_value)

DIM_F3 = 765
_LOG_START = math.log(START_COUNT)
_ALL_SET = frozenset(ALL_TILES)
# blok başlangıçları (features.layout("F3") ile aynı; T-HY9 doğrular)
O_HAND, O_BOARD, O_PB, O_MISS, O_HS, O_ENDS, O_TURN, O_RECENT, O_OPENER, O_MATCH = (
    0, 28, 56, 140, 161, 165, 180, 183, 383, 387)
O_MINE, O_PLAYED, O_UNSEEN, O_CONTROL, O_DEAD, O_HSUM, O_POSS, O_CLOSE = 395, 402, 409, 416, 423, 430, 433, 517
O_MARG, O_EXP, O_UNC = 575, 659, 664
O_OPENED, O_COVERED, O_PW, O_FPD, O_NPASS = 665, 693, 721, 749, 761


def _event_into(v, off: int, me: int, e) -> None:
    """features.event_token'ın v[off:off+50]'ye yazılmış hali (v sıfırla başlar)."""
    v[off + (e.player - me) % 4] = 1.0
    if e.kind == "play":
        v[off + 5 + TILE_INDEX[norm(e.tile)]] = 1.0
        if e.side == "L":
            v[off + 33] = 1.0
        elif e.side == "R":
            v[off + 34] = 1.0
    else:
        v[off + 4] = 1.0
    lv, rv = e.left_end, e.right_end
    if lv < 0:
        v[off + 35] = 1.0
    if 0 <= lv < 7:
        v[off + 36 + lv] = 1.0
    if 0 <= rv < 7:
        v[off + 43 + rv] = 1.0


def encode_f3(obs: Observation, match: Optional[dict] = None) -> list:
    """features.encode(obs, "F3", match) ile aynı 765 öğelik float listesi."""
    if not isinstance(obs, Observation):
        raise TypeError("kodlayıcı yalnız Observation alır (gizli bilgi sızıntısını önlemek için)")
    me = obs.player
    v = [0.0] * DIM_F3
    events = obs.events
    # ── olay günlüğü, tek geçiş
    played_by = ([], [], [], [])
    missing = (set(), set(), set(), set())
    opened = [[0] * 7 for _ in range(4)]
    covered = [[0] * 7 for _ in range(4)]
    plays_with = [[0] * 7 for _ in range(4)]
    first_double = [None] * 4
    n_passes = [0] * 4
    for e in events:
        p = e.player
        if e.kind == "pass":
            n_passes[p] += 1
            missing[p].add(e.left_end)
            missing[p].add(e.right_end)
            continue
        a, b = e.tile
        played_by[p].append(norm(e.tile))
        pw = plays_with[p]
        pw[a] += 1
        if b != a:
            pw[b] += 1
        if first_double[p] is None:
            first_double[p] = (a == b)
        if e.left_end == -1:
            opened[p][a] += 1
            if b != a:
                opened[p][b] += 1
        else:
            matched = e.left_end if e.side == "L" else e.right_end
            covered[p][matched] += 1
            opened[p][a + b - matched] += 1
    played = set()
    for s in played_by:
        played.update(s)
    hand = obs.hand
    mine = set(norm(t) for t in hand)
    unseen = _ALL_SET - played - mine
    played_with = [0] * 7
    mine_with = [0] * 7
    unseen_with = [0] * 7
    for (a, b) in played:
        played_with[a] += 1
        if b != a:
            played_with[b] += 1
    for (a, b) in mine:
        mine_with[a] += 1
        if b != a:
            mine_with[b] += 1
    for (a, b) in unseen:
        unseen_with[a] += 1
        if b != a:
            unseen_with[b] += 1
    seats = (me, (me + 1) % 4, (me + 2) % 4, (me + 3) % 4)
    # ── F0
    for t in hand:
        v[O_HAND + TILE_INDEX[norm(t)]] = 1.0
    for t in played:
        v[O_BOARD + TILE_INDEX[t]] = 1.0
    for r in (1, 2, 3):
        off = O_PB + 28 * (r - 1)
        for t in played_by[seats[r]]:
            v[off + TILE_INDEX[t]] = 1.0
        off = O_MISS + 7 * (r - 1)
        for n in missing[seats[r]]:
            if 0 <= n < 7:
                v[off + n] = 1.0
    hs = obs.hand_sizes
    for r in range(4):
        v[O_HS + r] = hs[seats[r]] / 7.0
    if not obs.board:
        v[O_ENDS] = 1.0
    lv, rv = obs.left_end, obs.right_end
    if 0 <= lv < 7:
        v[O_ENDS + 1 + lv] = 1.0
    if 0 <= rv < 7:
        v[O_ENDS + 8 + rv] = 1.0
    v[O_TURN] = 1.0 if obs.current_player == me else 0.0
    v[O_TURN + 1] = obs.consecutive_passes / 3.0
    v[O_TURN + 2] = len(played) / N_TILES
    ne = len(events)
    for k in range(min(RECENT_EVENTS, ne)):
        _event_into(v, O_RECENT + k * EVENT_WIDTH, me, events[ne - 1 - k])
    if events:
        v[O_OPENER + (events[0].player - me) % 4] = 1.0
    v[O_MATCH:O_MATCH + 8] = _match_block(me, match)
    # ── F1 (K1)
    for n in range(7):
        v[O_MINE + n] = mine_with[n] / 7.0
        v[O_PLAYED + n] = played_with[n] / 7.0
        v[O_UNSEEN + n] = unseen_with[n] / 7.0
        if unseen_with[n] == 0:
            if mine_with[n] > 0:
                v[O_CONTROL + n] = 1.0
            else:
                v[O_DEAD + n] = 1.0
    nh = len(hand)
    hn = [norm(t) for t in hand]
    v[O_HSUM] = sum(1 for t in hn if t[0] == t[1]) / 7.0
    v[O_HSUM + 1] = sum(1 for n in range(7) if mine_with[n] > 0) / 7.0
    v[O_HSUM + 2] = sum(tile_value(t, nh) for t in hn) / HAND_VALUE_SCALE
    # K2: Belief(others) ile aynı maskeler ve kapasiteler
    others = seats[1:]
    forb = [missing[q] for q in others]
    caps = tuple(hs[q] for q in others)
    gs = [0] * 8
    tmask = []
    for t in sorted(unseen):
        m = 0
        for i in range(3):
            f = forb[i]
            if t[0] not in f and t[1] not in f:
                m |= 1 << i
        tmask.append((t, m))
        gs[m] += 1
    gst = tuple(gs)
    total = _count(gst, 0, caps)
    if total == 0:
        raise InconsistentObservation("gözlemle uyumlu dağılım yok")
    jc = {}
    for m in range(8):
        if gs[m] == 0:
            continue
        g2 = list(gs)
        g2[m] -= 1
        g2 = tuple(g2)
        row = []
        for i in range(3):
            if not (m >> i) & 1 or caps[i] == 0:
                row.append(0)
            else:
                c2 = list(caps)
                c2[i] -= 1
                row.append(_count(g2, 0, tuple(c2)))
        jc[m] = row
    terms = ([], [], [])          # beklenen el değeri terimleri; referansla aynı sum() (3.12+ Neumaier) aynı sırada
    for t, m in tmask:
        ti = TILE_INDEX[t]
        row = jc[m]
        for i in range(3):
            c = row[i]
            if c > 0:
                v[O_POSS + 28 * i + ti] = 1.0
            p = c / total
            v[O_MARG + 28 * i + ti] = p
            terms[i].append(p * tile_value(t, caps[i]))
    ev = [sum(x) for x in terms]
    # kesin kapatma (Info.certain_closing_moves ile aynı)
    if obs.current_player == me and lv >= 0:
        closing = []
        for mv in legal_moves_for(hand, lv, rv):
            if mv.is_pass:
                continue
            nl, nr = new_ends_after(lv, rv, mv)
            rest = [t for t in hand if t != mv.tile]
            if not rest:
                continue
            if (unseen_with[nl] == 0 and unseen_with[nr] == 0
                    and not any(nl in t or nr in t for t in rest)):
                closing.append(mv)
        if closing:
            v[O_CLOSE] = 1.0
            for mv in closing:
                v[O_CLOSE + 1 + move_index(mv)] = 1.0
    # ── F2
    mine_v = sum(tile_value(t, nh) for t in hand)
    for i in range(3):
        v[O_EXP + i] = ev[i] / HAND_VALUE_SCALE
    v[O_EXP + 3] = (mine_v + ev[1]) / HAND_VALUE_SCALE
    v[O_EXP + 4] = (ev[0] + ev[2]) / HAND_VALUE_SCALE
    v[O_UNC] = math.log(total) / _LOG_START
    # ── F3 (K3)
    for r in range(4):
        q = seats[r]
        for n in range(7):
            v[O_OPENED + 7 * r + n] = opened[q][n] / 7.0
            v[O_COVERED + 7 * r + n] = covered[q][n] / 7.0
            v[O_PW + 7 * r + n] = plays_with[q][n] / 7.0
        fd = first_double[q]
        v[O_FPD + 3 * r + (0 if fd is None else (2 if fd else 1))] = 1.0
        v[O_NPASS + r] = n_passes[q] / 7.0
    return v


def features_f3(obs: Observation, match: Optional[dict] = None) -> np.ndarray:
    return np.asarray(encode_f3(obs, match), dtype=np.float32)
