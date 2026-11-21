"""
Referans motor: doğruluk odaklı, okunabilir. Kurallar: Domino-Tez/03 - Kurallar.md

Katmanlar
- Round : tek tur (hamle, pas, bitirme, kapanma, tur puanı)
- Match : 101 puanlık maç (13 puan eşiği, bekleyen puan, eşitlik havuzu, tur başlatıcı)

Public/private ayrımı: ajanlar yalnız Round.observation(player) görür.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Tile = Tuple[int, int]
ALL_TILES: Tuple[Tile, ...] = tuple((i, j) for i in range(7) for j in range(i, 7))
OPENING_TILE: Tile = (1, 1)
ZERO_ZERO: Tile = (0, 0)
ZERO_ZERO_SINGLE_VALUE = 10  # R-12


class IllegalMove(Exception):
    pass


def norm(t: Tile) -> Tile:
    a, b = t
    return (a, b) if a <= b else (b, a)


def team_of(player: int) -> int:
    return player % 2  # P0+P2 = takım 0, P1+P3 = takım 1  (R-02)


def hand_value(hand) -> int:
    """El değeri. R-12: kalan tek taş 0|0 ise 10."""
    hand = list(hand)
    if len(hand) == 1 and norm(hand[0]) == ZERO_ZERO:
        return ZERO_ZERO_SINGLE_VALUE
    return sum(a + b for a, b in hand)


# ── Hamle / olay / sonuç ─────────────────────────────────────

@dataclass(frozen=True)
class Move:
    tile: Optional[Tile] = None
    side: Optional[str] = None  # 'L' | 'R'

    @property
    def is_pass(self) -> bool:
        return self.tile is None

    def __repr__(self) -> str:
        return "PAS" if self.is_pass else f"{self.side} {self.tile[0]}|{self.tile[1]}"


PASS = Move()


@dataclass(frozen=True)
class Event:
    """Public olay günlüğü kaydı. left_end/right_end: olaydan ÖNCEKİ uçlar (-1 = masa boş)."""
    player: int
    kind: str  # 'play' | 'pass'
    tile: Optional[Tile]
    side: Optional[str]
    left_end: int
    right_end: int


@dataclass(frozen=True)
class RoundResult:
    reason: str                    # 'out' | 'blocked'
    winner_team: Optional[int]     # None = kapanmada eşitlik
    points: int                    # kazanan takıma yazılacak tur puanı (eşitlikte 0)
    pot_add: int                   # eşitlikte havuza eklenecek (dört elin toplamı), aksi halde 0
    team_totals: Tuple[int, int]   # kalan el toplamları (R-12 dahil)
    last_tile_player: int          # son taşı koyan (R-13/R-14)
    finisher: Optional[int]        # elini bitiren (yalnız 'out')


@dataclass(frozen=True)
class Observation:
    """Bir oyuncunun görebildiği her şey. Rakip/partner elleri YOK."""
    player: int
    hand: Tuple[Tile, ...]
    board: Tuple[Tile, ...]
    left_end: int
    right_end: int
    hand_sizes: Tuple[int, int, int, int]
    current_player: int
    consecutive_passes: int
    events: Tuple[Event, ...]


def missing_from_events(events: Sequence[Event]) -> List[set]:
    """R-08: pas, o andaki iki uç sayısının elde olmadığını gösterir. [oyuncu] -> {sayılar}"""
    missing: List[set] = [set() for _ in range(4)]
    for e in events:
        if e.kind == "pass":
            missing[e.player].update({e.left_end, e.right_end})
    return missing


def legal_moves_for(hand, lv: int, rv: int) -> List[Move]:
    """Dolu masada yasal hamleler (R-06, R-07). Round ve bilgi katmanı bunu paylaşır."""
    moves: List[Move] = []
    for t in sorted(hand):
        if lv in t:
            moves.append(Move(t, "L"))
        if rv in t and rv != lv:                         # uçlar eşitse tekrar yok
            moves.append(Move(t, "R"))
    return moves if moves else [PASS]                    # R-07


def new_ends_after(lv: int, rv: int, move: Move) -> Tuple[int, int]:
    """Hamleden sonraki (sol, sağ) uç sayıları."""
    a, b = move.tile
    if move.side == "L":
        return (a + b - lv, rv)
    return (lv, a + b - rv)


# ── Tur ──────────────────────────────────────────────────────

class Round:
    def __init__(self, hands: Sequence[Sequence[Tile]], starter: int,
                 forced_opening: Optional[Tile] = None):
        if len(hands) != 4:
            raise ValueError("4 el gerekli")
        self._hands: List[set] = [set(norm(t) for t in h) for h in hands]
        seen: set = set()
        for h, orig in zip(self._hands, hands):
            if len(h) != len(list(orig)):
                raise ValueError("elde tekrar eden taş")
            if seen & h:
                raise ValueError("aynı taş iki elde")
            seen |= h
        if not all(t in ALL_TILES for t in seen):
            raise ValueError("geçersiz taş")
        if forced_opening is not None and norm(forced_opening) not in self._hands[starter]:
            raise ValueError("başlayan oyuncuda zorunlu açılış taşı yok")
        self.initial_hands: Tuple[Tuple[Tile, ...], ...] = tuple(tuple(sorted(h)) for h in self._hands)
        self.forced_opening = norm(forced_opening) if forced_opening else None
        self.starter = starter
        self.board: List[Tile] = []          # yönlü taşlar, soldan sağa
        self.current_player = starter
        self.consecutive_passes = 0
        self.events: List[Event] = []
        self.last_tile_player: Optional[int] = None
        self.result: Optional[RoundResult] = None

    # -- durum
    @property
    def is_over(self) -> bool:
        return self.result is not None

    @property
    def left_end(self) -> int:
        return self.board[0][0] if self.board else -1

    @property
    def right_end(self) -> int:
        return self.board[-1][1] if self.board else -1

    def hand(self, player: int) -> Tuple[Tile, ...]:
        return tuple(sorted(self._hands[player]))

    def hand_sizes(self) -> Tuple[int, int, int, int]:
        return tuple(len(h) for h in self._hands)  # type: ignore[return-value]

    # -- hamleler
    def legal_moves(self) -> List[Move]:
        if self.is_over:
            return []
        hand = sorted(self._hands[self.current_player])
        if not self.board:
            if self.forced_opening is not None:          # R-04
                return [Move(self.forced_opening, "R")]
            return [Move(t, "R") for t in hand]          # R-05: serbest
        return legal_moves_for(self._hands[self.current_player], self.left_end, self.right_end)

    def apply(self, move: Move) -> None:
        legal = self.legal_moves()
        if move not in legal:
            # Uçlar eşitken 'R' ve 'L' aynı hamlenin iki fiziksel yerleşimidir (tahta aynalanır). Ajanlara tek
            # hamle sunulur (kanonik 'L') ama kayıtlı oyunlarda sonraki hamleler fiziksel yöne bağlı olduğundan
            # 'R' de kabul edilir ve gerçekten sağa yerleştirilir.
            mirror = (bool(self.board) and not move.is_pass and move.side == "R"
                      and self.left_end == self.right_end and Move(move.tile, "L") in legal)
            if not mirror:
                raise IllegalMove(f"yasadışı hamle: {move}")
        p = self.current_player
        lv, rv = self.left_end, self.right_end
        if move.is_pass:
            self.events.append(Event(p, "pass", None, None, lv, rv))
            self.consecutive_passes += 1
        else:
            a, b = move.tile
            self._hands[p].remove(move.tile)
            if not self.board:
                self.board.append((a, b))
            elif move.side == "L":
                self.board.insert(0, (a, b) if b == lv else (b, a))
            else:
                self.board.append((a, b) if a == rv else (b, a))
            self.consecutive_passes = 0
            self.last_tile_player = p
            self.events.append(Event(p, "play", move.tile, move.side, lv, rv))

        if not move.is_pass and not self._hands[p]:
            self._finish("out", finisher=p)
        elif self.consecutive_passes >= 4:               # R-10: 4 art arda pas
            self._finish("blocked", finisher=None)
        else:
            self.current_player = (p + 1) % 4

    # -- bitiş ve puan
    def team_totals(self) -> Tuple[int, int]:
        v = [hand_value(h) for h in self._hands]
        return (v[0] + v[2], v[1] + v[3])

    def _finish(self, reason: str, finisher: Optional[int]) -> None:
        t0, t1 = self.team_totals()
        if reason == "out":
            w = team_of(finisher)
            # R-09: yalnız rakip takımın iki eli sayılır
            points = (t1 if w == 0 else t0)
            res = RoundResult("out", w, points, 0, (t0, t1), self.last_tile_player, finisher)
        else:
            if t0 < t1:
                res = RoundResult("blocked", 0, t1, 0, (t0, t1), self.last_tile_player, None)
            elif t1 < t0:
                res = RoundResult("blocked", 1, t0, 0, (t0, t1), self.last_tile_player, None)
            else:                                        # R-11: eşitlik -> havuz
                res = RoundResult("blocked", None, 0, t0 + t1, (t0, t1), self.last_tile_player, None)
        self.result = res

    # -- gözlem
    def observation(self, player: int) -> Observation:
        return Observation(
            player=player,
            hand=self.hand(player),
            board=tuple(self.board),
            left_end=self.left_end,
            right_end=self.right_end,
            hand_sizes=self.hand_sizes(),
            current_player=self.current_player,
            consecutive_passes=self.consecutive_passes,
            events=tuple(self.events),
        )

    def clone(self) -> "Round":
        """Bağımsız kopya. Değişebilir alanlar (eller, tahta, olaylar) kopyalanır; geri kalanı değişmez
        (tuple, frozen dataclass, int). Aramada çok çağrıldığı için deepcopy yerine (T-C1 deepcopy ile eşdeğerlik)."""
        r = Round.__new__(Round)
        r.__dict__.update(self.__dict__)
        r._hands = [set(h) for h in self._hands]
        r.board = list(self.board)
        r.events = list(self.events)
        return r

    def deepclone(self) -> "Round":
        return copy.deepcopy(self)


# ── Maç ──────────────────────────────────────────────────────

@dataclass
class MatchConfig:
    target: int = 101               # R-03
    threshold: int = 13             # R-16 (yalnız turun KENDİ puanına uygulanır, Q-18)


def deal_hands(match_seed, round_no: int) -> List[List[Tile]]:
    """Dağıtım, (tohum, tur no) ile deterministik. Oyun gidişatından bağımsız."""
    rng = random.Random(f"{match_seed}:{round_no}")
    tiles = list(ALL_TILES)
    rng.shuffle(tiles)
    return [sorted(tiles[i * 7:(i + 1) * 7]) for i in range(4)]


class Match:
    def __init__(self, seed=0, config: Optional[MatchConfig] = None):
        self.seed = seed
        self.cfg = config or MatchConfig()
        self.score = [0, 0]
        self.pending = [0, 0]            # R-17 bekleyen puan (eşitlik payları dahil, R-11)
        self.unlocked = [False, False]   # R-18 ilk puan yazıldı mı
        self.round_no = 0
        self.winner: Optional[int] = None
        self._use_opening_rule = True    # R-04/R-19: 1|1 sahibi 1|1 ile başlar
        self._next_starter: Optional[int] = None

    def new_round(self) -> Round:
        self.round_no += 1
        hands = deal_hands(self.seed, self.round_no)
        if self._use_opening_rule:
            starter = next(i for i, h in enumerate(hands) if OPENING_TILE in h)
            return Round(hands, starter, forced_opening=OPENING_TILE)
        return Round(hands, self._next_starter)

    def apply_result(self, res: RoundResult) -> None:
        cfg = self.cfg
        if res.winner_team is None:
            # R-11: eşitlikte dört elin toplamı İKİ takımın da bekleyenine eklenir.
            # Kendi bekleyeni olan takımda "bekleyen + eşitlik", diğerinde yalnız eşitlik.
            self.pending[0] += res.pot_add
            self.pending[1] += res.pot_add
            self._use_opening_rule = False
            self._next_starter = res.last_tile_player
            return

        w, l = res.winner_team, 1 - res.winner_team
        self.pending[l] = 0                               # R-17: kaybedenin bekleyeni yok olur

        if self.unlocked[w]:
            self.score[w] += self.pending[w] + res.points  # bekleyen yalnız eşitlik payıdır
            self.pending[w] = 0
        elif res.points >= cfg.threshold:                 # R-16: eşik yalnız turun kendi puanına
            self.score[w] += self.pending[w] + res.points
            self.pending[w] = 0
            self.unlocked[w] = True
        else:
            self.pending[w] += res.points                 # Q-17: bekler, toplanır

        if self.score[w] >= cfg.target:
            self.winner = w
        self._use_opening_rule = not any(self.unlocked)   # R-19
        self._next_starter = res.last_tile_player         # R-14

    def public_state(self) -> dict:
        """Maç durumu: tamamen public (D-11)."""
        return {
            "score": tuple(self.score),
            "pending": tuple(self.pending),
            "unlocked": tuple(self.unlocked),
            "round_no": self.round_no,
            "target": self.cfg.target,
        }
