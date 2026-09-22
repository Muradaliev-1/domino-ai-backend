"""
domino_engine.py
================
C# DominoEngine'in tam Python portu.
MCTS + NN agent dahil.
"""

import random
import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════
# 1. VERİ TİPLERİ
# ═══════════════════════════════════════════════════════════

Tile = Tuple[int, int]  # (a, b) domino taşı

@dataclass
class Move:
    tile: Tile = (0, 0)
    to_left: bool = False
    is_pass: bool = False

    def __repr__(self):
        if self.is_pass:
            return "PAS"
        side = "SOL" if self.to_left else "SAG"
        return f"{side} ({self.tile[0]}|{self.tile[1]})"

    def to_dict(self):
        return {
            "tile": list(self.tile),
            "to_left": self.to_left,
            "is_pass": self.is_pass
        }


# ═══════════════════════════════════════════════════════════
# 2. OYUNCU PROFİLİ
# ═══════════════════════════════════════════════════════════

class PlayerProfile:
    def __init__(self):
        self.missing_numbers = [False] * 7
        self.number_strength = [0] * 7
        self.weak_numbers    = [False] * 7

    def observe_tile(self, tile: Tile, left_val: int, right_val: int):
        self.number_strength[tile[0]] += 1
        self.number_strength[tile[1]] += 1

        if left_val >= 0 and right_val >= 0:
            can_left  = tile[1] == left_val  or tile[0] == left_val
            can_right = tile[0] == right_val or tile[1] == right_val
            if can_left  and not can_right and 0 <= right_val <= 6:
                self.weak_numbers[right_val] = True
            if can_right and not can_left  and 0 <= left_val  <= 6:
                self.weak_numbers[left_val]  = True

    def observe_pass(self, left_val: int, right_val: int):
        if 0 <= left_val  <= 6: self.missing_numbers[left_val]  = True
        if 0 <= right_val <= 6: self.missing_numbers[right_val] = True

    def to_features(self) -> List[float]:
        f = []
        for i in range(7): f.append(1.0 if self.missing_numbers[i] else 0.0)
        for i in range(7): f.append(min(self.number_strength[i] / 3.0, 1.0))
        for i in range(7): f.append(1.0 if self.weak_numbers[i]    else 0.0)
        return f

    def clone(self):
        p = PlayerProfile()
        p.missing_numbers = self.missing_numbers[:]
        p.number_strength = self.number_strength[:]
        p.weak_numbers    = self.weak_numbers[:]
        return p


# ═══════════════════════════════════════════════════════════
# 3. OYUN MOTORU
# ═══════════════════════════════════════════════════════════

ALL_TILES: List[Tile] = [(i, j) for i in range(7) for j in range(i, 7)]

class DominoEngine:
    def __init__(self):
        self.players: List[List[Tile]] = [[] for _ in range(4)]
        self.board:   List[Tile]       = []
        self.current_player: int       = 0
        self.pass_count:     int       = 0
        self.is_game_over:   bool      = False

        # Pas hafızası [oyuncu][sayı]
        self.missing: List[List[bool]] = [[False]*7 for _ in range(4)]

        # Gözlem matrisi [gözlemci][hedef]
        self.profiles: List[List[Optional[PlayerProfile]]] = [
            [PlayerProfile() if obs != tgt else None for tgt in range(4)]
            for obs in range(4)
        ]

    def clone(self):
        e = DominoEngine()
        e.players       = [h[:] for h in self.players]
        e.board         = self.board[:]
        e.current_player = self.current_player
        e.pass_count    = self.pass_count
        e.is_game_over  = self.is_game_over
        e.missing       = [r[:] for r in self.missing]
        e.profiles      = [
            [self.profiles[o][t].clone() if self.profiles[o][t] else None
             for t in range(4)]
            for o in range(4)
        ]
        return e

    @property
    def left_val(self) -> int:
        return self.board[0][0] if self.board else -1

    @property
    def right_val(self) -> int:
        return self.board[-1][1] if self.board else -1

    def get_legal_moves(self) -> List[Move]:
        moves = []
        if not self.board:
            return moves

        lv = self.left_val
        rv = self.right_val
        hand = self.players[self.current_player]

        for tile in hand:
            a, b = tile
            if a == lv or b == lv:
                moves.append(Move(tile=tile, to_left=True))
            if a == rv or b == rv:
                moves.append(Move(tile=tile, to_left=False))

        if not moves:
            moves.append(Move(is_pass=True))
        return moves

    def apply_move(self, move: Move):
        actor = self.current_player
        lv    = self.left_val
        rv    = self.right_val

        if move.is_pass:
            self.pass_count += 1
            if self.board:
                if 0 <= lv <= 6: self.missing[actor][lv] = True
                if 0 <= rv <= 6: self.missing[actor][rv] = True
                for obs in range(4):
                    if obs != actor and self.profiles[obs][actor]:
                        self.profiles[obs][actor].observe_pass(lv, rv)
        else:
            self.pass_count = 0
            self.players[actor].remove(move.tile)

            for obs in range(4):
                if obs != actor and self.profiles[obs][actor]:
                    self.profiles[obs][actor].observe_tile(move.tile, lv, rv)

            a, b = move.tile
            if not self.board:
                self.board.append((a, b))
            elif move.to_left:
                fmt = (a, b) if b == self.board[0][0] else (b, a)
                self.board.insert(0, fmt)
            else:
                fmt = (a, b) if a == self.board[-1][1] else (b, a)
                self.board.append(fmt)

        if len(self.players[actor]) == 0 or self.pass_count >= 4:
            self.is_game_over = True
        else:
            self.current_player = (self.current_player + 1) % 4

    def score(self, player_idx: int) -> int:
        return sum(a + b for a, b in self.players[player_idx])

    def team_score(self, team: int) -> int:
        """team 0 = P0+P2, team 1 = P1+P3"""
        if team == 0:
            return self.score(0) + self.score(2)
        return self.score(1) + self.score(3)

    def encode_state(self) -> List[float]:
        me       = self.current_player
        teammate = (me + 2) % 4
        opp1     = (me + 1) % 4
        opp2     = (me + 3) % 4

        f = []

        # 1. Masadaki taşlar (28)
        for tile in ALL_TILES:
            f.append(1.0 if any(
                (b[0]==tile[0] and b[1]==tile[1]) or
                (b[0]==tile[1] and b[1]==tile[0])
                for b in self.board
            ) else 0.0)

        # 2. Kendi el (28)
        hand = self.players[me]
        for tile in ALL_TILES:
            f.append(1.0 if any(
                (h[0]==tile[0] and h[1]==tile[1]) or
                (h[0]==tile[1] and h[1]==tile[0])
                for h in hand
            ) else 0.0)

        # 3. Pas hafızası (28 = 4×7)
        for p in range(4):
            for n in range(7):
                f.append(1.0 if self.missing[p][n] else 0.0)

        # 4. Profiller (21×3 = 63)
        for idx in [teammate, opp1, opp2]:
            prof = self.profiles[me][idx]
            f.extend(prof.to_features() if prof else [0.0]*21)

        # 5. Uçlar (2)
        f.append(self.left_val  / 6.0 if self.board else -1.0)
        f.append(self.right_val / 6.0 if self.board else -1.0)

        # 6. Skor farkı (1)
        my_team  = 0 if me in (0, 2) else 1
        my_sc    = self.team_score(my_team)
        opp_sc   = self.team_score(1 - my_team)
        f.append((opp_sc - my_sc) / 84.0)

        return f  # 150 özellik

    def to_dict(self) -> dict:
        """Frontend için JSON serileştirme"""
        return {
            "board":          [list(t) for t in self.board],
            "current_player": self.current_player,
            "pass_count":     self.pass_count,
            "is_game_over":   self.is_game_over,
            "left_val":       self.left_val,
            "right_val":      self.right_val,
            "hand_sizes":     [len(h) for h in self.players],
            "scores":         [self.score(i) for i in range(4)],
            "team_scores":    [self.team_score(0), self.team_score(1)],
        }


# ═══════════════════════════════════════════════════════════
# 4. OYUN BAŞLATICI
# ═══════════════════════════════════════════════════════════

def create_game() -> DominoEngine:
    """Yeni oyun oluştur, (1,1) sahibini başlatıcı yap"""
    engine  = DominoEngine()
    tiles   = ALL_TILES[:]
    random.shuffle(tiles)

    for i in range(4):
        engine.players[i] = list(tiles[i*7:(i+1)*7])
        if (1, 1) in engine.players[i]:
            engine.current_player = i

    # İlk hamle: (1,1) oyna
    engine.apply_move(Move(tile=(1, 1), to_left=False))
    return engine


# ═══════════════════════════════════════════════════════════
# 5. MCTS AGENT
# ═══════════════════════════════════════════════════════════

class MCTSNode:
    def __init__(self, state: DominoEngine, parent=None, action: Move=None):
        self.state         = state.clone()
        self.parent        = parent
        self.action        = action
        self.children      = []
        self.total_reward  = 0.0
        self.visits        = 0
        self.untried_moves = state.get_legal_moves()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def select_best_child(self, c=1.41):
        return max(self.children,
                   key=lambda ch: (ch.total_reward / ch.visits) +
                                  c * math.sqrt(math.log(self.visits) / ch.visits))

    def expand(self):
        move      = self.untried_moves.pop(0)
        next_state = self.state.clone()
        next_state.apply_move(move)
        child = MCTSNode(next_state, self, move)
        self.children.append(child)
        return child

    def update(self, result: float):
        self.visits       += 1
        self.total_reward += result
        if self.parent:
            self.parent.update(result)


def _create_scenario(real: DominoEngine) -> DominoEngine:
    scenario = real.clone()
    me       = real.current_player

    known = set()
    for t in real.board:
        known.add((min(t), max(t)))
    for t in real.players[me]:
        known.add((min(t), max(t)))

    unknown = [t for t in ALL_TILES if (min(t), max(t)) not in known]
    random.shuffle(unknown)

    for i in range(4):
        if i == me:
            continue
        valid  = [t for t in unknown
                  if not scenario.missing[i][t[0]] and not scenario.missing[i][t[1]]]
        needed = 7 - len(scenario.players[i])
        if len(valid) >= needed:
            assigned = valid[:needed]
        else:
            rest     = [t for t in unknown if t not in valid]
            assigned = valid + rest[:needed - len(valid)]

        scenario.players[i] = assigned[:]
        for t in assigned:
            if t in unknown:
                unknown.remove(t)

    return scenario


def _rollout(state: DominoEngine, my_team: int) -> float:
    game = state.clone()
    me   = game.current_player

    while not game.is_game_over:
        moves = game.get_legal_moves()
        move  = _choose_rollout_move(game, moves)
        game.apply_move(move)

    a = game.team_score(0)
    b = game.team_score(1)

    if my_team == 0:
        if a < b: return 1.0
        if a == b: return 0.5
        return 0.0
    else:
        if b < a: return 1.0
        if b == a: return 0.5
        return 0.0


def _choose_rollout_move(game: DominoEngine, moves: List[Move]) -> Move:
    if len(moves) == 1:
        return moves[0]

    me       = game.current_player
    teammate = (me + 2) % 4
    opp1     = (me + 1) % 4
    opp2     = (me + 3) % 4

    tm_prof  = game.profiles[me][teammate]
    op1_prof = game.profiles[me][opp1]
    op2_prof = game.profiles[me][opp2]

    lv = game.left_val
    rv = game.right_val

    def score_move(m: Move) -> float:
        if m.is_pass:
            return 0.0
        s = 0.0
        a, b = m.tile
        new_left  = a if m.to_left  else lv
        new_right = b if not m.to_left else rv

        if tm_prof:
            if tm_prof.number_strength[a] > 1: s += 0.35
            if tm_prof.number_strength[b] > 1: s += 0.35
            if tm_prof.missing_numbers[a]:      s -= 0.60
            if tm_prof.missing_numbers[b]:      s -= 0.60
            if tm_prof.weak_numbers[a]:         s -= 0.25
            if tm_prof.weak_numbers[b]:         s -= 0.25

        for opp in [op1_prof, op2_prof]:
            if not opp: continue
            if 0 <= new_left  <= 6 and opp.missing_numbers[new_left]:  s += 0.55
            if 0 <= new_right <= 6 and opp.missing_numbers[new_right]: s += 0.55
            if 0 <= new_left  <= 6 and opp.weak_numbers[new_left]:     s += 0.20
            if 0 <= new_right <= 6 and opp.weak_numbers[new_right]:    s += 0.20

        return s

    scored = sorted(moves, key=score_move, reverse=True)
    return scored[0] if random.random() < 0.7 else random.choice(moves)


def get_agent_move(game: DominoEngine, iterations: int = 500) -> Move:
    """MCTS ile en iyi hamleyi bul"""
    root     = MCTSNode(game)
    my_team  = game.current_player % 2

    for _ in range(iterations):
        scenario = _create_scenario(game)
        node     = root

        while node.is_fully_expanded() and node.children:
            node = node.select_best_child()

        if not node.state.is_game_over and not node.is_fully_expanded():
            node = node.expand()

        result = _rollout(node.state, my_team)
        node.update(result)

    if not root.children:
        moves = game.get_legal_moves()
        return moves[0] if moves else Move(is_pass=True)

    best = max(root.children, key=lambda c: c.visits)
    return best.action
