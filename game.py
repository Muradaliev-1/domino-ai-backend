"""
game.py - Demo oyun mantığı (taşıma katmanından bağımsız, senkron, testlenebilir).

Kurallar ve motor: araştırma çekirdeği `domino` (domino-research, sabit commit: DOMINO_VERSION, D-15/D-26).
- Tam maç: 101 puan, 13 eşiği, bekleyen puan, eşitlik havuzu (domino.engine.Match, 03 - Kurallar.md)
- Koltuklar: P0 ve P2 insan (Takım A), P1 ve P3 bot (Takım B)
- Bot YALNIZ kendi Observation'ını, yasal hamle listesini ve public maç durumunu görür (B-01 kapanışı).
  Bu kısıt `bot_decide` fonksiyonunda yapısaldır: ajana Round/Match nesnesi hiç gitmez.
- Her biten tur domino.record biçiminde JSONL'e yazılır (tools/replay.py ile doğrulanır).

WebSocket protokolü (frontend değişmedi): hamle = {"tile": [a, b], "to_left": bool, "is_pass": bool}.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from typing import List, Optional, Sequence

from domino.agents import Agent, make_agent
from domino.engine import (PASS, IllegalMove, Match, Move, Observation, Round, RoundResult,
                           missing_from_events, norm, team_of)
from domino.record import from_round

HUMAN_SEATS = (0, 2)
BOT_SEATS = (1, 3)
DEFAULT_BOT = "ismcts-rand@200"
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


def domino_version() -> str:
    try:
        with open(os.path.join(BACKEND_DIR, "DOMINO_VERSION"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def bot_name() -> str:
    return os.environ.get("DOMINO_BOT", DEFAULT_BOT)


def record_path() -> str:
    return os.environ.get("DOMINO_RECORD_PATH", os.path.join(BACKEND_DIR, "records", "demo.jsonl"))


# ── Protokol dönüşümleri ─────────────────────────────────────

def move_to_dict(m: Move) -> dict:
    if m.is_pass:
        return {"tile": [0, 0], "to_left": False, "is_pass": True}
    return {"tile": list(m.tile), "to_left": m.side == "L", "is_pass": False}


def move_from_dict(d: dict, board_empty: bool) -> Move:
    if d.get("is_pass"):
        return PASS
    t = norm(tuple(int(x) for x in d["tile"]))
    if len(t) != 2:
        raise ValueError("taş iki sayı olmalı")
    if board_empty:
        return Move(t, "R")                         # açılış: taraf anlamsız (motorun kanonik biçimi)
    return Move(t, "L" if d.get("to_left") else "R")


# ── Bot kararı: tek giriş noktası, sızıntısız ────────────────

def bot_decide(agent: Agent, obs: Observation, legal: Sequence[Move], match_state: dict) -> Move:
    """Bot yalnız (Observation, yasal hamleler, public maç durumu) alır. Round/Match nesnesi buraya gelmez."""
    mv = agent.act(obs, list(legal), match_state)
    if mv not in legal:
        raise IllegalMove(f"ajan yasal olmayan hamle döndürdü: {mv}")
    return mv


class DemoGame:
    """Bir odanın maçı. Taşıma (WebSocket) main.py'de; burada yalnız durum ve kurallar."""

    def __init__(self, seed: Optional[str] = None, bot: Optional[str] = None,
                 record_to: Optional[str] = None, room_id: str = "", team_full_info: bool = False):
        self.base_seed = seed if seed is not None else secrets.token_hex(8)
        self.match_no = 1
        self.seed = f"{self.base_seed}:m1"
        self.bot = bot or bot_name()
        self.record_to = record_to if record_to is not None else record_path()
        self.room_id = room_id
        # True: iki insan koltuğunu (P0, P2) aynı kişi oynuyor, takım iki eli de biliyor (kurala aykırı güçlü rakip).
        # Kayda yazılır ki bu turlar normal insan oyunu sanılmasın (D-27)
        self.team_full_info = team_full_info
        self.match = Match(self.seed)
        self.round: Optional[Round] = None
        self.agents: dict = {}
        self.match_before: dict = {}
        self.records: List[dict] = []     # bu maçta yazılan kayıtlar (test/inceleme için)
        self.last_result: Optional[RoundResult] = None

    # -- tur
    def start_round(self) -> Round:
        if self.match.winner is not None:                # maç bitti: yeni maç
            self.match_no += 1
            self.seed = f"{self.base_seed}:m{self.match_no}"
            self.match = Match(self.seed)
        self.match_before = self.match.public_state()
        self.round = self.match.new_round()
        rn = self.match.round_no
        # Her tur her bot koltuğuna kendi tohumuyla yeni ajan (arena.py ile aynı düzen)
        self.agents = {s: make_agent(self.bot, f"{self.seed}:{rn}:s{s}") for s in BOT_SEATS}
        self.last_result = None
        return self.round

    @property
    def round_over(self) -> bool:
        return self.round is None or self.round.is_over

    def is_bot_turn(self) -> bool:
        return not self.round_over and self.round.current_player in BOT_SEATS

    def bot_step(self) -> tuple:
        """Sıradaki bot hamlesini hesaplar ve uygular. (oyuncu, hamle) döndürür."""
        r = self.round
        p = r.current_player
        assert p in BOT_SEATS and not r.is_over
        mv = bot_decide(self.agents[p], r.observation(p), r.legal_moves(), self.match.public_state())
        self._apply(mv)
        return p, mv

    def human_move(self, player: int, data: dict) -> Move:
        """İnsan hamlesi. Geçersizse ValueError / IllegalMove."""
        r = self.round
        if r is None or r.is_over or r.current_player != player:
            raise IllegalMove("Şimdi senin sıran değil!")
        mv = move_from_dict(data, board_empty=not r.board)
        self._apply(mv)                                  # Round.apply yasallığı denetler (aynalı R dahil, D-18)
        return mv

    def _apply(self, mv: Move) -> None:
        self.round.apply(mv)
        if self.round.is_over:
            self._finish_round()

    def _finish_round(self) -> None:
        r = self.round
        res = r.result
        meta = {"match_seed": self.seed, "round_no": self.match.round_no, "human_seats": list(HUMAN_SEATS),
                "bots": self.bot, "match_before": self.match_before, "domino_version": domino_version(),
                "room": self.room_id, "team_full_info": self.team_full_info, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        rec = from_round(r, "demo", meta).to_json()
        self.records.append(rec)
        if self.record_to:
            d = os.path.dirname(self.record_to)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.record_to, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.match.apply_result(res)
        self.last_result = res

    # -- mesajlar
    def state_for(self, player: int, names: dict) -> dict:
        r = self.round
        moves = []
        if r is not None and not r.is_over and r.current_player == player:
            moves = [move_to_dict(m) for m in r.legal_moves()]
        missing = missing_from_events(r.events) if r is not None else [set() for _ in range(4)]
        return {
            "type": "game_state",
            "board": [list(t) for t in r.board] if r else [],
            "your_hand": [list(t) for t in r.hand(player)] if r else [],
            "your_index": player,
            "current_player": r.current_player if r else -1,
            "is_your_turn": bool(moves),
            "legal_moves": moves,
            "left_val": r.left_end if r else -1,
            "right_val": r.right_end if r else -1,
            "hand_sizes": list(r.hand_sizes()) if r else [7, 7, 7, 7],
            "pass_count": r.consecutive_passes if r else 0,
            "is_game_over": self.round_over,
            "team_scores": list(self.match.score),       # maç skoru (public, D-11)
            "players": names,
            "missing_numbers": [[n in missing[p] for n in range(7)] for p in range(4)],
            "match": self.match.public_state(),
        }

    def round_over_message(self) -> dict:
        """Tur sonu mesajı. team_a/b_score: kurala göre sayılan el toplamları (düşük olan kazandı;
        bitirmede bitiren takımın sayılan toplamı 0, R-09). Frontend emojisi bu karşılaştırmaya bakar."""
        r, res, m = self.round, self.last_result, self.match
        if res.reason == "out":
            counted = [0, 0]
            counted[1 - res.winner_team] = res.points
        else:
            counted = list(res.team_totals)
        team = {0: "Takım A (Siz)", 1: "Takım B (Bot)"}
        if res.winner_team is None:
            line1 = f"Tur {m.round_no}: kapandı, eşitlik (+{res.pot_add} havuza)"
        else:
            how = "bitirdi" if res.reason == "out" else "kapanma"
            line1 = f"Tur {m.round_no}: {team[res.winner_team]} kazandı, {how}, {res.points} puan"
            if m.pending[res.winner_team] and not m.unlocked[res.winner_team]:
                line1 += f" (13 altı, bekliyor: {m.pending[res.winner_team]})"
        line2 = f"Maç: A {m.score[0]} - B {m.score[1]} (hedef {m.cfg.target})"
        if m.winner is not None:
            line2 = f"MAÇ BİTTİ: {team[m.winner]} kazandı! A {m.score[0]} - B {m.score[1]}"
        return {
            "type": "game_over",
            "team_a_score": counted[0],
            "team_b_score": counted[1],
            "winner": line1 + "\n" + line2,
            "hands": [[list(t) for t in r.hand(i)] for i in range(4)],
            "reason": res.reason,
            "round_winner_team": res.winner_team,
            "match": m.public_state(),
            "match_winner": m.winner,
        }

    def opening_text(self) -> str:
        r = self.round
        if r.forced_opening:
            return f"Tur {self.match.round_no} başladı: (1|1) sahibi P{r.starter + 1} açar."
        return f"Tur {self.match.round_no} başladı: P{r.starter + 1} açar (son taşı koyan)."
