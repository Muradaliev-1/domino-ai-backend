"""
Tur kaydı: gerçek (masada ya da uygulamada oynanmış) turları kaydetme, motorla yeniden oynatma ve
doğrulama (T-10). Kayıt, tek bir turun: başlangıç elleri + başlayan + sıralı olaylar (+ isteğe bağlı iddia
edilen sonuç). JSON / JSONL. Biçim: docs/Domino-Tez/15 - Veri Toplama ve Gercek Oyunlar.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

from .engine import PASS, IllegalMove, Move, Round, RoundResult, Tile, norm

VERSION = 1
RESULT_KEYS = ("reason", "winner_team", "points", "pot_add", "finisher", "last_tile_player")


@dataclass
class RoundRecord:
    hands: List[List[Tile]]
    starter: int
    forced_opening: Optional[Tile]
    events: List[dict]                       # {"player","tile","side"} | {"player","pass":True}
    result_claimed: Optional[dict] = None    # kullanıcının/uygulamanın bildirdiği sonuç (kısmi olabilir)
    source: str = "manual"
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "version": VERSION,
            "source": self.source,
            "hands": [[list(t) for t in h] for h in self.hands],
            "starter": self.starter,
            "forced_opening": list(self.forced_opening) if self.forced_opening else None,
            "events": self.events,
            "result_claimed": self.result_claimed,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, d: dict) -> "RoundRecord":
        fo = d.get("forced_opening")
        return cls(
            hands=[[norm(tuple(t)) for t in h] for h in d["hands"]],
            starter=d["starter"],
            forced_opening=norm(tuple(fo)) if fo else None,
            events=list(d["events"]),
            result_claimed=d.get("result_claimed"),
            source=d.get("source", "manual"),
            meta=d.get("meta", {}),
        )


def from_round(rnd: Round, source: str = "engine", meta: Optional[dict] = None,
               claim_result: bool = True) -> RoundRecord:
    events = []
    for e in rnd.events:
        if e.kind == "pass":
            events.append({"player": e.player, "pass": True})
        else:
            events.append({"player": e.player, "tile": list(e.tile), "side": e.side})
    claimed = None
    if claim_result and rnd.result is not None:
        claimed = {k: getattr(rnd.result, k) for k in RESULT_KEYS}
    return RoundRecord([list(h) for h in rnd.initial_hands], rnd.starter, rnd.forced_opening,
                       events, claimed, source, meta or {})


@dataclass
class ReplayReport:
    ok: bool
    errors: List[str]
    mismatches: List[str]
    round: Optional[Round]
    finished: bool
    result: Optional[RoundResult]


def _to_move(ev: dict, r: Round) -> Move:
    if ev.get("pass"):
        return PASS
    t = norm(tuple(ev["tile"]))
    if not r.board:
        return Move(t, "R")                              # açılış: taraf anlamsız
    side = ev.get("side")
    if r.left_end == r.right_end:                        # uçlar eşit: kayıtlı tarafı AYNEN koru (fiziksel yön)
        return Move(t, side if side in ("L", "R") else "L")
    if side in ("L", "R"):
        return Move(t, side)
    cands = [m for m in r.legal_moves() if m.tile == t]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise IllegalMove(f"{t[0]}|{t[1]} uçlara ({r.left_end}, {r.right_end}) oynanamaz")
    raise IllegalMove(f"{t[0]}|{t[1]} iki uca da oynanabilir, taraf (L/R) belirtilmeli")


def replay(rec: RoundRecord) -> ReplayReport:
    errors: List[str] = []
    try:
        r = Round(rec.hands, rec.starter, rec.forced_opening)
    except ValueError as ex:
        return ReplayReport(False, [f"başlangıç elleri geçersiz: {ex}"], [], None, False, None)
    for i, ev in enumerate(rec.events):
        if r.is_over:
            errors.append(f"olay {i}: tur zaten bitmişti")
            break
        if ev["player"] != r.current_player:
            errors.append(f"olay {i}: sıra P{r.current_player}'de ama kayıtta P{ev['player']}")
            break
        try:
            r.apply(_to_move(ev, r))
        except IllegalMove as ex:
            errors.append(f"olay {i} (P{ev['player']}): {ex}")
            break
    mismatches: List[str] = []
    if rec.result_claimed and not errors:
        if r.result is None:
            mismatches.append("kayıtta sonuç var ama motorda tur bitmedi")
        else:
            for k in RESULT_KEYS:
                if k in rec.result_claimed and rec.result_claimed[k] != getattr(r.result, k):
                    mismatches.append(f"{k}: kayıtta {rec.result_claimed[k]}, motor {getattr(r.result, k)}")
    return ReplayReport(not errors and not mismatches, errors, mismatches, r, r.is_over, r.result)


def save_records(path: str, records: List[RoundRecord]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")


def load_records(path: str) -> List[RoundRecord]:
    text = open(path, encoding="utf-8").read().strip()
    if not text:
        return []
    if path.endswith(".jsonl") or "\n{" in text and not text.startswith("["):
        return [RoundRecord.from_json(json.loads(l)) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    return [RoundRecord.from_json(d) for d in (data if isinstance(data, list) else [data])]
