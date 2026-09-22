"""
main.py - Domino AI Backend
===========================
FastAPI + WebSocket ile gerçek zamanlı domino oyunu.

Kurulum:
    pip install fastapi uvicorn websockets torch numpy

Çalıştırma:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Oyun akışı:
    1. İki insan /join endpoint'ine bağlanır
    2. Oda doluşunca oyun başlar
    3. İnsan hamleleri WebSocket üzerinden gelir
    4. Agent hamleleri backend hesaplar, tüm oyunculara gönderir
"""

import asyncio
import json
import uuid
import threading
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from domino_engine import (
    DominoEngine, Move, create_game, get_agent_move, ALL_TILES
)

# ── NN Model (opsiyonel — yoksa sadece MCTS kullanır) ────────
MODEL_PATH = Path("models/best_model.pt")
nn_model   = None

try:
    import torch
    import torch.nn as nn

    class DominoNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(150, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            )
            self.policy_head = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,28))
            self.value_head  = nn.Sequential(nn.Linear(128,32), nn.ReLU(), nn.Linear(32,1), nn.Tanh())

        def forward(self, x):
            t = self.trunk(x)
            return self.policy_head(t), self.value_head(t)

    if MODEL_PATH.exists():
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        nn_model   = DominoNet().to(device)
        nn_model.load_state_dict(checkpoint["model_state_dict"])
        nn_model.eval()
        print(f"[NN] Model yuklendi — epoch {checkpoint['epoch']}, device={device}")
    else:
        print("[NN] Model bulunamadi, sadece MCTS kullanilacak")
except ImportError:
    print("[NN] PyTorch bulunamadi, sadece MCTS kullanilacak")


# ═══════════════════════════════════════════════════════════
# ODA YÖNETİCİSİ
# ═══════════════════════════════════════════════════════════


class GameRoom:
    def __init__(self, room_id: str):
        self.room_id     = room_id
        self.engine:     Optional[DominoEngine] = None

        # İnsan oyuncular: slot 0 = P1 (indeks 0), slot 1 = P3 (indeks 2)
        # Agent: P2 (indeks 1), P4 (indeks 3)
        self.human_slots: Dict[int, WebSocket] = {}  # {0: ws, 1: ws}
        self.human_names: Dict[int, str]       = {}  # {0: "Ali", 1: "Veli"}

        self.game_started = False
        self.lock = asyncio.Lock()

    def is_full(self) -> bool:
        return len(self.human_slots) == 2

    def player_index(self, slot: int) -> int:
        """Slot 0 → oyuncu indeks 0, slot 1 → oyuncu indeks 2"""
        return 0 if slot == 0 else 2

    async def broadcast(self, message: dict):
        """Odadaki tüm insanlara mesaj gönder"""
        data = json.dumps(message, ensure_ascii=False)
        for ws in self.human_slots.values():
            try:
                await ws.send_text(data)
            except Exception:
                pass

    async def send_to(self, slot: int, message: dict):
        ws = self.human_slots.get(slot)
        if ws:
            try:
                await ws.send_text(json.dumps(message, ensure_ascii=False))
            except Exception:
                pass

    def game_state_message(self, for_slot: int) -> dict:
        """Bir oyuncuya özel oyun durumu — sadece kendi eli görünür"""
        e       = self.engine
        p_index = self.player_index(for_slot)

        moves = []
        if not e.is_game_over and e.current_player == p_index:
            moves = [m.to_dict() for m in e.get_legal_moves()]

        return {
            "type":           "game_state",
            "board":          [list(t) for t in e.board],
            "your_hand":      [list(t) for t in e.players[p_index]],
            "your_index":     p_index,
            "current_player": e.current_player,
            "is_your_turn":   e.current_player == p_index and not e.is_game_over,
            "legal_moves":    moves,
            "left_val":       e.left_val,
            "right_val":      e.right_val,
            "hand_sizes":     [len(h) for h in e.players],
            "pass_count":     e.pass_count,
            "is_game_over":   e.is_game_over,
            "team_scores":    [e.team_score(0), e.team_score(1)],
            "players":        self.human_names,
            "missing_numbers": e.missing,
        }

    async def start_game(self):
        self.engine       = create_game()
        self.game_started = True

        await self.broadcast({
            "type":    "game_started",
            "message": "Oyun basladi! (1|1) ile acildi.",
            "players": self.human_names,
        })

        # Güncel durumu gönder
        for slot in self.human_slots:
            await self.send_to(slot, self.game_state_message(slot))

        # Eğer agent sırası geldiyse hemen oynasın
        await self.maybe_agent_move()

    async def maybe_agent_move(self):
        """Sıra agent'taysa hamle yaptır"""
        e = self.engine
        while not e.is_game_over and e.current_player in (1, 3):
            await self.broadcast({
                "type":    "agent_thinking",
                "player":  e.current_player,
                "message": f"Agent (Oyuncu {e.current_player + 1}) dusunuyor..."
            })

            # Agent hamlesi thread'de hesapla (async'i bloklamasın)
            loop = asyncio.get_event_loop()
            move = await loop.run_in_executor(
                None,
                lambda: get_agent_move(e.clone(), iterations=150)
            )

            e.apply_move(move)

            await self.broadcast({
                "type":   "agent_move",
                "player": (e.current_player - 1) % 4,  # hamleyi yapan
                "move":   move.to_dict(),
                "board":  [list(t) for t in e.board],
            })

            # Kısa bekleme (UI'da görünsün)
            await asyncio.sleep(0.8)

            # Oyun bittiyse çık
            if e.is_game_over:
                break

        # Tüm oyunculara güncel durum
        for slot in self.human_slots:
            await self.send_to(slot, self.game_state_message(slot))

        if e.is_game_over:
            await self.end_game()

    async def handle_human_move(self, slot: int, move_data: dict):
        e       = self.engine
        p_index = self.player_index(slot)

        if e.is_game_over or e.current_player != p_index:
            await self.send_to(slot, {"type": "error", "message": "Simdi senin siran degil!"})
            return

        # Hamleyi parse et
        try:
            if move_data.get("is_pass"):
                move = Move(is_pass=True)
            else:
                tile    = tuple(move_data["tile"])
                to_left = bool(move_data["to_left"])
                move    = Move(tile=tile, to_left=to_left)
        except Exception as ex:
            await self.send_to(slot, {"type": "error", "message": f"Gecersiz hamle: {ex}"})
            return

        # Yasal mı kontrol et
        legal = e.get_legal_moves()
        valid = any(
            (m.is_pass == move.is_pass and
             (move.is_pass or (m.tile == move.tile and m.to_left == move.to_left)))
            for m in legal
        )
        if not valid:
            await self.send_to(slot, {"type": "error", "message": "Yasadisi hamle!"})
            return

        e.apply_move(move)

        await self.broadcast({
            "type":   "human_move",
            "player": p_index,
            "name":   self.human_names.get(slot, f"Oyuncu {p_index+1}"),
            "move":   move.to_dict(),
            "board":  [list(t) for t in e.board],
        })

        if e.is_game_over:
            for s in self.human_slots:
                await self.send_to(s, self.game_state_message(s))
            await self.end_game()
        else:
            await self.maybe_agent_move()

    async def end_game(self):
        e  = self.engine
        a  = e.team_score(0)
        b  = e.team_score(1)

        if a < b:
            winner = "Takim A (Siz) kazandi!"
        elif b < a:
            winner = "Takim B (Agent) kazandi!"
        else:
            winner = "Berabere!"

        await self.broadcast({
            "type":        "game_over",
            "team_a_score": a,
            "team_b_score": b,
            "winner":      winner,
            "hands":       [[list(t) for t in e.players[i]] for i in range(4)],
        })


# ═══════════════════════════════════════════════════════════
# FASTAPI UYGULAMASI
# ═══════════════════════════════════════════════════════════

app = FastAPI(title="Domino AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Aktif odalar
rooms: Dict[str, GameRoom] = {}
# Bekleyen oda (dolmamış)
waiting_room: Optional[str] = None


@app.get("/")
async def root():
    return {"status": "Domino AI Backend", "nn_loaded": nn_model is not None}


@app.get("/rooms")
async def list_rooms():
    return {
        "active_rooms": len(rooms),
        "waiting":      waiting_room is not None
    }


@app.websocket("/ws/{player_name}")
async def websocket_endpoint(ws: WebSocket, player_name: str):
    global waiting_room

    await ws.accept()

    # Oda bul veya oluştur
    if waiting_room and waiting_room in rooms:
        room_id = waiting_room
        room    = rooms[room_id]
        slot    = 1
        waiting_room = None
    else:
        room_id      = str(uuid.uuid4())[:8]
        room         = GameRoom(room_id)
        rooms[room_id] = room
        slot         = 0
        waiting_room = room_id

    room.human_slots[slot] = ws
    room.human_names[slot] = player_name

    await ws.send_text(json.dumps({
        "type":    "joined",
        "room_id": room_id,
        "slot":    slot,
        "message": f"Odaya katildin! Sen Oyuncu {room.player_index(slot) + 1} (Takim A) olacaksin.",
        "waiting": not room.is_full(),
    }))

    if room.is_full():
        await room.broadcast({
            "type":    "room_full",
            "message": f"Oda doldu! {room.human_names[0]} ve {room.human_names[1]} hazir. Oyun basliyor...",
        })
        await asyncio.sleep(1)
        await room.start_game()

    # Mesaj döngüsü
    try:
        while True:
            data = await ws.receive_text()
            msg  = json.loads(data)

            if msg.get("type") == "move":
                async with room.lock:
                    await room.handle_human_move(slot, msg)

            elif msg.get("type") == "rematch":
                async with room.lock:
                    await room.start_game()

            elif msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        # Oyuncu ayrıldı
        room.human_slots.pop(slot, None)
        await room.broadcast({
            "type":    "player_left",
            "message": f"{player_name} oyundan ayrildi.",
        })
        if not room.human_slots:
            rooms.pop(room_id, None)
            if waiting_room == room_id:
                waiting_room = None

    except Exception as ex:
        print(f"[WS] Hata: {ex}")
        room.human_slots.pop(slot, None)


# Frontend static dosyaları (build sonrası)
frontend_path = Path("frontend/dist")
if frontend_path.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_path / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        return FileResponse(str(frontend_path / "index.html"))
