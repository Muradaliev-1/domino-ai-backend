"""
main.py - Domino AI Backend
===========================
FastAPI + WebSocket ile gerçek zamanlı domino oyunu. 2 insan (P1, P3 = Takım A) vs 2 bot (P2, P4 = Takım B).

Motor, kurallar ve bot: araştırma çekirdeği `domino` (domino-research, sabit commit DOMINO_VERSION).
Oyun mantığı game.py'de (testlenebilir); burada yalnız oda yönetimi ve WebSocket taşıması var.
Bot gizli elleri görmez: yalnız kendi gözlemi + yasal hamleler + public maç durumu (game.bot_decide).

Çalıştırma:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

Ortam değişkenleri (isteğe bağlı): DOMINO_BOT (varsayılan: game.DEFAULT_BOT), DOMINO_RECORD_PATH
(varsayılan records/demo.jsonl), DOMINO_BOT_DELAY (bot hamlesinden sonra bekleme, sn, varsayılan 0.8).
"""

import asyncio
import json
import os
import uuid
from typing import Dict, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse

from domino.engine import IllegalMove
from game import DemoGame, HUMAN_SEATS, bot_name, domino_version, move_to_dict, record_path

BOT_DELAY = float(os.environ.get("DOMINO_BOT_DELAY", "0.8"))
START_DELAY = 0.6   # tur başında ilk bot hamlesinden önce bekleme (sn), bkz. start_game


def same_player(names: Dict[int, str]) -> bool:
    """İki sekmeden aynı ad (büyük/küçük harf ve baştaki/sondaki boşluk önemsiz) → aynı kişi, takım iki eli biliyor.
    Adların kendisi kayda yazılmaz, yalnız bu bayrak."""
    ns = [str(names.get(s, "")).strip().casefold() for s in (0, 1)]
    return bool(ns[0]) and ns[0] == ns[1]


# ═══════════════════════════════════════════════════════════
# ODA YÖNETİCİSİ
# ═══════════════════════════════════════════════════════════

class GameRoom:
    def __init__(self, room_id: str, seed: Optional[str] = None):
        self.room_id = room_id
        self.seed = seed
        self.game: Optional[DemoGame] = None

        # İnsan oyuncular: slot 0 = P1 (indeks 0), slot 1 = P3 (indeks 2). Bot: P2 (1), P4 (3)
        self.human_slots: Dict[int, WebSocket] = {}
        self.human_names: Dict[int, str] = {}

        self.game_started = False
        self.lock = asyncio.Lock()

    def is_full(self) -> bool:
        return len(self.human_slots) == 2

    def player_index(self, slot: int) -> int:
        """Slot 0 → oyuncu indeks 0, slot 1 → oyuncu indeks 2"""
        return HUMAN_SEATS[slot]

    async def broadcast(self, message: dict):
        data = json.dumps(message, ensure_ascii=False)
        for ws in list(self.human_slots.values()):
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
        """Bir oyuncuya özel oyun durumu — yalnız kendi eli görünür"""
        return self.game.state_for(self.player_index(for_slot), self.human_names)

    async def send_states(self):
        for slot in list(self.human_slots):
            await self.send_to(slot, self.game_state_message(slot))

    async def start_game(self):
        """Yeni tur (maç bittiyse yeni maç). Tur ortasında çağrılmaz."""
        if self.game is None:
            self.game = DemoGame(seed=self.seed, room_id=self.room_id)
        self.game.team_full_info = same_player(self.human_names)
        self.game.start_round()
        self.game_started = True

        await self.broadcast({
            "type":    "game_started",
            "message": self.game.opening_text(),
            "players": self.human_names,
        })
        await self.send_states()
        # Frontend ilk durumu 100 ms gecikmeyle işliyor (GameScene firstState); bot ondan önce oynarsa eski durum
        # yenisinin üstüne çizilir. Eski MCTS ~0,5 sn düşündüğü için görünmüyordu, yeni bot daha hızlı.
        await asyncio.sleep(START_DELAY)
        await self.maybe_agent_move()

    async def maybe_agent_move(self):
        """Sıra bottaysa hamle yaptır"""
        g = self.game
        moved = False
        while g.is_bot_turn():
            moved = True
            player = g.round.current_player
            await self.broadcast({
                "type":    "agent_thinking",
                "player":  player,
                "message": f"Bot (Oyuncu {player + 1}) düşünüyor..."
            })
            # Hesap thread'de (event loop bloklanmasın). Bot yalnız kendi gözlemini görür (game.bot_decide).
            loop = asyncio.get_running_loop()
            actor, move = await loop.run_in_executor(None, g.bot_step)

            await self.broadcast({
                "type":   "agent_move",
                "player": actor,                       # hamleyi yapan (B-09 düzeltildi)
                "move":   move_to_dict(move),
                "board":  [list(t) for t in g.round.board],
            })
            if g.round_over:
                break
            await asyncio.sleep(BOT_DELAY)

        if moved:                                   # bot oynamadıysa durum zaten gönderildi (çift mesaj yok)
            await self.send_states()
        if g.round_over:
            await self.end_game()

    async def handle_human_move(self, slot: int, move_data: dict):
        g = self.game
        p_index = self.player_index(slot)
        if g is None or g.round_over or g.round.current_player != p_index:
            await self.send_to(slot, {"type": "error", "message": "Şimdi senin sıran değil!"})
            return
        try:
            move = g.human_move(p_index, move_data)
        except IllegalMove:
            await self.send_to(slot, {"type": "error", "message": "Yasadışı hamle!"})
            return
        except Exception as ex:
            await self.send_to(slot, {"type": "error", "message": f"Geçersiz hamle: {ex}"})
            return

        await self.broadcast({
            "type":   "human_move",
            "player": p_index,
            "name":   self.human_names.get(slot, f"Oyuncu {p_index + 1}"),
            "move":   move_to_dict(move),
            "board":  [list(t) for t in g.round.board],
        })
        await self.send_states()
        if g.round_over:
            await self.end_game()
        else:
            await asyncio.sleep(0.3)               # frontend animasyonu
            await self.maybe_agent_move()

    async def handle_rematch(self, slot: int):
        """'Tekrar Oyna': yalnız tur bittiyse yeni tur (B-10 kısmen). Tur sürüyorsa (diğer insan zaten
        başlattı) durumu bu oyuncuya yeniden gönder: frontend butona basınca masayı temizliyor."""
        if self.game is not None and not self.game.round_over:
            await self.send_to(slot, self.game_state_message(slot))
            return
        await self.start_game()

    async def end_game(self):
        await self.broadcast(self.game.round_over_message())


# ═══════════════════════════════════════════════════════════
# FASTAPI UYGULAMASI
# ═══════════════════════════════════════════════════════════

app = FastAPI(title="Domino AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Aktif odalar
rooms: Dict[str, GameRoom] = {}
# Bekleyen oda (dolmamış)
waiting_room: Optional[str] = None
# Testler için: yeni odaların maç tohumu (None = rastgele)
ROOM_SEED: Optional[str] = None


@app.get("/")
async def root():
    return {"status": "Domino AI Backend", "bot": bot_name(), "domino_version": domino_version()}


@app.get("/rooms")
async def list_rooms():
    return {
        "active_rooms": len(rooms),
        "waiting":      waiting_room is not None
    }


@app.get("/records")
async def get_records():
    """Biten turların kaydı (JSONL, domino.record biçimi, oyuncu adı yok). Render diski kalıcı değil: hemen indir."""
    p = record_path()
    text = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
    return PlainTextResponse(text, media_type="application/x-ndjson")


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
        room         = GameRoom(room_id, seed=ROOM_SEED)
        rooms[room_id] = room
        slot         = 0
        waiting_room = room_id

    room.human_slots[slot] = ws
    room.human_names[slot] = player_name

    await ws.send_text(json.dumps({
        "type":    "joined",
        "room_id": room_id,
        "slot":    slot,
        "message": f"Odaya katıldın! Sen Oyuncu {room.player_index(slot) + 1} (Takım A) olacaksın.",
        "waiting": not room.is_full(),
    }, ensure_ascii=False))

    if room.is_full():
        await room.broadcast({
            "type":    "room_full",
            "message": f"Oda doldu! {room.human_names[0]} ve {room.human_names[1]} hazır. Oyun başlıyor...",
        })
        await asyncio.sleep(1)
        async with room.lock:
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
                    await room.handle_rematch(slot)

            elif msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        # Oyuncu ayrıldı
        room.human_slots.pop(slot, None)
        await room.broadcast({
            "type":    "player_left",
            "message": f"{player_name} oyundan ayrıldı.",
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
