"""Demo backend testleri (D-26). Çalıştırma (backend klasöründen):  python tests/run.py

T-DM1  Demo üzerinden oynanan maçlar referans motorla birebir: her tur kaydı domino.record.replay ile TAMAM,
       dağıtım deal_hands(tohum, tur) ile aynı, maç skoru kayıtlardan bağımsız Match ile yeniden hesaplanınca aynı.
T-DM2  Sızıntı (T-L1 tarzı): bot kararı gizli ellerden bağımsız. Aynı gözlem, karıştırılmış gizli eller,
       aynı tohum → aynı hamle; ajana giden argümanlar yalnız Observation / Move listesi / public maç durumu.
T-DM3  Protokol: move_to_dict / move_from_dict gidiş-dönüş, aynalı 'R' (D-18), açılış, pas.
T-DM4  İnsana giden durum mesajında başka elin taşı yok (yalnız your_hand), tur sonunda eller açılır.
T-DM5  WebSocket uçtan uca: gerçek uvicorn sunucusu + iki websockets istemcisi, birkaç tur, kayıtlar replay ile TAMAM,
       ekrana gelen maç skoru referansla aynı, 'Tekrar Oyna' tur ortasında yeni tur başlatmaz.
T-DM7  Aynı ad iki sekmeden → kayıt meta'sında team_full_info True (farklı ad → False); adın kendisi kayda yazılmaz.
T-DM6  Vendored `domino` paketi domino-research'teki kaynakla birebir (kaynak klasör yoksa atlanır).
"""
import asyncio
import copy
import json
import os
import random
import socket
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from domino.engine import (PASS, IllegalMove, Match, Move, Observation, Round, deal_hands,  # noqa: E402
                           ALL_TILES)
from domino.record import RoundRecord, load_records, replay  # noqa: E402
from domino.agents import make_agent  # noqa: E402
import game  # noqa: E402
from game import BOT_SEATS, HUMAN_SEATS, DemoGame, bot_decide, move_from_dict, move_to_dict  # noqa: E402

RESEARCH_SRC = os.environ.get("DOMINO_RESEARCH", r"C:\Users\murad\Projects\domino-research")


def human_choice(g: DemoGame, rng: random.Random) -> dict:
    """Frontend'in göndereceği gibi bir hamle sözlüğü: yasal listeden rastgele, uçlar eşitse bazen aynalı 'R'."""
    r = g.round
    legal = r.legal_moves()
    d = move_to_dict(rng.choice(legal))
    if not d["is_pass"] and r.board and r.left_end == r.right_end and rng.random() < 0.5:
        d["to_left"] = False                       # aynalı yerleşim (D-18): motor kabul etmeli
    return d


def play_full_match(seed, bot, rng, record_to="", max_rounds=60):
    g = DemoGame(seed=seed, bot=bot, record_to=record_to)
    while True:
        g.start_round()
        while not g.round_over:
            p = g.round.current_player
            if p in BOT_SEATS:
                g.bot_step()
            else:
                g.human_move(p, human_choice(g, rng))
        if g.match.winner is not None or g.match.round_no >= max_rounds:
            return g


def reference_match(seed, records):
    """Kayıtları referans motorda yeniden oynat, sonuçları bağımsız bir Match'e uygula."""
    m = Match(seed)
    for rec_json in records:
        rec = RoundRecord.from_json(rec_json)
        r_ref = m.new_round()
        assert [list(h) for h in r_ref.initial_hands] == [sorted(h) for h in rec.hands], "dağıtım farklı"
        assert [sorted(h) for h in rec.hands] == deal_hands(seed, m.round_no)
        assert r_ref.starter == rec.starter and r_ref.forced_opening == rec.forced_opening
        rep = replay(rec)
        assert rep.ok, (rep.errors, rep.mismatches)
        m.apply_result(rep.result)
    return m


# ── T-DM1 ────────────────────────────────────────────────────

def test_DM1_demo_matches_equal_reference_engine():
    rng = random.Random(1)
    n_rounds = 0
    for i in range(12):
        bot = "random" if i < 10 else "ismcts@10"
        g = play_full_match(f"test{i}", bot, rng)
        assert g.match.winner is not None
        ref = reference_match(g.seed, g.records)
        assert ref.score == g.match.score and ref.pending == g.match.pending
        assert ref.unlocked == g.match.unlocked and ref.winner == g.match.winner
        assert ref.round_no == g.match.round_no == len(g.records)
        for rec in g.records:
            assert rec["source"] == "demo" and rec["meta"]["match_seed"] == g.seed
            assert rec["meta"]["bots"] == bot and rec["meta"]["human_seats"] == [0, 2]
            assert rec["result_claimed"] is not None
        n_rounds += len(g.records)
    assert n_rounds >= 30


def test_DM1b_jsonl_file_replays():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "demo.jsonl")
        g = play_full_match("testfile", "random", random.Random(5), record_to=path)
        recs = load_records(path)
        assert len(recs) == len(g.records) > 0
        assert all(replay(r).ok for r in recs)
        # tools/replay.py biçimi: domino-research aracı varsa onunla da doğrula
        tool = os.path.join(RESEARCH_SRC, "tools", "replay.py")
        if os.path.exists(tool):
            import subprocess
            out = subprocess.run([sys.executable, tool, path], capture_output=True, text=True,
                                 env={**os.environ, "PYTHONUTF8": "1"})
            assert out.returncode == 0, out.stdout + out.stderr
            assert f"{len(recs)}/{len(recs)} kayıt tamam" in out.stdout


def test_DM1c_rules_first_round_forced_and_threshold():
    g = DemoGame(seed="testrules", bot="random", record_to="")
    r = g.start_round()
    assert r.forced_opening == (1, 1) and (1, 1) in r.hand(r.starter)      # R-04
    assert g.match.cfg.target == 101 and g.match.cfg.threshold == 13      # R-03, R-16
    # maç bitince yeni maç
    g2 = play_full_match("testnew", "random", random.Random(2))
    old_seed = g2.seed
    g2.start_round()
    assert g2.seed != old_seed and g2.match.round_no == 1 and g2.match.score == [0, 0]


# ── T-DM2 sızıntı ────────────────────────────────────────────

def shuffled_hidden(r: Round, p: int, rng) -> Round:
    """Aynı public geçmiş ve p'nin eli; diğer üç elin kalan taşları karıştırılmış (tutarlı olmayabilir)."""
    r2 = r.clone()
    others = [q for q in range(4) if q != p]
    pool = [t for q in others for t in r2._hands[q]]
    rng.shuffle(pool)
    i = 0
    for q in others:
        s = len(r2._hands[q])
        r2._hands[q] = set(pool[i:i + s])
        i += s
    return r2


def test_DM2_bot_decision_invariant_to_hidden_hands():
    rng = random.Random(7)
    changed = decisions = 0
    for bot in ("ismcts@20", "ismcts-rand@20"):
        for gi in range(3):
            g = DemoGame(seed=f"testleak{gi}", bot=bot, record_to="")
            g.start_round()
            while not g.round_over:
                r = g.round
                p = r.current_player
                if p in BOT_SEATS:
                    r2 = shuffled_hidden(r, p, rng)
                    changed += any(r2.hand(q) != r.hand(q) for q in range(4) if q != p)
                    seed = f"{g.seed}:{g.match.round_no}:s{p}"
                    ms = g.match.public_state()
                    m1 = bot_decide(make_agent(bot, seed), r.observation(p), r.legal_moves(), ms)
                    m2 = bot_decide(make_agent(bot, seed), r2.observation(p), r2.legal_moves(), ms)
                    assert m1 == m2
                    # demonun KENDİ ajanı (o ana kadarki iç durumu ve RNG'siyle): karıştırılmış gizli ellerle aynı karar
                    twin = copy.deepcopy(g.agents[p])
                    m3 = bot_decide(twin, r2.observation(p), r2.legal_moves(), ms)
                    assert g.bot_step()[1] == m3
                    decisions += 1
                else:
                    g.human_move(p, human_choice(g, rng))
    assert decisions > 20 and changed > 10


class _Spy:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def act(self, obs, legal, match=None):
        self.calls.append((obs, legal, match))
        return self.inner.act(obs, legal, match)


def test_DM2b_agent_receives_only_observation():
    g = DemoGame(seed="testspy", bot="ismcts@10", record_to="")
    rng = random.Random(3)
    allowed_match_keys = set(Match().public_state())
    n = 0
    for _ in range(3):
        g.start_round()
        g.agents = {s: _Spy(a) for s, a in g.agents.items()}
        while not g.round_over:
            p = g.round.current_player
            if p in BOT_SEATS:
                before = len(g.agents[p].calls)
                g.bot_step()
                obs, legal, ms = g.agents[p].calls[before]
                assert type(obs) is Observation and obs.player == p
                assert all(type(m) is Move for m in legal)
                assert isinstance(ms, dict) and set(ms) == allowed_match_keys
                # gözlemdeki taşlar: kendi eli + masa; başka elin taşı yok
                hidden = set(ALL_TILES) - set(obs.hand) - {tuple(sorted(t)) for t in obs.board}
                for q in range(4):
                    if q != p:
                        assert not (set(g.round.initial_hands[q]) & set(obs.hand))
                assert len(hidden) == sum(obs.hand_sizes) - len(obs.hand)
                n += 1
            else:
                g.human_move(p, human_choice(g, rng))
        if g.match.winner is not None:
            break
    assert n > 10


def test_DM2c_bot_step_does_not_pass_round_object():
    """bot_decide'a giden argümanlarda Round/Match nesnesi yok (yapısal kontrol)."""
    seen = []
    orig = game.bot_decide

    def wrapped(agent, obs, legal, ms):
        seen.append((obs, legal, ms))
        return orig(agent, obs, legal, ms)
    game.bot_decide = wrapped
    try:
        g = play_full_match("testwrap", "random", random.Random(9), max_rounds=2)
    finally:
        game.bot_decide = orig
    assert seen
    for obs, legal, ms in seen:
        for x in (obs, ms, *legal):
            assert not isinstance(x, (Round, Match))
        assert all(not isinstance(v, (Round, Match)) for v in vars(obs).values())


# ── T-DM3 protokol ───────────────────────────────────────────

def test_DM3_protocol_roundtrip():
    assert move_from_dict(move_to_dict(PASS), False) == PASS
    assert move_from_dict({"tile": [3, 1], "to_left": True, "is_pass": False}, True) == Move((1, 3), "R")
    for side in ("L", "R"):
        m = Move((2, 5), side)
        assert move_from_dict(move_to_dict(m), False) == m
    # aynalı R: uçlar eşitken frontend sağa bırakırsa motor kabul eder ve sağa koyar (D-18)
    g = DemoGame(seed="testmirror", bot="random", record_to="")
    found = False
    for k in range(200):
        g.start_round()
        rng = random.Random(k)
        while not g.round_over:
            r = g.round
            p = r.current_player
            if p in HUMAN_SEATS and r.board and r.left_end == r.right_end:
                legal = [m for m in r.legal_moves() if not m.is_pass]
                if legal:
                    m = legal[0]
                    right_before = r.board[-1]
                    g.human_move(p, {"tile": list(m.tile), "to_left": False, "is_pass": False})
                    assert r.board[-1] != right_before and r.board[-1][0] == right_before[1]
                    found = True
                    break
            if p in BOT_SEATS:
                g.bot_step()
            else:
                g.human_move(p, human_choice(g, rng))
        if found:
            break
    assert found


def test_DM3b_illegal_and_wrong_turn_rejected():
    g = DemoGame(seed="testillegal", bot="random", record_to="")
    g.start_round()
    r = g.round
    p = r.current_player
    other_human = next(h for h in HUMAN_SEATS if h != p) if p in HUMAN_SEATS else HUMAN_SEATS[0]
    try:
        g.human_move(other_human, {"is_pass": True})
        raise AssertionError("sıra dışı hamle kabul edildi")
    except IllegalMove:
        pass
    if p in HUMAN_SEATS:
        bad = next(t for t in r.hand(p) if t != (1, 1))
        try:
            g.human_move(p, {"tile": list(bad), "to_left": False, "is_pass": False})   # R-04: yalnız 1|1
            raise AssertionError("zorunlu açılış atlandı")
        except IllegalMove:
            pass


# ── T-DM4 insana giden durum ─────────────────────────────────

def test_DM4_state_message_hides_other_hands():
    g = DemoGame(seed="teststate", bot="random", record_to="")
    rng = random.Random(4)
    g.start_round()
    while not g.round_over:
        for h in HUMAN_SEATS:
            st = g.state_for(h, {})
            text = json.dumps({k: v for k, v in st.items() if k != "your_hand"})
            assert st["your_hand"] == [list(t) for t in g.round.hand(h)]
            assert "hands" not in st
            for q in range(4):
                if q != h:
                    assert json.dumps([list(t) for t in g.round.hand(q)]) not in text or not g.round.hand(q)
            assert st["team_scores"] == list(g.match.score)
            legal = st["legal_moves"]
            assert bool(legal) == (g.round.current_player == h)
        p = g.round.current_player
        if p in BOT_SEATS:
            g.bot_step()
        else:
            g.human_move(p, human_choice(g, rng))
    msg = g.round_over_message()
    res = g.last_result
    assert len(msg["hands"]) == 4
    a, b = msg["team_a_score"], msg["team_b_score"]
    if res.winner_team == 0:
        assert a < b
    elif res.winner_team == 1:
        assert b < a
    else:
        assert a == b


# ── T-DM5 WebSocket uçtan uca ────────────────────────────────

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_DM5_websocket_end_to_end():
    try:
        import uvicorn
        import websockets
    except ImportError as ex:
        raise unittest.SkipTest(f"uvicorn/websockets yok: {ex}")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ws.jsonl")
        os.environ["DOMINO_RECORD_PATH"] = path
        os.environ["DOMINO_BOT"] = "ismcts@10"
        import main
        main.BOT_DELAY = 0.0
        main.ROOM_SEED = "testws"
        main.rooms.clear()
        main.waiting_room = None
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
        th = threading.Thread(target=server.run, daemon=True)
        th.start()
        t0 = time.time()
        while not server.started:
            assert time.time() - t0 < 20, "sunucu başlamadı"
            time.sleep(0.05)
        try:
            log = asyncio.run(_ws_session(port, rounds=4))
        finally:
            server.should_exit = True
            th.join(10)
            os.environ.pop("DOMINO_RECORD_PATH", None)
            os.environ.pop("DOMINO_BOT", None)
        recs = load_records(path)
        assert len(recs) == 4
        assert all(replay(r).ok for r in recs)
        ref = reference_match("testws:m1", [r.to_json() for r in recs])
        assert log["last_scores"] == ref.score
        assert log["game_overs"] == 4
        for rec, final_board in zip(recs, log["final_boards"]):
            rep = replay(rec)
            assert [list(t) for t in rep.round.board] == final_board
        # yayınlanan her hamle (kim, ne, insan/bot) kayıttaki olayla birebir (B-09: hamleyi yapan doğru etiketli)
        for rec, moves in zip(recs, log["moves"]):
            assert len(moves) == len(rec.events)
            for (kind, player, mv), ev in zip(moves, rec.events):
                assert player == ev["player"]
                assert kind == ("agent_move" if player in BOT_SEATS else "human_move")
                if ev.get("pass"):
                    assert mv["is_pass"]
                else:
                    assert sorted(mv["tile"]) == sorted(ev["tile"]) and mv["to_left"] == (ev["side"] == "L")
        assert log["starts"] == 4                      # ara/ikinci 'Tekrar Oyna' yeni tur başlatmadı
        assert log["mid_rematch_sent"]
        assert log["leak_free"]


async def _ws_session(port, rounds):
    import websockets
    uri = f"ws://127.0.0.1:{port}/ws/"
    log = {"game_overs": 0, "final_boards": [], "last_scores": None, "starts": 0, "mid_rematch_sent": False,
           "leak_free": True, "moves": []}
    rng = random.Random(11)
    done = asyncio.Event()

    async def human(name, slot_expected):
        async with websockets.connect(uri + name) as ws:
            joined = json.loads(await ws.recv())
            assert joined["type"] == "joined" and joined["slot"] == slot_expected
            last_board = None
            sent_mid_rematch = False
            my_overs = 0
            acted = None
            while not done.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                except asyncio.TimeoutError:
                    raise AssertionError(f"{name}: mesaj gelmedi")
                msg = json.loads(raw)
                t = msg["type"]
                if t == "game_started" and slot_expected == 0:
                    log["starts"] += 1
                    log["moves"].append([])
                elif t in ("human_move", "agent_move"):
                    last_board = msg["board"]
                    if slot_expected == 0:
                        log["moves"][-1].append((t, msg["player"], msg["move"]))
                elif t == "game_state":
                    if "hands" in msg:
                        log["leak_free"] = False
                    log["last_scores"] = msg["team_scores"]
                    key = (json.dumps(msg["board"]), msg["hand_sizes"], msg["pass_count"], msg["match"]["round_no"])
                    if msg["is_your_turn"] and key != acted:      # aynı durum iki kez gelirse bir kez oyna
                        acted = key
                        legal = msg["legal_moves"]
                        mv = rng.choice(legal)
                        await ws.send(json.dumps({"type": "move", **mv}))
                        # bir kez: tur ortasında 'Tekrar Oyna' yeni tur başlatmamalı
                        if slot_expected == 1 and not sent_mid_rematch:
                            sent_mid_rematch = True
                            await ws.send(json.dumps({"type": "rematch"}))
                            log["mid_rematch_sent"] = True
                elif t == "game_over":
                    if slot_expected == 0:
                        log["game_overs"] += 1
                        log["final_boards"].append(last_board)
                        if log["game_overs"] >= rounds:
                            done.set()
                            break
                        await ws.send(json.dumps({"type": "rematch"}))
                    else:
                        my_overs += 1
                        if my_overs < rounds:
                            await ws.send(json.dumps({"type": "rematch"}))   # ikinci basış: biri yok sayılmalı
                elif t == "error":
                    raise AssertionError(f"{name}: hata {msg}")

    await asyncio.gather(human("A", 0), _delayed(human("B", 1)))
    return log


async def _delayed(coro):
    await asyncio.sleep(0.3)
    return await coro


# ── T-DM6 vendored paket ─────────────────────────────────────

def test_DM7_same_name_marks_team_full_info():
    import main
    assert main.same_player({0: "Murad", 1: " murad "}) and not main.same_player({0: "Murad", 1: "Ali"})
    assert not main.same_player({0: "", 1: ""}) and not main.same_player({0: "Murad"})

    class FakeWS:
        async def send_text(self, _):
            pass

    old = main.START_DELAY
    main.START_DELAY = 0
    try:
        for names, expect in (({0: "Murad", 1: "MURAD"}, True), ({0: "Murad", 1: "Ali"}, False)):
            with tempfile.TemporaryDirectory() as d:
                room = main.GameRoom("t", seed="dm7")
                room.human_slots = {0: FakeWS(), 1: FakeWS()}
                room.human_names = dict(names)
                asyncio.run(room.start_game())
                g = room.game
                g.record_to = os.path.join(d, "x.jsonl")
                rng = random.Random(7)
                while not g.round_over:
                    r = g.round
                    if r.current_player in (0, 2):
                        g.human_move(r.current_player, move_to_dict(rng.choice(r.legal_moves())))
                    else:
                        g.bot_step()
                rec = g.records[-1]
                assert rec["meta"]["team_full_info"] is expect
                text = json.dumps(rec, ensure_ascii=False).casefold()
                assert "murad" not in text and "ali\"" not in text
    finally:
        main.START_DELAY = old


def test_DM6_vendored_package_matches_research_repo():
    if not os.path.isdir(os.path.join(RESEARCH_SRC, "domino")):
        raise unittest.SkipTest("domino-research kaynağı yok")
    sys.path.insert(0, os.path.join(HERE, "tools"))
    import sync_domino
    assert sync_domino.check(RESEARCH_SRC) == [], "backend/domino kaynakla aynı değil: tools/sync_domino.py"
    import subprocess
    head = subprocess.run(["git", "-C", RESEARCH_SRC, "log", "-1", "--format=%H", "--", "domino"],
                          capture_output=True, text=True).stdout.strip()
    pinned = game.domino_version()
    # sabitlenen commit, domino/ klasörünün son değiştiği commit'ten yeni (ya da aynı) olmalı
    anc = subprocess.run(["git", "-C", RESEARCH_SRC, "merge-base", "--is-ancestor", head, pinned])
    assert anc.returncode == 0, f"DOMINO_VERSION {pinned[:7]} domino/'nun son değişikliğini ({head[:7]}) içermiyor"


# ── T-DM8 yeniden bağlanma (D-32) ────────────────────────────

def test_DM8_reconnect_with_token_keeps_seat():
    try:
        import uvicorn
        import websockets  # noqa: F401
    except ImportError as ex:
        raise unittest.SkipTest(f"uvicorn/websockets yok: {ex}")
    with tempfile.TemporaryDirectory() as d:
        os.environ["DOMINO_RECORD_PATH"] = os.path.join(d, "rc.jsonl")
        os.environ["DOMINO_BOT"] = "ismcts@10"
        import main
        main.BOT_DELAY = 0.0
        main.ROOM_SEED = "testrc"
        main.RECONNECT_GRACE = 30.0
        main.rooms.clear()
        main.waiting_room = None
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
        th = threading.Thread(target=server.run, daemon=True)
        th.start()
        t0 = time.time()
        while not server.started:
            assert time.time() - t0 < 20, "sunucu başlamadı"
            time.sleep(0.05)
        try:
            asyncio.run(_reconnect_session(port, main))
        finally:
            server.should_exit = True
            th.join(10)
            os.environ.pop("DOMINO_RECORD_PATH", None)
            os.environ.pop("DOMINO_BOT", None)
            main.RECONNECT_GRACE = 60.0
        recs = load_records(os.path.join(d, "rc.jsonl"))
        assert len(recs) == 1 and replay(recs[0]).ok


async def _recv_type(ws, types, timeout=30):
    """`types` türünden ilk mesaja kadar oku (arada gelenler atlanır)."""
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if msg["type"] in types:
            return msg


async def _reconnect_session(port, main):
    import websockets
    uri = f"ws://127.0.0.1:{port}/ws/"
    a1 = await websockets.connect(uri + "A")
    ja = json.loads(await a1.recv())
    assert ja["slot"] == 0 and ja["resumed"] is False and ja["token"]
    b = await websockets.connect(uri + "B")
    jb = json.loads(await b.recv())
    assert jb["slot"] == 1 and jb["room_id"] == ja["room_id"] and jb["token"] != ja["token"]
    st = await _recv_type(a1, ("game_state",))
    hand_before = st["your_hand"]

    # 1) kopma: koltuk tutulur, diğer oyuncu haber alır; jetonla dönünce aynı koltuk ve aynı el
    await a1.close()
    left = await _recv_type(b, ("player_left",))
    assert left["slot"] == 0
    assert 0 in main.rooms[ja["room_id"]].tokens
    a2 = await websockets.connect(uri + "A?token=" + ja["token"])
    j2 = json.loads(await a2.recv())
    assert j2["resumed"] is True and j2["slot"] == 0 and j2["room_id"] == ja["room_id"]
    st2 = await _recv_type(a2, ("game_state",))
    assert st2["your_index"] == 0 and st2["your_hand"] == hand_before   # sıra A'da değilken eli değişmez
    back = await _recv_type(b, ("player_back",))
    assert back["slot"] == 0

    # 2) eski bağlantı açıkken (sayfa yenileme) aynı jetonla yeni bağlantı: eskisi kapanır, yenisi koltukta kalır
    a3 = await websockets.connect(uri + "A?token=" + ja["token"])
    j3 = json.loads(await a3.recv())
    assert j3["resumed"] is True and j3["slot"] == 0
    st3 = await _recv_type(a3, ("game_state",))
    try:
        while True:
            await asyncio.wait_for(a2.recv(), timeout=5)
    except websockets.ConnectionClosed:
        pass
    await asyncio.sleep(0.2)
    room = main.rooms[ja["room_id"]]
    assert room.human_slots.get(0) is not None and 0 not in room.gone   # eski bağlantının kapanışı koltuğu düşürmedi

    # 3) yeni bağlantıyla turu bitir (iki taraf rastgele yasal hamle)
    rng = random.Random(5)

    async def play(ws, first=None):
        msg = first
        while True:
            if msg is None:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            if msg["type"] == "game_over":
                return msg
            if msg["type"] == "game_state" and msg["is_your_turn"]:
                await ws.send(json.dumps({"type": "move", **rng.choice(msg["legal_moves"])}))
            elif msg["type"] == "error":
                raise AssertionError(msg)
            msg = None

    await b.send(json.dumps({"type": "rematch"}))      # tur ortasında: durumu yeniden gönderir (atlanan mesajlar)
    over_a, over_b = await asyncio.gather(play(a3, st3), play(b))
    assert over_a["hands"] == over_b["hands"]

    # 4) tur bittikten sonra dönen oyuncu tur sonu mesajını da alır
    await a3.close()
    await _recv_type(b, ("player_left",))
    a4 = await websockets.connect(uri + "A?token=" + ja["token"])
    assert json.loads(await a4.recv())["resumed"] is True
    await _recv_type(a4, ("game_state",))
    assert (await _recv_type(a4, ("game_over",)))["hands"] == over_a["hands"]

    # 5) geçersiz jeton: yeni oda (önce oyun başlamadan bekleme odasından çıkan: koltuk tutulmaz, sonraki ikili eşleşir)
    w = await websockets.connect(uri + "W")
    jw = json.loads(await w.recv())
    assert jw["slot"] == 0 and jw["waiting"]
    await w.close()
    await asyncio.sleep(0.3)
    assert jw["room_id"] not in main.rooms and main.waiting_room is None
    p1 = await websockets.connect(uri + "P")
    p2 = await websockets.connect(uri + "Q")
    jp1, jp2 = json.loads(await p1.recv()), json.loads(await p2.recv())
    assert jp1["slot"] == 0 and jp2["slot"] == 1 and jp1["room_id"] == jp2["room_id"] != jw["room_id"]
    await _recv_type(p2, ("game_state",))
    await p1.close()
    await p2.close()
    c = await websockets.connect(uri + "C?token=yanlis")
    jc = json.loads(await c.recv())
    assert jc["resumed"] is False and jc["room_id"] != ja["room_id"]
    await c.close()

    # 6) süre dolunca jeton geçersiz: koltuk bırakılır, aynı jetonla dönen yeni odaya düşer
    main.RECONNECT_GRACE = 0.3
    await a4.close()
    await _recv_type(b, ("player_left",))
    await asyncio.sleep(0.8)
    assert 0 not in main.rooms[ja["room_id"]].tokens
    a5 = await websockets.connect(uri + "A?token=" + ja["token"])
    j5 = json.loads(await a5.recv())
    assert j5["resumed"] is False and j5["room_id"] != ja["room_id"]
    await a5.close()
    await b.close()


# ── T-DM9: 2 kişilik mod (R-20..R-28, domino.engine.Match2) ──────────

from domino.engine import Match2, Observation2, Round2, norm  # noqa: E402


def replay_2p(rec: dict) -> Round2:
    """2 kişilik kaydı Round2'de yeniden oynat: çekişler motorda kendiliğinden, kayıttakiyle aynı olmalı."""
    r = Round2([[tuple(t) for t in h] for h in rec["initial_hands"]], [tuple(t) for t in rec["initial_boneyard"]],
               rec["starter"], tuple(rec["forced_opening"]) if rec["forced_opening"] else None)
    plays = [e for e in rec["events"] if not e.get("draw")]
    for e in plays:
        if e.get("pass"):
            r.apply(PASS)
        else:
            r.apply(Move(norm(tuple(e["tile"])), e["side"] if r.board else "R"))
    got = [({"player": e.player, "tile": list(e.tile), "side": e.side} if e.kind == "play"
            else {"player": e.player, e.kind: True}) for e in r.events]
    assert got == rec["events"], "olay dizisi (çekişler dahil) kayıtla aynı değil"
    res = r.result
    assert res is not None and res.points == rec["result"]["points"] and res.winner_team == rec["result"]["winner_team"]
    return r


def play_full_match_2p(seed, rng, record_to="", max_rounds=80):
    g = DemoGame(seed=seed, record_to=record_to, players=2)
    draws_seen = 0
    while True:
        g.start_round()
        while not g.round_over:
            p = g.round.current_player
            if p in g.bot_seats:
                g.bot_step()
            else:
                g.human_move(p, human_choice(g, rng))
            draws_seen += len(g.draws)
        if g.match.winner is not None or g.match.round_no >= max_rounds:
            return g, draws_seen


def test_DM9_two_player_match_replays_and_equals_match2():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "2p.jsonl")
        total_draws = 0
        for s in range(6):
            g, nd = play_full_match_2p(f"t2p{s}", random.Random(s), record_to=path)
            assert g.match.winner in (0, 1) and isinstance(g.match, Match2)
            assert g.human_seats == (0,) and g.bot_seats == (1,) and g.bot == game.DEFAULT_BOT_2P
            m = Match2(g.seed)                                   # bağımsız referans maç
            for rec in g.records:
                assert rec["mode"] == "2p"
                r_ref = m.new_round()
                assert [list(map(list, h)) for h in r_ref.initial_hands] == rec["initial_hands"]
                assert [list(t) for t in r_ref.initial_boneyard] == rec["initial_boneyard"]
                r = replay_2p(rec)
                m.apply_result(r.result)
            assert m.score == g.match.score and m.winner == g.match.winner
            total_draws += nd
        assert total_draws > 0
        lines = [json.loads(x) for x in open(path, encoding="utf-8")]
        assert lines and all(x["mode"] == "2p" for x in lines)


def test_DM9b_two_player_state_hides_opponent_and_boneyard():
    g = DemoGame(seed="t2pstate", record_to="", players=2)
    g.start_round()
    rng = random.Random(3)
    seen_obs = []

    class Spy:
        def act(self, obs, legal, match):
            seen_obs.append(obs)
            return legal[0]
    g.agents = {1: Spy()}
    while not g.round_over:
        st = g.state_for(0, {0: "A"})
        assert st["num_players"] == 2 and len(st["hand_sizes"]) == 2 and len(st["missing_numbers"]) == 2
        assert st["boneyard_size"] == len(g.round.boneyard)
        assert not any(any(row) for row in st["missing_numbers"])          # R-28: kalıcı yokluk gösterilmez
        assert "boneyard" not in st and "hands" not in st
        assert sorted(map(tuple, st["your_hand"])) == sorted(g.round.hand(0))
        if g.round.current_player == 1:
            g.bot_step()
        else:
            g.human_move(0, human_choice(g, rng))
    assert seen_obs and all(isinstance(o, Observation2) and not hasattr(o, "boneyard") for o in seen_obs)
    over = g.round_over_message()
    assert over["num_players"] == 2 and len(over["hands"]) == 2
    assert "Sen " in over["winner"].splitlines()[1] and " A " not in over["winner"]


def test_DM9c_two_player_websocket():
    try:
        import uvicorn
        import websockets
    except ImportError as ex:
        raise unittest.SkipTest(f"uvicorn/websockets yok: {ex}")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ws2p.jsonl")
        os.environ["DOMINO_RECORD_PATH_2P"] = path
        import main
        main.BOT_DELAY = 0.0
        main.ROOM_SEED = "testws2p"
        main.rooms.clear()
        main.waiting_room = None
        port = _free_port()
        server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
        th = threading.Thread(target=server.run, daemon=True)
        th.start()
        t0 = time.time()
        while not server.started:
            assert time.time() - t0 < 20, "sunucu başlamadı"
            time.sleep(0.05)

        async def session():
            log = {"overs": 0, "drew": [], "bone": []}
            rng = random.Random(5)
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws/Solo?mode=2") as ws:
                joined = json.loads(await ws.recv())
                assert joined["type"] == "joined" and joined["slot"] == 0 and not joined["waiting"]
                assert joined["num_players"] == 2
                assert main.waiting_room is None                    # 4 kişilik bekleme odası etkilenmez
                acted = None
                while log["overs"] < 3:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                    t = msg["type"]
                    if t == "drew":
                        log["drew"].append(msg["player"])
                    elif t == "game_state":
                        assert msg["num_players"] == 2 and "hands" not in msg
                        log["bone"].append(msg["boneyard_size"])
                        key = (json.dumps(msg["board"]), msg["hand_sizes"], msg["pass_count"], msg["match"]["round_no"])
                        if msg["is_your_turn"] and key != acted:
                            acted = key
                            await ws.send(json.dumps({"type": "move", **rng.choice(msg["legal_moves"])}))
                    elif t == "game_over":
                        assert len(msg["hands"]) == 2
                        log["overs"] += 1
                        if log["overs"] < 3:
                            await ws.send(json.dumps({"type": "rematch"}))
                    elif t == "error":
                        raise AssertionError(msg)
            return log
        try:
            log = asyncio.run(session())
        finally:
            server.should_exit = True
            th.join(10)
            os.environ.pop("DOMINO_RECORD_PATH_2P", None)
        recs = [json.loads(x) for x in open(path, encoding="utf-8")]
        assert len(recs) == 3
        for rec in recs:
            replay_2p(rec)
        rec_draws = [e["player"] for rec in recs for e in rec["events"] if e.get("draw")]
        assert log["drew"] == rec_draws                       # her çekiş yayınlandı, kim çektiği doğru
        assert all(b >= 2 for b in log["bone"])
