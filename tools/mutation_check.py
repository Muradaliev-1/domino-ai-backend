"""Demo backend mutasyon kontrolü: her mutant bilerek bozulmuş bir kopya; tests/run.py her birini YAKALAMALI.
Önce mutasyonsuz temel koşu geçmek zorunda (geçmezse çıkış kodu 2). Backend klasöründen: python tools/mutation_check.py"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MUTANTS = [
    ("team_full_info kayda hep False", "game.py", '"team_full_info": self.team_full_info,', '"team_full_info": False,'),
    ("aynı ad karşılaştırması büyük/küçük harfe duyarlı", "main.py",
     'str(names.get(s, "")).strip().casefold()', 'str(names.get(s, ""))'),
    ("oda team_full_info bayrağını oyuna vermiyor", "main.py",
     "        self.game.team_full_info = self.players == 4 and same_player(self.human_names)\n", ""),
    ("el sızıntısı: insana sonraki oyuncunun eli", "game.py",
     '"your_hand": [list(t) for t in r.hand(player)]', '"your_hand": [list(t) for t in r.hand((player + 1) % 4)]'),
    ("maç sonucu uygulanmıyor", "game.py", "        self.match.apply_result(res)\n", "        pass\n"),
    ("taraf ters çevrilmiş", "game.py", 'return Move(t, "L" if d.get("to_left") else "R")',
     'return Move(t, "R" if d.get("to_left") else "L")'),
    ("bota Round nesnesi gidiyor", "game.py",
     "mv = bot_decide(self.agents[p], r.observation(p), r.legal_moves(), self.match.public_state())",
     "mv = bot_decide(self.agents[p], r.observation(p), r.legal_moves(), r)"),
    ("bot RNG'si gizli elle tohumlanıyor (sızıntı)", "game.py",
     "mv = bot_decide(self.agents[p], r.observation(p), r.legal_moves(), self.match.public_state())",
     "self.agents[p].rng.seed(str(r.hand((p + 1) % 4)))\n"
     "        mv = bot_decide(self.agents[p], r.observation(p), r.legal_moves(), self.match.public_state())"),
    ("bot partnerin gözlemini görüyor", "game.py",
     "mv = bot_decide(self.agents[p], r.observation(p),", "mv = bot_decide(self.agents[p], r.observation((p + 2) % 4),"),
    ("tur sonu sayılan toplamlar ters", "game.py", 'counted[1 - res.winner_team] = res.points',
     'counted[res.winner_team] = res.points'),
    ("kayıt yazılmıyor", "game.py", "        self.records.append(rec)\n", "        pass\n"),
    ("13 eşiği yok", "game.py", "        self.match = self._new_match()\n        self.round",
     "        self.match = self._new_match()\n        self.match.cfg.threshold = 0\n        self.round"),
    ("tur ortasında 'Tekrar Oyna' yeni tur başlatıyor (B-10)", "main.py",
     "if self.game is not None and not self.game.round_over:", "if False:"),
    ("hamleyi yapan yanlış etiketli (B-09)", "main.py", '"player": actor,',
     '"player": (g.round.current_player - 1) % 4,'),
    ("insana giden durumda maç skoru yerine el toplamları", "game.py",
     '"team_scores": list(self.match.score),', '"team_scores": list(r.team_totals()) if r else [0, 0],'),
    ("yeniden bağlanma: jeton yok sayılıyor (D-32)", "main.py",
     "    seat = find_seat(token)\n", "    seat = None\n"),
    ("yeniden bağlanma: eski bağlantının kapanışı yeni koltuğu düşürüyor", "main.py",
     "    if room.human_slots.get(slot) is not ws:\n        return", "    if False:\n        return"),
    ("yeniden bağlanana durum gönderilmiyor", "main.py",
     "        await self.send_to(slot, self.game_state_message(slot))\n        if self.game.round_over:",
     "        pass\n        if self.game.round_over:"),
    ("yeniden bağlanana tur sonu gönderilmiyor", "main.py",
     "            await self.send_to(slot, self.game.round_over_message())\n", "            pass\n"),
    ("süre dolunca jeton silinmiyor", "main.py", "    room.gone.pop(slot, None)\n    room.tokens.pop(slot, None)\n",
     "    room.gone.pop(slot, None)\n"),
    ("kopan koltuk hemen bırakılıyor (tutulmuyor)", "main.py",
     "    asyncio.create_task(expire_seat(room_id, slot, gone_at))\n",
     "    room.tokens.pop(slot, None)\n    room.gone.pop(slot, None)\n"),
    ("bekleme odasından çıkanın koltuğu da tutuluyor (sonraki ikili takılır)", "main.py",
     "    if not room.game_started:\n        # oyun başlamadan", "    if False:\n        # oyun başlamadan"),
    ("vendored motor kurala aykırı (R-12 yok)", "domino/engine.py",
     "        return ZERO_ZERO_SINGLE_VALUE\n", "        return 0\n"),
    ("2p: bot arama ajanı (4 kişilik varsayımlı)", "game.py", "if players == 4 else DEFAULT_BOT_2P)", "if True else DEFAULT_BOT_2P)"),
    ("2p: çekiş yayını bir olay kaymış", "game.py", "self.round.events[n + 1:]", "self.round.events[n + 2:]"),
    ("2p: gözlemde bazar sayısı 0", "game.py",
     '"boneyard_size": len(r.boneyard) if n == 2 and r is not None else 0,', '"boneyard_size": 0,'),
    ("2p: pas yokluğu kalıcı gösteriliyor (R-28)", "game.py",
     "            missing = [set() for _ in range(n)]", "            missing = missing_from_events(r.events)[:n]"),
    ("2p: oda ikinci insanı bekliyor", "main.py", "(2 if self.players == 4 else 1)", "2"),
    ("2p: kayıtta başlangıç bazarı yerine kalan bazar", "game.py",
     '"initial_boneyard": [list(t) for t in r.initial_boneyard]', '"initial_boneyard": [list(t) for t in r.boneyard]'),
    ("2p: oyun 4 kişilik açılıyor", "main.py", "room = GameRoom(room_id, seed=ROOM_SEED, players=2)", "room = GameRoom(room_id, seed=ROOM_SEED)"),
]


def run_tests(root):
    out = subprocess.run([sys.executable, os.path.join(root, "tests", "run.py")], capture_output=True, text=True,
                         cwd=root, env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return out.returncode == 0, out.stdout


def copy_tree(dst):
    for name in ("game.py", "main.py", "DOMINO_VERSION"):
        shutil.copy2(os.path.join(HERE, name), dst)
    for d in ("domino", "tests", "tools"):
        shutil.copytree(os.path.join(HERE, d), os.path.join(dst, d), ignore=shutil.ignore_patterns("__pycache__"))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "base")
        os.makedirs(base)
        copy_tree(base)
        ok, out = run_tests(base)
        if not ok:
            print("TEMEL KOŞU BAŞARISIZ (mutasyonsuz):\n" + out[-3000:])
            return 2
        print("temel koşu geçti")
        caught = 0
        for i, (name, rel, old, new) in enumerate(MUTANTS):
            d = os.path.join(tmp, f"m{i}")
            os.makedirs(d)
            copy_tree(d)
            path = os.path.join(d, rel)
            src = open(path, encoding="utf-8").read()
            if src.count(old) != 1:
                print(f"  ??  {name}: kalıp {src.count(old)} kez bulundu (1 olmalı)")
                continue
            # vendored motoru değiştiren mutantta T-DM6 (kaynakla birebir) zaten yakalar; asıl soru davranış testleri
            open(path, "w", encoding="utf-8").write(src.replace(old, new))
            ok, out = run_tests(d)
            failed = [l.split()[-1] for l in out.splitlines() if l.strip().startswith("FAIL")]
            if ok:
                print(f"  HAYATTA  {name}")
            else:
                caught += 1
                print(f"  yakalandı  {name}  ({', '.join(failed)})")
        print(f"\n{caught}/{len(MUTANTS)} mutant yakalandı")
        return 0 if caught == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
