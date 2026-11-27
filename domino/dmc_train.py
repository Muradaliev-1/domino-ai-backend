"""
DMC öğrenicisi (PyTorch), Faz 5, D-28. Aktörler `domino/dmc.py` (numpy, çok süreçli) ile self-play turları oynar;
öğrenici Q(s, a)'yı Monte Carlo hedefine (tur sonucu, hamleyi yapan takım açısından) MSE ile yaklaştırır.

Döngü (eşzamanlı, yeniden üretilebilir sıralı): her iterasyonda
  1) güncel ağ numpy'ye aktarılır, `rounds_per_iter` self-play turu `workers` süreçte oynanır (ε-açgözlü),
  2) toplanan örnekler karıştırılıp `epochs` kez mini-yığınlarla öğrenilir,
  3) `eval_every` iterasyonda bir kontrol noktası kaydedilir ve arena (L1, eşleştirilmiş) ile değerlendirilir.
Asenkron DouZero aktör/öğrenici yerine eşzamanlı döngü seçildi: tek makine, sonuç tohumla (CPU'da) tekrar edilebilir.

Girdi mimarisi ablation'ı (D-30): `seq` verilirse ağ `QNet` = dizi kodlayıcı (LSTM ya da ön-LN transformer, olay
dizisi `encode_events`) + MLP; kodlayıcının çıktısı F-vektörüne eklenir. numpy karşılıkları `dmc.lstm_embed` ve
`dmc.transformer_embed` (birebirlik T-DMC8). Dizi yoksa ağ eskisi gibi `nn.Sequential` (E-04q ile aynı ilklendirme).
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .dmc import REWARDS, Weights, selfplay_chunk, selfplay_match_chunk, seq_out_dim, vec_dim
from .features import EVENT_WIDTH, N_ACTIONS


def _mlp(in_dim: int, hidden: Sequence[int]) -> nn.Sequential:
    sizes = [in_dim] + list(hidden)
    mods: List[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        mods += [nn.Linear(a, b), nn.ReLU()]
    mods.append(nn.Linear(sizes[-1], N_ACTIONS))
    return nn.Sequential(*mods)


class LSTMEnc(nn.Module):
    """Tek katmanlı LSTM, dizinin son gizli durumu; boş dizi → sıfır (numpy `lstm_embed` ile aynı)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.lstm = nn.LSTM(EVENT_WIDTH, int(cfg["hidden"]), batch_first=True)

    def forward(self, toks: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        lc = lens.clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(toks, lc.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return h[-1] * (lens > 0).unsqueeze(1).to(h.dtype)

    def names(self):
        m = self.lstm
        return {"W_ih": m.weight_ih_l0, "W_hh": m.weight_hh_l0, "b_ih": m.bias_ih_l0, "b_hh": m.bias_hh_l0}


class TFBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.h = heads
        self.ln1, self.qkv, self.out = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ln2, self.ff1, self.ff2 = nn.LayerNorm(d), nn.Linear(d, ff), nn.Linear(ff, d)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        dh = d // self.h
        q, k, v = (t.view(B, T, self.h, dh).transpose(1, 2) for t in self.qkv(self.ln1(x)).split(d, dim=-1))
        a = (q @ k.transpose(-1, -2)) / (dh ** 0.5)
        a = a.masked_fill(~valid[:, None, None, :], float("-inf")).softmax(-1)
        x = x + self.out((a @ v).transpose(1, 2).reshape(B, T, d))
        return x + self.ff2(torch.relu(self.ff1(self.ln2(x))))


class TFEnc(nn.Module):
    """Ön-LN transformer kodlayıcı, öğrenilmiş konum, dolgu anahtarları maskeli; çıktı [ortalama havuz, son olay],
    boş dizi → sıfır (numpy `transformer_embed` ile aynı)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        d = int(cfg["d"])
        self.emb = nn.Linear(EVENT_WIDTH, d)
        self.pos = nn.Parameter(torch.randn(int(cfg["max_len"]), d) * 0.02)
        self.blocks = nn.ModuleList([TFBlock(d, int(cfg["heads"]), int(cfg["ff"])) for _ in range(int(cfg["layers"]))])
        self.lnf = nn.LayerNorm(d)

    def forward(self, toks: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        B, T, _ = toks.shape
        lc = lens.clamp(min=1)
        valid = torch.arange(T, device=toks.device)[None, :] < lc[:, None]
        x = self.emb(toks) + self.pos[:T]
        for blk in self.blocks:
            x = blk(x, valid)
        x = self.lnf(x)
        m = valid.unsqueeze(-1).to(x.dtype)
        mean = (x * m).sum(1) / lc[:, None].to(x.dtype)
        last = x[torch.arange(B, device=x.device), lc - 1]
        return torch.cat([mean, last], 1) * (lens > 0).unsqueeze(1).to(x.dtype)

    def names(self):
        S = {"emb.w": self.emb.weight, "emb.b": self.emb.bias, "pos": self.pos,
             "lnf.g": self.lnf.weight, "lnf.b": self.lnf.bias}
        for i, b in enumerate(self.blocks):
            for k, m in (("ln1", b.ln1), ("ln2", b.ln2)):
                S[f"l{i}.{k}.g"], S[f"l{i}.{k}.b"] = m.weight, m.bias
            for k, m in (("qkv", b.qkv), ("out", b.out), ("ff1", b.ff1), ("ff2", b.ff2)):
                S[f"l{i}.{k}.w"], S[f"l{i}.{k}.b"] = m.weight, m.bias
        return S


class QNet(nn.Module):
    """Dizi kodlayıcı + MLP: Q = MLP([F-vektörü, kodlayıcı(olaylar)])."""

    def __init__(self, level: str, hidden: Sequence[int], seq: dict):
        super().__init__()
        self.seq_cfg = dict(seq)
        self.enc = LSTMEnc(seq) if seq["kind"] == "lstm" else TFEnc(seq)
        self.mlp = _mlp(vec_dim(level) + seq_out_dim(seq), hidden)

    def forward(self, x, toks, lens):
        return self.mlp(torch.cat([x, self.enc(toks, lens)], 1))


def build_net(level: str = "F3", hidden: Sequence[int] = (512, 512, 512, 512), seq: Optional[dict] = None):
    if seq:
        return QNet(level, hidden, seq)
    return _mlp(vec_dim(level), hidden)


def _mlp_of(net):
    return net.mlp if isinstance(net, QNet) else net


def to_weights(net, meta: dict) -> Weights:
    lins = [m for m in _mlp_of(net) if isinstance(m, nn.Linear)]
    layers = [(m.weight.detach().cpu().numpy().copy(), m.bias.detach().cpu().numpy().copy()) for m in lins]
    if isinstance(net, QNet):
        seq = {k: v.detach().cpu().numpy().copy() for k, v in net.enc.names().items()}
        return Weights(layers, {**meta, "seq": net.seq_cfg}, seq)
    return Weights(layers, meta)


def load_into(net, w: Weights) -> None:
    lins = [m for m in _mlp_of(net) if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for m, (W, b) in zip(lins, w.layers):
            m.weight.copy_(torch.from_numpy(W))
            m.bias.copy_(torch.from_numpy(b))
        if isinstance(net, QNet):
            for k, p in net.enc.names().items():
                p.copy_(torch.from_numpy(w.seq[k]))


def q_loss(net: nn.Module, x: torch.Tensor, a: torch.Tensor, y: torch.Tensor, toks=None, lens=None) -> torch.Tensor:
    """Yalnız SEÇİLEN aksiyonun Q'su hedefe çekilir (diğer 56 çıkış bu örnekte eğitilmez)."""
    out = net(x) if toks is None else net(x, toks, lens)
    q = out.gather(1, a.view(-1, 1)).squeeze(1)
    return ((q - y) ** 2).mean()


def pad_events(Ls: np.ndarray, T: np.ndarray, device) -> torch.Tensor:
    """Art arda olaylar (ΣLs, 50) → dolgulu (n, max(1, max Ls), 50) uint8; i. örneğin olayları ilk Ls[i] satırda."""
    n = len(Ls)
    P = torch.zeros(n, max(1, int(Ls.max()) if n else 1), EVENT_WIDTH, dtype=torch.uint8, device=device)
    if T.shape[0]:
        rows = np.repeat(np.arange(n), Ls)
        starts = np.cumsum(Ls) - Ls
        cols = np.arange(int(Ls.sum())) - np.repeat(starts, Ls)
        P[torch.from_numpy(rows).to(device), torch.from_numpy(cols).to(device)] = torch.from_numpy(T).to(device)
    return P


def train_on(net, opt, X: np.ndarray, A: np.ndarray, Y: np.ndarray, batch: int, epochs: int, device,
             gen: torch.Generator, max_grad_norm: float = 40.0, Ls: Optional[np.ndarray] = None,
             T: Optional[np.ndarray] = None) -> float:
    n = len(Y)
    if n == 0:
        return float("nan")
    Xt = torch.from_numpy(X).to(device)
    At = torch.from_numpy(A).to(device)
    Yt = torch.from_numpy(Y).to(device)
    use_seq = isinstance(net, QNet)
    if use_seq:
        P = pad_events(Ls, T, device)
        Lt = torch.from_numpy(Ls).to(device)
    tot, cnt = 0.0, 0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).to(device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            if use_seq:
                lb = Lt[idx]
                tb = P[idx, :max(1, int(lb.max()))].to(torch.float32)
                loss = q_loss(net, Xt[idx], At[idx], Yt[idx], tb, lb)
            else:
                loss = q_loss(net, Xt[idx], At[idx], Yt[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()
            tot += loss.item() * len(idx)
            cnt += len(idx)
    return tot / cnt


DEFAULTS = dict(seed=0, level="F3", hidden=[512, 512, 512, 512], lr=1e-4, iters=100, rounds_per_iter=1800,
                epsilon=0.05, batch=256, epochs=1, workers=18, eval_every=10, eval_pairs=400,
                eval_opponents=["greedy"], device="auto", seq=None,
                mode="round", reward="L1", matches_per_iter=252, init=None, eval_level="L1")
# L2 ince ayarı (D-33): mode="match" → her iterasyonda `matches_per_iter` tam eğitim maçı (dmc.selfplay_match),
# reward ∈ {"L1" (kontrol), "win", "score", "met" (D-35, tur başı MET değişimi)}; init = başlangıç ağı (.npz), eval_level = izleme değerlendirmesi.


def train(cfg: dict, out_dir: str, log_fn=print, meta_extra: Optional[dict] = None) -> dict:
    """Eğitimi koşar. Kontrol noktaları `out_dir/it{N}.npz`, son ağ `out_dir/final.npz`, günlük `out_dir/log.json`."""
    from concurrent.futures import ProcessPoolExecutor
    from .arena import paired_parallel

    cfg = {**DEFAULTS, **cfg}
    dev = cfg["device"]
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    if cfg["mode"] not in ("round", "match") or cfg["reward"] not in REWARDS:
        raise ValueError("mode/reward geçersiz")
    if cfg["mode"] == "round" and cfg["reward"] != "L1":
        raise ValueError("tur self-play'i yalnız L1 ödülüyle")
    net = build_net(cfg["level"], cfg["hidden"], cfg["seq"]).to(device)
    if cfg["init"]:
        w0 = Weights.load(cfg["init"])
        if w0.level != cfg["level"] or list(w0.meta.get("hidden", [])) != list(cfg["hidden"])                 or (w0.seq_cfg or None) != (cfg["seq"] or None):
            raise ValueError("başlangıç ağı yapılandırmayla uyuşmuyor")
        load_into(net, w0)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    algo = "dmc-L1" if cfg["mode"] == "round" else f"dmc-match-{cfg['reward']}"
    meta = {"level": cfg["level"], "hidden": list(cfg["hidden"]), "algo": algo, "seed": cfg["seed"],
            **(meta_extra or {})}
    if cfg["mode"] == "match":
        meta["l1_match"] = "start"            # L1 değerlendirmesinde tur = maçın ilk turu (dmc.q_values)
    if cfg["init"]:
        meta["init"] = os.path.basename(cfg["init"])
    if cfg["reward"] == "met":                # tabloyu süreçler başlamadan bir kez çöz, diske yaz (D-35)
        import hashlib
        from . import dmc as _dmc
        from .met import load as met_load
        met_load(_dmc.MET_PATH)
        meta["met"] = os.path.basename(_dmc.MET_PATH)
        with open(_dmc.MET_PATH, "rb") as f:
            meta["met_sha256"] = hashlib.sha256(f.read()).hexdigest()[:16]
    if cfg["seq"]:
        meta["seq"] = dict(cfg["seq"])
    os.makedirs(out_dir, exist_ok=True)
    log = {"config": cfg, "device": str(device), "meta": meta, "iters": [], "evals": []}

    rounds_done = [0]

    def save(name, it):
        w = to_weights(net, {**meta, "iter": it, "rounds": rounds_done[0]})
        p = os.path.join(out_dir, f"{name}.npz")
        w.save(p)
        return p, w

    def evaluate(path, it):
        for opp in cfg["eval_opponents"]:
            t = time.time()
            lv = cfg["eval_level"]
            r = paired_parallel(lv, f"dmc:{path}", opp, cfg["eval_pairs"], seed="dmc-eval", workers=cfg["workers"])
            e = {"iter": it, "rounds": rounds_done[0], "opponent": opp, "level": lv, "pairs": r.n_pairs,
                 "diff": r.diff.mean if r.diff else None, "diff_ci95": list(r.diff.ci95) if r.diff else None,
                 "win": r.win.mean, "win_ci95": list(r.win.ci95), "sec": time.time() - t}
            log["evals"].append(e)
            head = (f"fark {e['diff']:+.2f} [{e['diff_ci95'][0]:+.2f}, {e['diff_ci95'][1]:+.2f}], " if r.diff else
                    f"[{e['win_ci95'][0]:.3f}, {e['win_ci95'][1]:.3f}] ")
            log_fn(f"  değerlendirme it {it}: vs {opp} {lv} {head}kazanma {e['win']:.3f} ({e['sec']:.0f} sn)")

    nw = cfg["workers"]
    match_mode = cfg["mode"] == "match"
    rpi = cfg["matches_per_iter"] if match_mode else cfg["rounds_per_iter"]
    bounds = [(k * rpi // nw, (k + 1) * rpi // nw) for k in range(nw)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for it in range(cfg["iters"]):
            if cfg["eval_every"] and it % cfg["eval_every"] == 0:
                p, _ = save(f"it{it}", it)
                evaluate(p, it)
            t = time.time()
            w = to_weights(net, meta)
            if match_mode:
                futs = [ex.submit(selfplay_match_chunk, w, f"{cfg['seed']}:{it}", s, e, cfg["epsilon"], cfg["reward"])
                        for s, e in bounds]
            else:
                futs = [ex.submit(selfplay_chunk, w, f"{cfg['seed']}:{it}", s, e, cfg["epsilon"]) for s, e in bounds]
            parts = [f.result() for f in futs]
            n_rounds = sum(p[5] for p in parts) if match_mode else rpi
            rounds_done[0] += n_rounds
            X = np.concatenate([p[0] for p in parts])
            A = np.concatenate([p[1] for p in parts])
            Y = np.concatenate([p[2] for p in parts])
            Ls = np.concatenate([p[3] for p in parts])
            T = np.concatenate([p[4] for p in parts])
            t_gen = time.time() - t
            net.train()
            loss = train_on(net, opt, X, A, Y, cfg["batch"], cfg["epochs"], device, gen, Ls=Ls, T=T)
            rec = {"iter": it, "samples": int(len(Y)), "rounds": n_rounds, "loss": loss, "y_mean": float(Y.mean()),
                   "gen_sec": t_gen, "train_sec": time.time() - t - t_gen, "elapsed": time.time() - t0}
            log["iters"].append(rec)
            log_fn(f"it {it:4d}: {rec['samples']} örnek, kayıp {loss:.4f}, üretim {t_gen:.1f} sn, "
                   f"eğitim {rec['train_sec']:.1f} sn")
            with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=1)
    p, _ = save("final", cfg["iters"])
    evaluate(p, cfg["iters"])
    log["final"] = p
    with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    return log
