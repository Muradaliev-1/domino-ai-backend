"""
Baseline ajanlar (Faz 2). Sızıntısız: bir ajan YALNIZ kendi Observation'ını, yasal hamle listesini ve public maç
durumunu görür. Yasal hamleler kendi el + uçlar + (public) zorunlu açılıştan türer, gizli bilgi içermez.

Arayüz:  agent.act(obs, legal, match=None) -> Move   (legal'in bir elemanı olmalı)
Ajanlar bir oyun boyunca durum tutabilir; her oyun için yeni ajan oluşturulur (arena.py, make_agent).
"""
from __future__ import annotations

import random
from typing import Callable, Dict, Optional, Sequence

from .engine import Move, Observation


class Agent:
    name = "agent"

    def act(self, obs: Observation, legal: Sequence[Move], match: Optional[dict] = None) -> Move:
        raise NotImplementedError


class RandomAgent(Agent):
    """Yasal hamleler arasından tekdüze rastgele. Tohumla yeniden üretilebilir."""
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, obs, legal, match=None):
        return legal[self.rng.randrange(len(legal))]


def pip_key(m: Move):
    """Açgözlü sıralama anahtarı: önce pip toplamı, eşitlikte taş sırası, sonra 'L'. Tamamen deterministik."""
    if m.is_pass:
        return (-1, (-1, -1), False)
    return (m.tile[0] + m.tile[1], m.tile, m.side == "L")


class GreedyAgent(Agent):
    """En yüksek pip toplamlı taşı oynar (elde kalırsa en çok puan yazdıracak taştan kurtulur, 05 Mimari).
    Strateji kodlamaz (D-22): yalnız 'büyük taşı at' tabanı."""
    name = "greedy"

    def __init__(self, seed=0):
        pass

    def act(self, obs, legal, match=None):
        return max(legal, key=pip_key)


AGENTS: Dict[str, Callable[..., Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
}


def make_agent(name: str, seed) -> Agent:
    """'oracle' ya da 'oracle@N' (N iterasyon): TAM BİLGİLİ referans (oracle.py), yalnız üst sınır için."""
    if name == "oracle" or name.startswith("oracle@"):
        from .oracle import OracleMCTS
        return OracleMCTS(seed, iterations=int(name.split("@")[1])) if "@" in name else OracleMCTS(seed)
    # gizli bilgisiz arama (search.py): 'pimc', 'pimc@N', 'ismcts', 'ismcts@N' (N: hamle başı toplam iterasyon);
    # ağırlıklı determinizasyon (D-25): 'ismcts-rand@N', 'ismcts-pip@N', 'pimc-rand@N', 'pimc-pip@N'
    base = name.split("@")[0]
    for prefix in ("pimc", "ismcts"):
        if base == prefix or base.startswith(prefix + "-"):
            from .search import ISMCTSAgent, PIMCAgent, PLAYER_MODELS
            cls = PIMCAgent if prefix == "pimc" else ISMCTSAgent
            model = base[len(prefix) + 1:] or None
            if model is not None and model not in PLAYER_MODELS:
                raise ValueError(f"bilinmeyen oyuncu modeli: {model} (seçenekler: {', '.join(PLAYER_MODELS)})")
            kw = {"model": model} if model else {}
            return cls(seed, iterations=int(name.split("@")[1]), **kw) if "@" in name else cls(seed, **kw)
    if name not in AGENTS:
        raise ValueError(f"bilinmeyen ajan: {name} (seçenekler: {', '.join(AGENTS)})")
    return AGENTS[name](seed)
