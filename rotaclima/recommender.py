"""Geração, avaliação multicritério e explicação de rotações de culturas (2 anos, 4 épocas)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .crops import AWC_MM_PER_M, CROPS, SECOND, SUMMER, Crop, RuleHit, transition_rules
from .engine import LOSS_THRESHOLD, CropRunner, second_season_delay

CRITERIA = {
    "agua": "Água",
    "solo": "Solo",
    "fito": "Fitossanidade",
    "risco": "Risco climático",
    "econ": "Economia",
}
PRESETS = {
    "A — Conservação da água": {"agua": 0.40, "risco": 0.25, "solo": 0.15, "fito": 0.10, "econ": 0.10},
    "B — Recuperação do solo": {"solo": 0.40, "fito": 0.20, "agua": 0.15, "risco": 0.10, "econ": 0.15},
    "C — Equilíbrio econômico e ambiental": {"econ": 0.30, "risco": 0.20, "solo": 0.20, "agua": 0.15, "fito": 0.15},
}
COVER_MIN_SAT = 0.6
"""Água atendida mínima para considerar que uma cobertura se estabelece plenamente (heurístico)."""
IRRIGATION_COST = 1.5
"""Custo variável da irrigação (R$ por mm por ha), demonstrativo."""
SLOT_LABELS = ["Verão ano 1", "2ª safra/inverno ano 1", "Verão ano 2", "2ª safra/inverno ano 2"]


@dataclass
class Talhao:
    nome: str
    area_ha: float
    textura: str
    ph: float
    mo_pct: float
    irrigado: bool
    rotacao_atual: tuple[str, str, str, str]
    lat: float = 0.0
    lon: float = 0.0

    @property
    def awc(self) -> float:
        return AWC_MM_PER_M.get(self.textura, 130.0)


@dataclass
class Economics:
    prod_ref: dict[str, float] = field(default_factory=lambda: {k: c.prod_ref for k, c in CROPS.items()})
    preco: dict[str, float] = field(default_factory=lambda: {k: c.preco for k, c in CROPS.items()})
    custo: dict[str, float] = field(default_factory=lambda: {k: c.custo for k, c in CROPS.items()})


def rotation_label(rot: tuple[str, ...]) -> str:
    return " → ".join(CROPS[k].nome for k in rot)


def rotation_short(rot: tuple[str, ...]) -> str:
    return " → ".join(CROPS[k].nome.split(" (")[0] for k in rot)


def enumerate_rotations(allowed: set[str], summer_shift: int = 0) -> list[tuple[tuple[str, ...], tuple[int, int]]]:
    """Rotações viáveis em calendário: (rotação, (atraso 2ª safra ano 1, atraso ano 2))."""
    summers = [k for k in SUMMER if k in allowed]
    seconds = [k for k in SECOND if k in allowed or k == "pousio"]
    pairs = []
    for s1 in summers:
        for s2 in seconds:
            d = second_season_delay(CROPS[s1], CROPS[s2], summer_shift)
            if d is not None:
                pairs.append((s1, s2, d))
    rots = []
    for (a1, a2, d1), (b1, b2, d2) in itertools.product(pairs, pairs):
        rots.append(((a1, a2, b1, b2), (d1, d2)))
    return rots


def delays_for(rot: tuple[str, ...], summer_shift: int = 0) -> tuple[int, int] | None:
    d1 = second_season_delay(CROPS[rot[0]], CROPS[rot[1]], summer_shift)
    d2 = second_season_delay(CROPS[rot[2]], CROPS[rot[3]], summer_shift)
    if d1 is None or d2 is None:
        return None
    return d1, d2


def _ph_factor(crop: Crop, ph: float) -> float:
    lo, hi = crop.ph
    return 1.0 if lo <= ph <= hi else 0.85


_ARRAY_COLS = ["etc", "eta", "deficit", "sat", "irrig", "rel_yield"]


def _slot_arrays(runner: CropRunner, key: str, delay: int, years: np.ndarray, second_year: bool) -> dict:
    cache = runner.__dict__.setdefault("_arrays", {})
    ck = (key, delay, second_year)
    if ck not in cache:
        r = runner.runs(key, delay).reindex(years + 1 if second_year else years)
        arr = {c: r[c].fillna(0.0).to_numpy() for c in _ARRAY_COLS}
        arr["valid"] = r["etc"].notna().to_numpy()
        cache[ck] = arr
    return cache[ck]


def rotation_samples(rot: tuple[str, ...], delays: tuple[int, int], runner: CropRunner, talhao: Talhao,
                     eco: Economics) -> pd.DataFrame:
    """Resultados por ano histórico: ano 1 usa o clima do ano Y e o ano 2, o de Y+1."""
    years = np.array([y for y in runner.years if y + 1 in runner.years])
    n = len(years)
    data = {c: np.zeros(n) for c in ["margem", "etc", "eta", "irrig", "deficit_com", "etc_com", "quebras", "n_com"]}
    valid = np.ones(n, dtype=bool)
    for i, key in enumerate(rot):
        delay = delays[0] if i == 1 else delays[1] if i == 3 else 0
        crop = CROPS[key]
        a = _slot_arrays(runner, key, delay, years, i >= 2)
        valid &= a["valid"]
        rel = a["rel_yield"] * _ph_factor(crop, talhao.ph)
        prod = eco.prod_ref.get(key, crop.prod_ref) * rel
        margem = prod * 1000 * eco.preco.get(key, crop.preco) - eco.custo.get(key, crop.custo) - a["irrig"] * IRRIGATION_COST
        data["margem"] = data["margem"] + margem
        data[f"margem_{i}"] = margem
        data[f"prod_{i}"] = prod
        data[f"sat_{i}"] = a["sat"]
        data[f"rel_{i}"] = rel
        data["etc"] = data["etc"] + a["etc"]
        data["eta"] = data["eta"] + a["eta"]
        data["irrig"] = data["irrig"] + a["irrig"]
        if crop.comercial:
            data["n_com"] = data["n_com"] + 1
            data["deficit_com"] = data["deficit_com"] + a["deficit"]
            data["etc_com"] = data["etc_com"] + a["etc"]
            data["quebras"] = data["quebras"] + (rel < 1 - LOSS_THRESHOLD)
    out = pd.DataFrame(data, index=pd.Index(years, name="ano"))
    return out[valid]


def _soil_components(rot: tuple[str, ...], establishment: list[float]) -> dict:
    """Componentes de solo; `establishment` (0–1) reduz a contribuição de coberturas com pouca água."""
    crops = [CROPS[k] for k in rot]
    cover = sum(min(c.ciclo + c.residuo * 90, 365) * e for c, e in zip(crops, establishment) if c.key != "pousio")
    families = {c.familia for c in crops if c.familia != "—"}
    return {
        "cobertura_dias": min(cover, 730),
        "residuo_medio": float(np.mean([c.residuo * e for c, e in zip(crops, establishment)])),
        "familias": len(families),
        "leguminosa": any(c.fixa_n for c in crops),
        "raiz_profunda": any(c.raiz_profunda for c in crops),
    }


def rule_hits(rot: tuple[str, ...]) -> list[RuleHit]:
    hits: list[RuleHit] = []
    seq = list(rot) + [rot[0]]
    seen = set()
    for a, b in zip(seq[:-1], seq[1:]):
        for h in transition_rules(CROPS[a], CROPS[b]):
            if h.texto not in seen:
                seen.add(h.texto)
                hits.append(h)
    if rot[0] == rot[2]:
        hits.append(RuleHit(-10, "fito", f"{CROPS[rot[0]].nome} nos dois verões: sem rotação na safra principal (sucessão repetida)."))
    return hits


def evaluate(talhao: Talhao, runner: CropRunner, eco: Economics, allowed: set[str],
             summer_shift: int = 0, include: list[tuple[str, ...]] | None = None,
             only_include: bool = False) -> pd.DataFrame:
    """Calcula os indicadores (0–100) de todas as rotações viáveis. Use `rank` para aplicar pesos."""
    candidates = [] if only_include else enumerate_rotations(allowed, summer_shift)
    keys = {c[0] for c in candidates}
    for rot in include or []:
        if rot not in keys:
            d = delays_for(rot, summer_shift)
            if d is not None:
                candidates.append((rot, d))
                keys.add(rot)
    records = []
    for rot, delays in candidates:
        s = rotation_samples(rot, delays, runner, talhao, eco)
        if s.empty:
            continue
        establishment = []
        for i, key in enumerate(rot):
            c = CROPS[key]
            establishment.append(1.0 if c.comercial or key == "pousio"
                                 else float(np.clip(s[f"sat_{i}"].mean() / COVER_MIN_SAT, 0, 1)))
        soil = _soil_components(rot, establishment)
        hits = rule_hits(rot)
        fito = 80 + sum(h.efeito for h in hits if h.criterio == "fito")
        solo_rules = sum(h.efeito for h in hits if h.criterio == "solo")
        solo = (35 * soil["cobertura_dias"] / 730 + 20 * soil["residuo_medio"] + 20 * (soil["familias"] - 1) / 3
                + 5 * soil["leguminosa"] + 5 * soil["raiz_profunda"] + solo_rules)
        deficit_ratio = (s["deficit_com"] / s["etc_com"].replace(0, np.nan)).mean()
        n_com = s["n_com"].iloc[0]
        p_quebra = (s["quebras"] / max(n_com, 1)).mean()
        records.append({
            "rot": rot, "delays": delays, "Rotação": rotation_short(rot),
            "margem_media": s["margem"].mean(), "margem_p10": s["margem"].quantile(0.10),
            "margem_p90": s["margem"].quantile(0.90),
            "demanda_mm": s["etc"].mean(), "irrig_mm": s["irrig"].mean(),
            "deficit_pct": 100 * deficit_ratio, "p_quebra": 100 * p_quebra,
            "anos_quebra": int((s["quebras"] > 0).sum()), "n_anos": len(s),
            "cobertura_dias": soil["cobertura_dias"], "familias": soil["familias"],
            "fito": float(np.clip(fito, 0, 100)), "solo": float(np.clip(solo, 0, 100)),
            "risco": float(100 * (1 - p_quebra)),
            "hits": hits, "samples": s, "estabelecimento": establishment,
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    water_use = df["demanda_mm"] + df["irrig_mm"]
    span = water_use.max() - water_use.min() or 1.0
    df["agua"] = (100 - 0.6 * df["deficit_pct"].clip(0, 100) - 40 * (water_use - water_use.min()) / span).clip(0, 100)
    eco_val = 0.7 * df["margem_media"] + 0.3 * df["margem_p10"]
    span_e = eco_val.max() - eco_val.min() or 1.0
    df["econ"] = 100 * (eco_val - eco_val.min()) / span_e
    return df


def rank(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Pontuação ponderada (0–100) e ordenação."""
    if df.empty:
        return df
    total_w = sum(weights.values()) or 1.0
    out = df.copy()
    out["pontuacao"] = sum(out[k] * w for k, w in weights.items()) / total_w
    return out.sort_values("pontuacao", ascending=False).reset_index(drop=True)


def system_key(rot: tuple[str, ...]) -> tuple[str, ...]:
    """Identifica o mesmo sistema de rotação independentemente da fase (ano 1/ano 2 trocados)."""
    return min(tuple(rot), tuple(rot[2:]) + tuple(rot[:2]))


def top_unique(ranked: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Melhores rotações sem repetir o mesmo sistema em fase trocada."""
    if ranked.empty:
        return ranked
    keys = ranked["rot"].map(system_key)
    return ranked[~keys.duplicated()].head(n).reset_index(drop=True)


def explain(row: pd.Series, talhao: Talhao, reference: pd.Series | None = None, max_items: int = 4) -> dict:
    """Justificativas em linguagem simples: pontos fortes, riscos e comparação com a rotação atual."""
    rot = row["rot"]
    s: pd.DataFrame = row["samples"]
    pros, cons = [], []
    hits = sorted(row["hits"], key=lambda h: -abs(h.efeito))
    pros += [h.texto for h in hits if h.efeito > 0][:max_items]
    cons += [h.texto for h in hits if h.efeito < 0][:max_items]
    n = row["n_anos"]
    for i, key in enumerate(rot):
        crop = CROPS[key]
        if not crop.comercial:
            est = row["estabelecimento"][i]
            if key != "pousio" and est < 0.8:
                cons.append(f"{crop.nome} ({SLOT_LABELS[i].lower()}): pouca água no período "
                            f"(atendimento médio {s[f'sat_{i}'].mean():.0%}); estabelecimento e produção de palhada incertos.")
            continue
        sat = s[f"sat_{i}"].mean()
        bad = int((s[f"rel_{i}"] < 1 - LOSS_THRESHOLD).sum())
        label = f"{crop.nome} ({SLOT_LABELS[i].lower()})"
        if sat >= 0.9 and bad == 0:
            pros.append(f"{label}: a chuva histórica atendeu em média {sat:.0%} da demanda de água da cultura.")
        elif bad > 0:
            cons.append(f"{label}: em {bad} de {n} anos históricos a perda estimada passaria de {LOSS_THRESHOLD:.0%} "
                        f"(água atendida em média: {sat:.0%}).")
        lo, hi = crop.ph
        if not lo <= talhao.ph <= hi:
            cons.append(f"{crop.nome}: pH {talhao.ph:.1f} fora da faixa {lo}–{hi}; avaliar calagem com base na análise de solo.")
    if row["familias"] >= 3:
        pros.append(f"Diversifica {row['familias']} famílias botânicas em dois anos.")
    if row["cobertura_dias"] >= 600:
        pros.append(f"Mantém o solo coberto (cultura ou palhada) em cerca de {row['cobertura_dias']:.0f} de 730 dias.")
    elif row["cobertura_dias"] < 450:
        cons.append(f"Solo coberto em apenas ~{row['cobertura_dias']:.0f} de 730 dias.")
    if talhao.irrigado and row["irrig_mm"] > 0:
        cons.append(f"Usa em média {row['irrig_mm']:.0f} mm de irrigação em dois anos.")
    if talhao.mo_pct < 2.0 or talhao.textura == "arenosa":
        pros_or_cons = pros if row["solo"] >= 60 else cons
        pros_or_cons.append(f"Talhão vulnerável (textura {talhao.textura}, MO {talhao.mo_pct:.1f}%): "
                            f"{'a rotação favorece' if row['solo'] >= 60 else 'a rotação pouco contribui para'} cobertura e aporte de palhada.")
    comp = []
    if reference is not None and reference["rot"] != rot:
        dm = row["margem_media"] - reference["margem_media"]
        dw = (row["demanda_mm"] + row["irrig_mm"]) - (reference["demanda_mm"] + reference["irrig_mm"])
        dc = row["cobertura_dias"] - reference["cobertura_dias"]
        dq = row["p_quebra"] - reference["p_quebra"]
        comp.append(f"Margem média {'+' if dm >= 0 else '−'}R$ {abs(dm):,.0f}/ha em 2 anos".replace(",", "."))
        comp.append(f"Consumo de água {dw:+.0f} mm")
        comp.append(f"Cobertura do solo {dc:+.0f} dias")
        comp.append(f"Probabilidade de quebra por cultura comercial {dq:+.0f} p.p.")
    return {"pros": pros, "cons": cons, "comparacao": comp}
