"""Base de culturas e regras agronômicas de sucessão.

Parâmetros hídricos: coeficientes de cultura (Kc) da FAO-56 (Allen et al., 1998, Tabela 12) e
fatores de resposta da produtividade à água (Ky) da FAO-33 (Doorenbos & Kassam, 1979).
Valores marcados como "demonstrativos" (produtividade de referência, preço e custo) são ilustrativos
para o protótipo e devem ser substituídos por dados do produtor ou de referências regionais.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

VERAO = "verão"
SEGUNDA = "2ª safra / inverno"


@dataclass(frozen=True)
class Crop:
    key: str
    nome: str
    familia: str
    slot: str
    semeadura: str            # "MM-DD", referência hemisfério sul
    ciclo: int                # dias
    kc: tuple[float, float, float]
    ky: float
    raiz_m: float
    p: float                  # fração de esgotamento sem estresse (FAO-56)
    temp_calor: float         # Tmax (°C) acima da qual há estresse no período reprodutivo
    sensivel_geada: bool
    ph: tuple[float, float]   # faixa de pH (H2O) adequada
    residuo: float            # 0–1: aporte relativo de palhada
    raiz_profunda: bool       # contribui para descompactação biológica
    fixa_n: bool
    comercial: bool
    hosp_mofo_branco: bool    # Sclerotinia sclerotiorum
    hosp_nematoide_cisto: bool  # Heterodera glycines
    prod_ref: float = 0.0     # t/ha (demonstrativo)
    preco: float = 0.0        # R$/kg (demonstrativo)
    custo: float = 0.0        # R$/ha (demonstrativo)
    notas: tuple[str, ...] = field(default_factory=tuple)


CROPS: dict[str, Crop] = {c.key: c for c in [
    Crop("soja", "Soja", "Fabaceae", VERAO, "10-15", 115, (0.40, 1.15, 0.50), 0.85, 1.0, 0.50, 35, True,
         (5.5, 6.5), 0.30, False, True, True, True, True, 3.6, 2.00, 4500,
         ("Fixação biológica de N; palhada de rápida decomposição.",)),
    Crop("milho", "Milho (verão)", "Poaceae", VERAO, "10-15", 140, (0.30, 1.20, 0.35), 1.25, 1.2, 0.55, 35, True,
         (5.5, 6.5), 0.85, False, False, True, False, False, 9.0, 1.00, 6000,
         ("Grande aporte de palhada; ciclo longo limita a 2ª safra comercial.",)),
    Crop("feijao", "Feijão (das águas)", "Fabaceae", VERAO, "10-15", 90, (0.40, 1.15, 0.35), 1.15, 0.7, 0.45, 32, True,
         (5.5, 6.5), 0.20, False, True, True, True, False, 1.8, 4.00, 5200,
         ("Sensível a calor no florescimento e a déficit hídrico.",)),
    Crop("milho_2s", "Milho 2ª safra", "Poaceae", SEGUNDA, "02-20", 130, (0.30, 1.20, 0.35), 1.25, 1.2, 0.55, 35, True,
         (5.5, 6.5), 0.80, False, False, True, False, False, 6.0, 1.00, 4500,
         ("Depende das chuvas de fim de verão; risco de déficit no enchimento de grãos.",)),
    Crop("sorgo", "Sorgo granífero", "Poaceae", SEGUNDA, "02-20", 115, (0.30, 1.05, 0.55), 0.90, 1.4, 0.55, 40, True,
         (5.5, 6.5), 0.70, True, False, True, False, False, 4.0, 0.85, 2800,
         ("Maior tolerância à seca que o milho.",)),
    Crop("feijao_2s", "Feijão 2ª safra", "Fabaceae", SEGUNDA, "02-20", 90, (0.40, 1.15, 0.35), 1.15, 0.7, 0.45, 32, True,
         (5.5, 6.5), 0.20, False, True, True, True, False, 1.6, 4.00, 4800, ()),
    Crop("girassol", "Girassol", "Asteraceae", SEGUNDA, "02-20", 110, (0.35, 1.10, 0.35), 0.95, 1.6, 0.45, 38, True,
         (5.2, 6.5), 0.40, True, False, True, True, False, 1.6, 2.20, 2500,
         ("Raiz pivotante profunda; hospedeiro de mofo-branco.",)),
    Crop("trigo", "Trigo", "Poaceae", SEGUNDA, "05-01", 120, (0.30, 1.15, 0.30), 1.15, 1.0, 0.55, 30, False,
         (5.5, 6.5), 0.60, False, False, True, False, False, 3.0, 1.40, 3800,
         ("Cultura de inverno; geada no espigamento pode causar perdas.",)),
    Crop("aveia", "Aveia-preta (cobertura)", "Poaceae", SEGUNDA, "04-20", 110, (0.30, 1.10, 0.30), 1.00, 0.9, 0.55, 32, False,
         (5.0, 6.5), 0.80, False, False, False, False, False, 0.0, 0.0, 600,
         ("Cobertura de inverno; supressão de plantas daninhas.",)),
    Crop("braquiaria", "Braquiária (cobertura)", "Poaceae", SEGUNDA, "02-20", 200, (0.40, 0.95, 0.85), 0.80, 1.5, 0.60, 42, True,
         (4.5, 6.5), 0.95, True, False, False, False, False, 0.0, 0.0, 400,
         ("Palhada persistente para plantio direto; tolerante à seca.",)),
    Crop("crotalaria", "Crotalária (cobertura)", "Fabaceae", SEGUNDA, "02-20", 120, (0.40, 1.00, 0.80), 0.80, 1.2, 0.55, 40, True,
         (5.0, 6.5), 0.60, True, True, False, False, False, 0.0, 0.0, 500,
         ("Espécies como C. spectabilis/ochroleuca são usadas no manejo de nematoides de galha e das lesões.",)),
    Crop("nabo", "Nabo forrageiro (cobertura)", "Brassicaceae", SEGUNDA, "04-15", 100, (0.35, 1.00, 0.80), 0.80, 1.5, 0.55, 32, False,
         (5.0, 6.5), 0.45, True, False, False, True, False, 0.0, 0.0, 350,
         ("Raiz pivotante agressiva; hospedeiro de mofo-branco.",)),
    Crop("pousio", "Pousio", "—", SEGUNDA, "03-01", 150, (0.30, 0.30, 0.30), 0.0, 0.3, 0.50, 99, False,
         (0.0, 14.0), 0.0, False, False, False, False, False, 0.0, 0.0, 150,
         ("Solo exposto ou com plantas espontâneas; custo de dessecação/manejo.",)),
]}

SUMMER = [k for k, c in CROPS.items() if c.slot == VERAO]
SECOND = [k for k, c in CROPS.items() if c.slot == SEGUNDA]

AWC_MM_PER_M = {"arenosa": 70.0, "média": 130.0, "argilosa": 165.0}
"""Água disponível total aproximada por textura (mm por m de solo), faixa típica FAO-56 Tabela 19."""

TEXTURES = list(AWC_MM_PER_M)


def crops_table() -> pd.DataFrame:
    rows = []
    for c in CROPS.values():
        rows.append({
            "key": c.key, "Cultura": c.nome, "Família": c.familia, "Época": c.slot,
            "Semeadura ref.": c.semeadura, "Ciclo (d)": c.ciclo, "Kc ini/méd/fim": "/".join(f"{x:.2f}" for x in c.kc),
            "Ky": c.ky, "Raiz (m)": c.raiz_m, "Tmax estresse (°C)": c.temp_calor,
            "Sensível a geada": c.sensivel_geada, "pH": f"{c.ph[0]}–{c.ph[1]}",
            "Produtividade ref. (t/ha)": c.prod_ref, "Preço (R$/kg)": c.preco, "Custo (R$/ha)": c.custo,
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class RuleHit:
    efeito: int        # pontos no critério fitossanitário/solo (positivo = benefício)
    criterio: str      # "fito" ou "solo"
    texto: str


def transition_rules(prev: Crop, nxt: Crop) -> list[RuleHit]:
    """Regras agronômicas verificáveis para a sucessão prev -> nxt."""
    hits: list[RuleHit] = []
    if prev.key == "pousio" or nxt.key == "pousio":
        if nxt.key == "pousio":
            hits.append(RuleHit(-12, "solo", f"Pousio após {prev.nome.lower()}: solo sem cobertura ativa na entressafra, maior risco de erosão e de plantas daninhas."))
        return hits
    base_prev = prev.key.split("_")[0]
    base_next = nxt.key.split("_")[0]
    if base_prev == base_next:
        hits.append(RuleHit(-25, "fito", f"{nxt.nome} após {prev.nome.lower()}: mesma espécie em sequência favorece pragas, doenças e plantas daninhas associadas."))
    elif prev.familia == nxt.familia:
        hits.append(RuleHit(-10, "fito", f"{prev.nome} → {nxt.nome.lower()}: mesma família botânica ({prev.familia}) em sequência reduz a quebra de ciclo de patógenos."))
    if prev.hosp_mofo_branco and nxt.hosp_mofo_branco:
        hits.append(RuleHit(-15, "fito", f"{prev.nome} → {nxt.nome.lower()}: ambas são hospedeiras do mofo-branco (Sclerotinia sclerotiorum), favorecendo o acúmulo de escleródios no solo."))
    if prev.hosp_mofo_branco and nxt.familia == "Poaceae" and nxt.residuo >= 0.7:
        hits.append(RuleHit(8, "fito", f"Palhada de {nxt.nome.lower()} após {prev.nome.lower()}: gramíneas não são hospedeiras do mofo-branco e a cobertura dificulta a germinação dos escleródios."))
    if prev.hosp_nematoide_cisto and not nxt.hosp_nematoide_cisto and nxt.familia == "Poaceae":
        hits.append(RuleHit(5, "fito", f"{nxt.nome} após soja: gramínea não hospedeira do nematoide-de-cisto (Heterodera glycines) ajuda a quebrar o ciclo."))
    if nxt.key == "crotalaria":
        hits.append(RuleHit(8, "fito", "Crotalária: utilizada no manejo de nematoides de galha e das lesões (confirmar espécie e população com análise nematológica)."))
    if base_prev == "milho" and nxt.key == "trigo":
        hits.append(RuleHit(-10, "fito", "Trigo após milho: restos culturais do milho podem aumentar o inóculo de Fusarium graminearum (giberela)."))
    if prev.fixa_n and nxt.familia == "Poaceae":
        hits.append(RuleHit(6, "solo", f"{prev.nome} → {nxt.nome.lower()}: leguminosa antes de gramínea favorece o aproveitamento do N residual."))
    if nxt.raiz_profunda:
        hits.append(RuleHit(5, "solo", f"{nxt.nome}: sistema radicular profundo contribui para a estruturação do solo e ciclagem de nutrientes."))
    return hits
