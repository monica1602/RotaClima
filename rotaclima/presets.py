"""Propriedades demonstrativas (dados de talhões fictícios; clima e NDVI reais da NASA)."""
from __future__ import annotations

import math

TALHAO_COLUMNS = ["nome", "lat", "lon", "area_ha", "textura", "ph", "mo_pct", "irrigado",
                  "verao_1", "segunda_1", "verao_2", "segunda_2"]

PRESETS = {
    "Rio Verde (GO) — Cerrado": {
        "lat": -17.78, "lon": -51.05,
        "descricao": "Cerrado goiano: verão chuvoso e inverno seco; sucessão soja/milho 2ª safra predominante.",
        "talhoes": [
            ("Talhão 1 — Sede", 0.000, 0.000, 60, "argilosa", 5.8, 3.0, False, "soja", "milho_2s", "soja", "milho_2s"),
            ("Talhão 2 — Baixada", -0.010, 0.012, 45, "média", 5.2, 2.0, False, "soja", "pousio", "soja", "pousio"),
            ("Talhão 3 — Chapada arenosa", 0.011, 0.010, 70, "arenosa", 5.0, 1.2, False, "soja", "milho_2s", "soja", "pousio"),
            ("Talhão 4 — Pivô", 0.009, -0.012, 40, "argilosa", 6.0, 3.2, True, "milho", "trigo", "soja", "feijao_2s"),
        ],
    },
    "Londrina (PR) — Norte do Paraná": {
        "lat": -23.25, "lon": -51.05,
        "descricao": "Transição subtropical: chuvas bem distribuídas e risco de geada no inverno.",
        "talhoes": [
            ("Talhão 1", 0.000, 0.000, 50, "argilosa", 5.6, 3.1, False, "soja", "milho_2s", "soja", "milho_2s"),
            ("Talhão 2", -0.010, 0.011, 35, "argilosa", 5.4, 2.6, False, "soja", "trigo", "soja", "aveia"),
            ("Talhão 3", 0.010, 0.010, 55, "média", 5.1, 1.8, False, "soja", "pousio", "soja", "milho_2s"),
            ("Talhão 4", 0.009, -0.011, 30, "argilosa", 6.0, 3.4, False, "milho", "aveia", "soja", "trigo"),
        ],
    },
    "Passo Fundo (RS) — Planalto Médio": {
        "lat": -28.20, "lon": -52.45,
        "descricao": "Clima subtropical úmido: soja no verão e cereais de inverno (trigo, aveia).",
        "talhoes": [
            ("Talhão 1", 0.000, 0.000, 80, "argilosa", 5.7, 3.5, False, "soja", "trigo", "soja", "trigo"),
            ("Talhão 2", -0.011, 0.010, 40, "argilosa", 5.3, 2.8, False, "soja", "aveia", "soja", "aveia"),
            ("Talhão 3", 0.010, 0.012, 60, "média", 5.0, 2.2, False, "milho", "pousio", "soja", "pousio"),
            ("Talhão 4", 0.010, -0.010, 35, "argilosa", 6.1, 3.8, False, "soja", "nabo", "milho", "aveia"),
        ],
    },
    "Sorriso (MT) — Médio-Norte": {
        "lat": -12.60, "lon": -55.85,
        "descricao": "Cerrado/Amazônia: estação chuvosa longa e seca intensa de maio a setembro.",
        "talhoes": [
            ("Talhão 1", 0.000, 0.000, 120, "argilosa", 5.6, 2.8, False, "soja", "milho_2s", "soja", "milho_2s"),
            ("Talhão 2", -0.012, 0.012, 90, "média", 5.2, 2.0, False, "soja", "milho_2s", "soja", "milho_2s"),
            ("Talhão 3", 0.012, 0.012, 100, "arenosa", 5.0, 1.1, False, "soja", "pousio", "soja", "pousio"),
            ("Talhão 4", 0.012, -0.012, 80, "argilosa", 5.9, 3.0, False, "soja", "braquiaria", "soja", "milho_2s"),
        ],
    },
    "Petrolina (PE) — Semiárido": {
        "lat": -9.15, "lon": -40.40,
        "descricao": "Semiárido: chuvas concentradas e irregulares; agricultura de sequeiro de alto risco.",
        "talhoes": [
            ("Talhão 1 — Sequeiro", 0.000, 0.000, 20, "arenosa", 6.0, 0.9, False, "feijao", "pousio", "milho", "pousio"),
            ("Talhão 2 — Sequeiro", -0.008, 0.009, 15, "média", 6.2, 1.2, False, "milho", "pousio", "feijao", "pousio"),
            ("Talhão 3 — Irrigado", 0.008, 0.009, 12, "média", 6.5, 1.5, True, "feijao", "milho_2s", "milho", "feijao_2s"),
            ("Talhão 4 — Irrigado", 0.008, -0.008, 10, "arenosa", 6.3, 1.0, True, "milho", "sorgo", "feijao", "pousio"),
        ],
    },
}


def talhoes_records(preset_name: str) -> list[dict]:
    p = PRESETS[preset_name]
    out = []
    for t in p["talhoes"]:
        nome, dlat, dlon, area, tex, ph, mo, irr, v1, s1, v2, s2 = t
        out.append(dict(zip(TALHAO_COLUMNS, [nome, round(p["lat"] + dlat, 4), round(p["lon"] + dlon, 4), area, tex,
                                              ph, mo, irr, v1, s1, v2, s2])))
    return out


def square_polygon(lat: float, lon: float, area_ha: float) -> list[tuple[float, float]]:
    """Quadrado aproximado com a área do talhão, centrado no ponto (para visualização)."""
    side_m = math.sqrt(max(area_ha, 0.1) * 10_000)
    dlat = side_m / 2 / 111_320
    dlon = side_m / 2 / (111_320 * math.cos(math.radians(lat)))
    return [(lat - dlat, lon - dlon), (lat - dlat, lon + dlon), (lat + dlat, lon + dlon),
            (lat + dlat, lon - dlon), (lat - dlat, lon - dlon)]
