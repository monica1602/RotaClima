import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rotaclima import nasa_power, presets  # noqa: E402

DEMO = "Rio Verde (GO) — Cerrado"


@pytest.fixture(scope="session")
def demo_climate():
    p = presets.PRESETS[DEMO]
    try:
        df, meta = nasa_power.prepare(p["lat"], p["lon"], use_cache=True)
    except Exception as exc:  # sem cache e sem rede
        pytest.skip(f"Dados NASA POWER indisponíveis: {exc}")
    return p["lat"], p["lon"], df, meta
