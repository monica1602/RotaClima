"""NDVI MODIS (MOD13Q1, 250 m, composições de 16 dias) via ORNL DAAC MODIS Web Service.

Documentação: https://modis.ornl.gov/data/modis_webservice.html
O serviço aceita no máximo 10 datas por requisição; por isso a série é baixada em blocos.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://modis.ornl.gov/rst/api/v1/MOD13Q1"
BAND = "250m_16_days_NDVI"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
HEADERS = {"Accept": "application/json"}


def _cache_path(lat: float, lon: float) -> Path:
    return CACHE_DIR / f"modis_ndvi_{lat:.4f}_{lon:.4f}.csv"


def available_dates(lat: float, lon: float, timeout: int = 60) -> list[str]:
    r = requests.get(f"{BASE}/dates", params={"latitude": lat, "longitude": lon}, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return [d["modis_date"] for d in r.json()["dates"]]


def fetch_ndvi_point(lat: float, lon: float, n_composites: int = 23, use_cache: bool = True,
                     timeout: int = 120) -> pd.DataFrame:
    """Série de NDVI do pixel MODIS que contém o ponto (últimas n composições, ~1 ano)."""
    path = _cache_path(lat, lon)
    if use_cache and path.exists():
        return pd.read_csv(path, parse_dates=["data"])
    dates = available_dates(lat, lon)[-n_composites:]
    rows = []
    for i in range(0, len(dates), 10):
        chunk = dates[i:i + 10]
        params = {
            "latitude": lat, "longitude": lon, "band": BAND,
            "startDate": chunk[0], "endDate": chunk[-1],
            "kmAboveBelow": 0, "kmLeftRight": 0,
        }
        r = requests.get(f"{BASE}/subset", params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        for item in r.json().get("subset", []):
            value = item["data"][0]
            ndvi = np.nan if value <= -3000 else value * 0.0001
            rows.append({"data": pd.to_datetime(item["calendar_date"]), "ndvi": ndvi})
    df = pd.DataFrame(rows).drop_duplicates("data").sort_values("data")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def cached_ndvi(lat: float, lon: float) -> pd.DataFrame | None:
    path = _cache_path(lat, lon)
    return pd.read_csv(path, parse_dates=["data"]) if path.exists() else None
