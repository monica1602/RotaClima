"""Acesso à API NASA POWER (dados diários, comunidade AG) com cache em disco.

Documentação oficial: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
Resolução espacial nativa: MERRA-2 0,5° x 0,625°; CERES (radiação) 1° x 1°.
"""
from __future__ import annotations

import datetime as dt
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = [
    "T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M",
    "ALLSKY_SFC_SW_DWN", "WS2M", "GWETROOT",
]
PARAM_LABELS = {
    "T2M": "Temperatura média do ar a 2 m (°C)",
    "T2M_MAX": "Temperatura máxima a 2 m (°C)",
    "T2M_MIN": "Temperatura mínima a 2 m (°C)",
    "PRECTOTCORR": "Precipitação corrigida (mm/dia)",
    "RH2M": "Umidade relativa a 2 m (%)",
    "ALLSKY_SFC_SW_DWN": "Radiação solar global (MJ/m²/dia)",
    "WS2M": "Velocidade do vento a 2 m (m/s)",
    "GWETROOT": "Umidade do solo na zona radicular (0–1, MERRA-2)",
}
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
START_DATE = dt.date(2001, 1, 1)


def _cache_path(lat: float, lon: float) -> Path:
    return CACHE_DIR / f"power_{lat:.2f}_{lon:.2f}.csv"


def _parse_csv(text: str) -> tuple[pd.DataFrame, float]:
    header, _, body = text.partition("-END HEADER-")
    elevation = float("nan")
    for line in header.splitlines():
        if "elevation" in line.lower() and "=" in line:
            try:
                elevation = float(line.split("=")[1].split()[0])
            except (ValueError, IndexError):
                pass
    df = pd.read_csv(io.StringIO(body.strip()))
    df["date"] = pd.to_datetime(df["YEAR"].astype(str), format="%Y") + pd.to_timedelta(df["DOY"] - 1, unit="D")
    df = df.drop(columns=["YEAR", "DOY"]).set_index("date")
    df = df.replace(-999.0, np.nan)
    return df, elevation


def fetch_daily(lat: float, lon: float, end: dt.date | None = None, use_cache: bool = True,
                timeout: int = 120) -> tuple[pd.DataFrame, float, str]:
    """Retorna (dados diários, elevação em m, origem) para um ponto.

    A origem é "API NASA POWER" ou "cache local". Em caso de falha de rede, usa o cache se existir.
    """
    lat, lon = round(lat, 2), round(lon, 2)
    path = _cache_path(lat, lon)
    if use_cache and path.exists():
        df, elev = _parse_csv(path.read_text())
        return df, elev, f"cache local ({path.name})"

    end = end or (dt.date.today() - dt.timedelta(days=5))
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": START_DATE.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "CSV",
    }
    try:
        resp = requests.get(API_URL, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        if path.exists():
            df, elev = _parse_csv(path.read_text())
            return df, elev, f"cache local ({path.name}) — API indisponível"
        raise
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text)
    df, elev = _parse_csv(resp.text)
    return df, elev, "API NASA POWER"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove dias finais sem dados e interpola lacunas curtas (até 3 dias)."""
    df = df.copy()
    essential = ["T2M_MAX", "T2M_MIN", "PRECTOTCORR"]
    last_valid = df[essential].dropna().index.max()
    df = df.loc[:last_valid]
    df = df.interpolate(limit=3, limit_direction="both")
    df["PRECTOTCORR"] = df["PRECTOTCORR"].clip(lower=0)
    return df


def extraterrestrial_radiation(lat: float, doy: np.ndarray) -> np.ndarray:
    """Ra (MJ/m²/dia), FAO-56 eq. 21."""
    phi = math.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )


def et0_penman_monteith(df: pd.DataFrame, lat: float, elevation: float) -> pd.Series:
    """ET0 diária (mm/dia) pelo método FAO-56 Penman-Monteith.

    Quando radiação, umidade ou vento estão ausentes, usa Hargreaves-Samani (FAO-56 eq. 52).
    """
    z = 0.0 if np.isnan(elevation) else elevation
    tmax, tmin = df["T2M_MAX"], df["T2M_MIN"]
    tmean = (tmax + tmin) / 2
    doy = df.index.dayofyear.values
    ra = extraterrestrial_radiation(lat, doy)

    p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26
    gamma = 0.000665 * p
    es_tmax = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    es_tmin = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    es = (es_tmax + es_tmin) / 2
    ea = es * df["RH2M"] / 100
    delta = 4098 * (0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2
    rs = df["ALLSKY_SFC_SW_DWN"]
    rso = (0.75 + 2e-5 * z) * ra
    rns = 0.77 * rs
    sigma = 4.903e-9
    rnl = sigma * (((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2) * (0.34 - 0.14 * np.sqrt(ea.clip(lower=0))) * (
        1.35 * (rs / rso).clip(upper=1.0) - 0.35)
    rn = rns - rnl
    u2 = df["WS2M"]
    pm = (0.408 * delta * rn + gamma * (900 / (tmean + 273)) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))

    hargreaves = 0.0023 * 0.408 * ra * (tmean + 17.8) * np.sqrt((tmax - tmin).clip(lower=0))
    et0 = pm.where(pm.notna(), hargreaves)
    return et0.clip(lower=0)


def prepare(lat: float, lon: float, use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """Baixa, limpa e acrescenta ET0. Retorna (df, metadados)."""
    raw, elev, origin = fetch_daily(lat, lon, use_cache=use_cache)
    df = clean(raw)
    df["ET0"] = et0_penman_monteith(df, lat, elev)
    meta = {
        "origem": origin,
        "elevacao_m": elev,
        "inicio": df.index.min().date(),
        "fim": df.index.max().date(),
        "lat": round(lat, 2),
        "lon": round(lon, 2),
    }
    return df, meta


def power_cell_bounds(lat: float, lon: float) -> tuple[float, float, float, float]:
    """Limites aproximados da célula MERRA-2 (0,5° lat x 0,625° lon) que contém o ponto."""
    lat0 = round(lat / 0.5) * 0.5
    lon0 = round(lon / 0.625) * 0.625
    return lat0 - 0.25, lat0 + 0.25, lon0 - 0.3125, lon0 + 0.3125
