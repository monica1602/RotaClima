"""Indicadores de diagnóstico climático e ambiental a partir da série NASA POWER."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def complete_years(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["PRECTOTCORR"].groupby(df.index.year).count()
    full = counts[counts >= 360].index
    return df[df.index.year.isin(full)]


def annual_rain_mean(df: pd.DataFrame) -> float:
    full = complete_years(df)
    return float(full["PRECTOTCORR"].groupby(full.index.year).sum().mean())


def monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    full = complete_years(df)
    monthly = full.resample("MS").agg({
        "PRECTOTCORR": "sum", "ET0": "sum", "T2M": "mean", "T2M_MAX": "mean", "T2M_MIN": "mean",
        "RH2M": "mean", "ALLSKY_SFC_SW_DWN": "mean", "GWETROOT": "mean",
    })
    clim = monthly.groupby(monthly.index.month).mean()
    clim["P10_chuva"] = monthly["PRECTOTCORR"].groupby(monthly.index.month).quantile(0.10)
    clim["P90_chuva"] = monthly["PRECTOTCORR"].groupby(monthly.index.month).quantile(0.90)
    clim["balanco"] = clim["PRECTOTCORR"] - clim["ET0"]
    clim.index = MONTHS
    return clim


def annual_series(df: pd.DataFrame, heat_threshold: float = 35.0, frost_threshold: float = 2.0) -> pd.DataFrame:
    full = complete_years(df)
    g = full.groupby(full.index.year)
    out = pd.DataFrame({
        "chuva": g["PRECTOTCORR"].sum(),
        "et0": g["ET0"].sum(),
        "temp_media": g["T2M"].mean(),
        "dias_calor": g["T2M_MAX"].apply(lambda s: int((s > heat_threshold).sum())),
        "dias_geada": g["T2M_MIN"].apply(lambda s: int((s <= frost_threshold).sum())),
    })
    out["balanco"] = out["chuva"] - out["et0"]
    return out


def max_dry_spell(rain: pd.Series, threshold: float = 1.0) -> int:
    dry = (rain < threshold).astype(int).to_numpy()
    best = run = 0
    for v in dry:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def wet_season_dry_spells(df: pd.DataFrame, lat: float) -> pd.Series:
    """Maior sequência de dias sem chuva (<1 mm) entre dezembro e fevereiro (jun–ago no hemisfério norte)."""
    months = [12, 1, 2] if lat <= 0 else [6, 7, 8]
    sub = df[df.index.month.isin(months)]
    season_year = np.where(sub.index.month == 12, sub.index.year + 1, sub.index.year) if lat <= 0 else sub.index.year
    return sub["PRECTOTCORR"].groupby(season_year).apply(max_dry_spell)


def recent_anomaly(df: pd.DataFrame, months: int = 24) -> pd.DataFrame:
    monthly = df["PRECTOTCORR"].resample("MS").sum()
    counts = df["PRECTOTCORR"].resample("MS").count()
    monthly = monthly[counts >= 25]
    clim = monthly.groupby(monthly.index.month).mean()
    recent = monthly.iloc[-months:]
    return pd.DataFrame({
        "chuva": recent,
        "normal": [clim[m] for m in recent.index.month],
    }).assign(anomalia=lambda d: d["chuva"] - d["normal"])


def dry_months(clim: pd.DataFrame) -> list[str]:
    """Meses em que a chuva média é menor que metade da ET0 (déficit hídrico climático acentuado)."""
    return [m for m, row in clim.iterrows() if row["PRECTOTCORR"] < 0.5 * row["ET0"]]


def summary_text(df: pd.DataFrame, lat: float) -> list[str]:
    clim = monthly_climatology(df)
    annual = annual_series(df)
    dm = dry_months(clim)
    lines = []
    p10, p90 = annual["chuva"].quantile([0.1, 0.9])
    lines.append(f"Chuva anual média de {annual['chuva'].mean():.0f} mm; em 8 de cada 10 anos ficou entre "
                 f"{p10:.0f} e {p90:.0f} mm (série {annual.index.min()}–{annual.index.max()}).")
    if dm:
        lines.append(f"Meses com déficit hídrico climático acentuado (chuva < 50% da ET0): {', '.join(dm)}. "
                     "Culturas nesse período dependem de água armazenada no solo ou irrigação.")
    wettest = clim["PRECTOTCORR"].idxmax()
    lines.append(f"Mês mais chuvoso: {wettest} ({clim.loc[wettest, 'PRECTOTCORR']:.0f} mm em média).")
    if annual["dias_geada"].mean() >= 1:
        lines.append(f"Em média {annual['dias_geada'].mean():.1f} dias/ano com mínima a 2 m ≤ 2 °C (risco de geada).")
    if annual["dias_calor"].mean() >= 5:
        lines.append(f"Em média {annual['dias_calor'].mean():.0f} dias/ano com máxima acima de 35 °C.")
    if len(annual) >= 10:
        slope = np.polyfit(annual.index, annual["temp_media"], 1)[0] * 10
        lines.append(f"Tendência linear da temperatura média: {slope:+.2f} °C por década "
                     "(indicativa; série curta para inferência climática).")
    return lines
