"""Balanço hídrico diário da cultura (FAO-56, coeficiente único) e cenários climáticos hipotéticos."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .crops import CROPS, VERAO, Crop
from .nasa_power import et0_penman_monteith

FROST_TMIN = 2.0      # °C a 2 m; geada na superfície pode ocorrer com o ar a 2 m acima de 0 °C
LOSS_THRESHOLD = 0.30  # perda de produtividade considerada "quebra de safra" para o indicador de risco
SUMMER_SOW = (10, 15)
MIN_GAP_DAYS = 7
MAX_SECOND_DELAY = 30


@dataclass(frozen=True)
class Scenario:
    """Cenário hipotético (análise "e se"), aplicado sobre a série histórica. Não é previsão nem projeção."""
    delta_chuva_pct: float = 0.0
    delta_temp: float = 0.0
    atraso_chuva_dias: int = 0
    veranico_dias: int = 0
    veranico_mes: int = 1
    calor_extremo: float = 0.0
    limite_irrigacao_mm: float = 400.0

    def is_baseline(self) -> bool:
        return (self.delta_chuva_pct == 0 and self.delta_temp == 0 and self.atraso_chuva_dias == 0
                and self.veranico_dias == 0 and self.calor_extremo == 0)

    def describe(self) -> str:
        parts = []
        if self.delta_chuva_pct:
            parts.append(f"chuva {self.delta_chuva_pct:+.0f}%")
        if self.delta_temp:
            parts.append(f"temperatura {self.delta_temp:+.1f} °C")
        if self.atraso_chuva_dias:
            parts.append(f"início das chuvas {self.atraso_chuva_dias} dias mais tarde")
        if self.veranico_dias:
            parts.append(f"veranico de {self.veranico_dias} dias no mês {self.veranico_mes}")
        if self.calor_extremo:
            parts.append(f"dias mais quentes {self.calor_extremo:+.1f} °C")
        return ", ".join(parts) if parts else "clima histórico (sem alterações)"


def apply_scenario(df: pd.DataFrame, sc: Scenario, lat: float, elevation: float,
                   rain_factor: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    rain = out["PRECTOTCORR"].to_numpy(copy=True) * rain_factor * (1 + sc.delta_chuva_pct / 100)
    idx = out.index
    south = lat <= 0

    if sc.atraso_chuva_dias > 0:
        start_month = 8 if south else 2
        for year in sorted(set(idx.year)):
            start = np.searchsorted(idx.values, np.datetime64(f"{year}-{start_month:02d}-01"))
            end = min(start + 365, len(rain))
            if start >= len(rain) or end - start <= sc.atraso_chuva_dias:
                continue
            seg = rain[start:end].copy()
            n = sc.atraso_chuva_dias
            shifted = np.concatenate([np.zeros(n), seg[:-n]])
            rain[start:end] = shifted

    if sc.veranico_dias > 0:
        mask = (idx.month == sc.veranico_mes) & (idx.day >= 5)
        for year in sorted(set(idx.year)):
            pos = np.flatnonzero(mask & (idx.year == year))
            if len(pos):
                rain[pos[0]:pos[0] + sc.veranico_dias] = 0.0

    out["PRECTOTCORR"] = rain
    if sc.delta_temp or sc.calor_extremo:
        for col in ["T2M", "T2M_MAX", "T2M_MIN"]:
            out[col] = out[col] + sc.delta_temp
        if sc.calor_extremo:
            hot = out["T2M_MAX"] >= out["T2M_MAX"].quantile(0.90)
            out.loc[hot, "T2M_MAX"] += sc.calor_extremo
            out.loc[hot, "T2M"] += sc.calor_extremo / 2
        out["ET0"] = et0_penman_monteith(out, lat, elevation)
    return out


def sow_date(crop: Crop, year: int, lat: float, summer_shift: int = 0) -> pd.Timestamp:
    """Data de semeadura de referência para o ano agrícola que começa no verão de `year`."""
    if crop.slot == VERAO:
        d = pd.Timestamp(year, *SUMMER_SOW) + pd.Timedelta(days=summer_shift)
    else:
        mm, dd = (int(x) for x in crop.semeadura.split("-"))
        d = pd.Timestamp(year + 1, mm, dd)
    if lat > 0:
        d += pd.Timedelta(days=182)
    return d


def second_season_delay(prev: Crop, nxt: Crop, summer_shift: int = 0) -> int | None:
    """Atraso (dias) da 2ª safra imposto pela colheita da cultura de verão; None se inviável."""
    prev_end = pd.Timestamp(2001, *SUMMER_SOW) + pd.Timedelta(days=summer_shift + prev.ciclo + MIN_GAP_DAYS)
    mm, dd = (int(x) for x in nxt.semeadura.split("-"))
    default = pd.Timestamp(2002, mm, dd)
    delay = max(0, (prev_end - default).days)
    if delay > MAX_SECOND_DELAY:
        return None
    if nxt.key != "pousio":
        end = default + pd.Timedelta(days=delay + nxt.ciclo)
        next_summer = pd.Timestamp(2002, *SUMMER_SOW) + pd.Timedelta(days=summer_shift)
        if end > next_summer - pd.Timedelta(days=5):
            return None
    return delay


def _kc_curve(crop: Crop, n: int) -> np.ndarray:
    ini, mid, end = crop.kc
    l1, l2, l3 = int(0.15 * n), int(0.25 * n), int(0.35 * n)
    l4 = n - l1 - l2 - l3
    return np.concatenate([
        np.full(l1, ini), np.linspace(ini, mid, l2, endpoint=False),
        np.full(l3, mid), np.linspace(mid, end, l4),
    ])


def simulate(clim: pd.DataFrame, crop: Crop, start: pd.Timestamp, awc: float, irrigated: bool,
             irrig_cap: float) -> dict | None:
    """Balanço hídrico diário de uma cultura a partir de `start`. Retorna None sem dados suficientes."""
    idx = clim.index.values
    i0 = np.searchsorted(idx, np.datetime64(start))
    n = crop.ciclo
    if i0 - 30 < 0 or i0 + n > len(idx):
        return None
    rain = clim["PRECTOTCORR"].to_numpy()
    et0 = clim["ET0"].to_numpy()
    tmax = clim["T2M_MAX"].to_numpy()
    tmin = clim["T2M_MIN"].to_numpy()

    taw0 = awc * 0.6
    dr = 0.5 * taw0
    for j in range(i0 - 30, i0):
        dr = max(dr - rain[j], 0.0)
        ks = 1.0 if dr <= 0.5 * taw0 else max(0.0, (taw0 - dr) / (0.5 * taw0))
        dr = min(dr + 0.5 * et0[j] * ks, taw0)
    frac = dr / taw0

    kc = _kc_curve(crop, n)
    grow = max(1, int(0.4 * n))
    zr = np.minimum(0.3 + (crop.raiz_m - 0.3) * np.arange(n) / grow, crop.raiz_m) if crop.raiz_m > 0.3 else np.full(n, crop.raiz_m)
    etc_tot = eta_tot = irr = 0.0
    dr = frac * awc * zr[0]
    for k in range(n):
        j = i0 + k
        taw = awc * zr[k]
        if k > 0 and zr[k] != zr[k - 1]:
            dr = dr / (awc * zr[k - 1]) * taw
        raw = crop.p * taw
        dr = max(dr - rain[j], 0.0)
        if irrigated and dr > raw and irr < irrig_cap and crop.key != "pousio":
            dose = min(dr, irrig_cap - irr)
            irr += dose
            dr -= dose
        ks = 1.0 if dr <= raw else max(0.0, (taw - dr) / ((1 - crop.p) * taw))
        etc = kc[k] * et0[j]
        eta = ks * etc
        etc_tot += etc
        eta_tot += eta
        dr = min(dr + eta, taw)

    sl = slice(i0, i0 + n)
    mid = slice(i0 + int(0.40 * n), i0 + int(0.75 * n))
    heat_days = int((tmax[mid] > crop.temp_calor).sum())
    frost_cycle = int((tmin[sl] <= FROST_TMIN).sum())
    frost_mid = int((tmin[mid] <= FROST_TMIN).sum())
    dry = rain[mid] < 1.0
    max_dry = 0
    run = 0
    for v in dry:
        run = run + 1 if v else 0
        max_dry = max(max_dry, run)

    sat = eta_tot / etc_tot if etc_tot > 0 else 1.0
    f_water = max(0.0, 1 - crop.ky * (1 - sat))
    f_heat = max(0.7, 1 - 0.01 * heat_days)
    f_frost = max(0.5, 1 - 0.10 * frost_cycle) if crop.sensivel_geada else max(0.6, 1 - 0.05 * frost_mid)
    rel_yield = f_water * f_heat * f_frost if crop.comercial else 0.0
    return {
        "etc": etc_tot, "eta": eta_tot, "deficit": etc_tot - eta_tot, "sat": sat, "irrig": irr,
        "chuva": float(rain[sl].sum()), "dias_calor": heat_days,
        "dias_geada": frost_cycle if crop.sensivel_geada else frost_mid,
        "veranico_max": max_dry, "rel_yield": rel_yield,
        "f_agua": f_water, "f_calor": f_heat, "f_geada": f_frost,
    }


def agricultural_years(clim: pd.DataFrame, lat: float) -> list[int]:
    """Anos agrícolas com dados suficientes para as duas épocas (até ~setembro do ano seguinte)."""
    years = []
    last = clim.index.max()
    for y in range(clim.index.min().year, last.year + 1):
        start = pd.Timestamp(y, *SUMMER_SOW) - pd.Timedelta(days=40)
        end = pd.Timestamp(y + 1, 10, 1)
        if lat > 0:
            start += pd.Timedelta(days=182)
            end += pd.Timedelta(days=182)
        if start >= clim.index.min() and end <= last:
            years.append(y)
    return years


class CropRunner:
    """Executa e memoriza simulações (cultura, atraso, ano) para um talhão e um clima."""

    def __init__(self, clim: pd.DataFrame, lat: float, awc: float, irrigated: bool, irrig_cap: float,
                 summer_shift: int = 0):
        self.clim, self.lat, self.awc = clim, lat, awc
        self.irrigated, self.irrig_cap, self.summer_shift = irrigated, irrig_cap, summer_shift
        self.years = agricultural_years(clim, lat)
        self._cache: dict[tuple[str, int], pd.DataFrame] = {}

    def runs(self, crop_key: str, delay: int = 0) -> pd.DataFrame:
        key = (crop_key, delay)
        if key not in self._cache:
            crop = CROPS[crop_key]
            rows = []
            for y in self.years:
                start = sow_date(crop, y, self.lat, self.summer_shift) + pd.Timedelta(days=delay)
                r = simulate(self.clim, crop, start, self.awc, self.irrigated, self.irrig_cap)
                if r is not None:
                    r["ano"] = y
                    r["semeadura"] = start.date()
                    rows.append(r)
            self._cache[key] = pd.DataFrame(rows).set_index("ano")
        return self._cache[key]


def scenario_dict(sc: Scenario) -> dict:
    return asdict(sc)
