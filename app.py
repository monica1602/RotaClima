"""RotaClima — protótipo de apoio à decisão para rotação de culturas com dados da NASA.

Execute com:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from rotaclima import diagnostics, engine, modis, nasa_power, presets, recommender
from rotaclima.crops import AWC_MM_PER_M, CROPS, SECOND, SUMMER, TEXTURES, crops_table
from rotaclima.engine import Scenario
from rotaclima.recommender import CRITERIA, PRESETS as WEIGHT_PRESETS, SLOT_LABELS, Economics, Talhao

st.set_page_config(page_title="RotaClima — rotação de culturas com dados da NASA", layout="wide",
                   initial_sidebar_state="expanded")

PALETTE = px.colors.qualitative.Safe
NAME_TO_KEY = {c.nome: k for k, c in CROPS.items()}
SUMMER_NAMES = [CROPS[k].nome for k in SUMMER]
SECOND_NAMES = [CROPS[k].nome for k in SECOND]
ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
PAGES = [
    "1. Propriedade", "2. Diagnóstico ambiental", "3. Mapa dos talhões", "4. Planejamento de rotação",
    "5. Comparação de cenários", "6. Simulador climático", "7. Relatório",
]
DISCLAIMER = ("Protótipo de apoio à decisão. As recomendações não substituem a avaliação de um engenheiro "
              "agrônomo, a análise de solo e o Zoneamento Agrícola de Risco Climático (ZARC).")

ss = st.session_state


# ----------------------------------------------------------------------------- estado

def brl(x: float) -> str:
    return ("R$ " + f"{x:,.0f}").replace(",", ".")


def esc(text: str) -> str:
    """Evita que "R$ ... R$" seja interpretado como fórmula LaTeX pelo Markdown do Streamlit."""
    return text.replace("$", "\\$")


def load_preset(name: str) -> None:
    p = presets.PRESETS[name]
    ss.preset = name
    ss.lat, ss.lon = p["lat"], p["lon"]
    ss.prop_nome = f"Fazenda demonstrativa — {name.split(' — ')[0]}"
    df = pd.DataFrame(presets.talhoes_records(name))
    for col in ["verao_1", "segunda_1", "verao_2", "segunda_2"]:
        df[col] = df[col].map(lambda k: CROPS[k].nome)
    ss.talhoes = df
    ss.talhoes_ver = ss.get("talhoes_ver", 0) + 1
    ss.polygons = {}
    ss.chuva_local = 0.0


def init_state() -> None:
    if "preset" not in ss:
        load_preset(next(iter(presets.PRESETS)))
        ss.summer_shift = 0
        ss.allowed = [k for k in CROPS if k != "pousio"]
        ss.weight_mode = list(WEIGHT_PRESETS)[2]
        ss.custom_weights = dict(WEIGHT_PRESETS[ss.weight_mode])
        ss.eco = pd.DataFrame([{"key": k, "Cultura": c.nome, "Produtividade (t/ha)": c.prod_ref,
                                "Preço (R$/kg)": c.preco, "Custo (R$/ha)": c.custo} for k, c in CROPS.items()])
        ss.eco_ver = 0
        ss.page = PAGES[0]


init_state()


# ----------------------------------------------------------------------------- dados

@st.cache_data(show_spinner="Consultando NASA POWER…", max_entries=16)
def load_climate(lat: float, lon: float) -> tuple[pd.DataFrame, dict]:
    return nasa_power.prepare(lat, lon)


def climate() -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    try:
        return load_climate(round(ss.lat, 2), round(ss.lon, 2))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Não foi possível obter dados da NASA POWER para este ponto ({exc}). "
                 "Verifique a conexão ou escolha uma propriedade demonstrativa (dados em cache).")
        return None, None


def rain_factor() -> float:
    df, _ = climate()
    if df is None or not ss.chuva_local:
        return 1.0
    return float(ss.chuva_local) / diagnostics.annual_rain_mean(df)


def climate_adjusted() -> pd.DataFrame | None:
    df, meta = climate()
    if df is None:
        return None
    f = rain_factor()
    return engine.apply_scenario(df, Scenario(), ss.lat, meta["elevacao_m"], f) if f != 1.0 else df


def talhao_obj(row: pd.Series) -> Talhao:
    rot = tuple(NAME_TO_KEY[row[c]] for c in ["verao_1", "segunda_1", "verao_2", "segunda_2"])
    return Talhao(str(row["nome"]), float(row["area_ha"]), str(row["textura"]), float(row["ph"]),
                  float(row["mo_pct"]), bool(row["irrigado"]), rot, float(row["lat"]), float(row["lon"]))


def talhao_tuple(t: Talhao) -> tuple:
    return (t.nome, t.area_ha, t.textura, t.ph, t.mo_pct, t.irrigado, t.rotacao_atual, t.lat, t.lon)


def eco_tuple() -> tuple:
    return tuple((r["key"], float(r["Produtividade (t/ha)"]), float(r["Preço (R$/kg)"]), float(r["Custo (R$/ha)"]))
                 for _, r in ss.eco.iterrows())


def economics_from(eco_t: tuple) -> Economics:
    return Economics(prod_ref={k: p for k, p, _, _ in eco_t}, preco={k: v for k, _, v, _ in eco_t},
                     custo={k: c for k, _, _, c in eco_t})


@st.cache_data(show_spinner="Simulando balanço hídrico e avaliando rotações…", max_entries=64)
def evaluate_cached(lat: float, lon: float, t_tuple: tuple, allowed: tuple, eco_t: tuple, sc: Scenario,
                    rfactor: float, summer_shift: int, only: tuple | None = None) -> pd.DataFrame:
    df, meta = load_climate(lat, lon)
    clim = engine.apply_scenario(df, sc, lat, meta["elevacao_m"], rfactor)
    t = Talhao(*t_tuple)
    runner = engine.CropRunner(clim, lat, t.awc, t.irrigado, sc.limite_irrigacao_mm, summer_shift)
    include = [t.rotacao_atual] + list(only or [])
    return recommender.evaluate(t, runner, economics_from(eco_t), set(allowed), summer_shift,
                                include=include, only_include=only is not None)


def evaluate_talhao(t: Talhao, sc: Scenario | None = None, only: tuple | None = None) -> pd.DataFrame:
    return evaluate_cached(round(ss.lat, 2), round(ss.lon, 2), talhao_tuple(t), tuple(sorted(ss.allowed)),
                           eco_tuple(), sc or Scenario(), rain_factor(), int(ss.summer_shift), only)


def current_weights() -> dict[str, float]:
    if ss.weight_mode in WEIGHT_PRESETS:
        return WEIGHT_PRESETS[ss.weight_mode]
    return ss.custom_weights


def select_talhao(label: str = "Talhão") -> Talhao | None:
    if ss.talhoes.empty:
        st.warning("Cadastre ao menos um talhão na tela 1.")
        return None
    names = ss.talhoes["nome"].astype(str).tolist()
    cur = ss.get("talhao_atual", names[0])
    idx = names.index(cur) if cur in names else 0
    sel = st.selectbox(label, names, index=idx)
    ss.talhao_atual = sel
    return talhao_obj(ss.talhoes[ss.talhoes["nome"].astype(str) == sel].iloc[0])


def rotation_timeline(rot: tuple[str, ...], delays: tuple[int, int], title: str = "") -> go.Figure:
    year = dt.date.today().year
    rows = []
    for i, key in enumerate(rot):
        crop = CROPS[key]
        y = year if i < 2 else year + 1
        delay = delays[0] if i == 1 else delays[1] if i == 3 else 0
        start = engine.sow_date(crop, y, ss.lat, int(ss.summer_shift)) + pd.Timedelta(days=delay)
        rows.append({"Época": SLOT_LABELS[i], "Cultura": crop.nome, "Início": start,
                     "Fim": start + pd.Timedelta(days=crop.ciclo), "Família": crop.familia})
    df = pd.DataFrame(rows)
    fig = px.timeline(df, x_start="Início", x_end="Fim", y="Época", color="Cultura", text="Cultura",
                      color_discrete_sequence=PALETTE, hover_data=["Família"])
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=30 if title else 10, b=10), showlegend=False,
                      title=title or None)
    return fig


def criteria_bar(row: pd.Series) -> go.Figure:
    vals = [row[k] for k in CRITERIA]
    fig = go.Figure(go.Bar(x=vals, y=list(CRITERIA.values()), orientation="h", marker_color=PALETTE[:5],
                           text=[f"{v:.0f}" for v in vals], textposition="outside"))
    fig.update_layout(height=220, margin=dict(l=10, r=30, t=10, b=10), xaxis=dict(range=[0, 110], title="0–100"))
    return fig


def base_map(center: tuple[float, float], zoom: float, satellite: bool) -> go.Figure:
    fig = go.Figure()
    layout = dict(center=dict(lat=center[0], lon=center[1]), zoom=zoom)
    if satellite:
        layout["style"] = "white-bg"
        layout["layers"] = [{"below": "traces", "sourcetype": "raster", "source": [ESRI_TILES],
                             "sourceattribution": "Esri, Maxar, Earthstar Geographics"}]
    else:
        layout["style"] = "open-street-map"
    fig.update_layout(map=layout, margin=dict(l=0, r=0, t=0, b=0), height=480, showlegend=True,
                      legend=dict(bgcolor="rgba(255,255,255,0.8)"))
    return fig


def talhao_polygon(row: pd.Series) -> list[tuple[float, float]]:
    poly = ss.polygons.get(str(row["nome"]))
    return poly or presets.square_polygon(float(row["lat"]), float(row["lon"]), float(row["area_ha"]))


def add_talhao_layer(fig: go.Figure, values: dict[str, float | str], colors: dict[str, str], fmt) -> None:
    for _, row in ss.talhoes.iterrows():
        name = str(row["nome"])
        poly = talhao_polygon(row)
        val = values.get(name)
        fig.add_trace(go.Scattermap(
            lat=[p[0] for p in poly], lon=[p[1] for p in poly], mode="lines", fill="toself",
            fillcolor=colors.get(name, "rgba(200,200,200,0.4)"), line=dict(color="white", width=2),
            name=f"{name}: {fmt(val)}", hoverinfo="name"))
        fig.add_trace(go.Scattermap(lat=[row["lat"]], lon=[row["lon"]], mode="text", text=[name.split(" — ")[0]],
                                    textfont=dict(color="white", size=13), showlegend=False, hoverinfo="skip"))


def color_scale(values: dict[str, float], scale: str = "RdYlGn", reverse: bool = False,
                vmin: float | None = None, vmax: float | None = None) -> dict[str, str]:
    nums = [v for v in values.values() if v is not None and not np.isnan(v)]
    if not nums:
        return {}
    lo = vmin if vmin is not None else min(nums)
    hi = vmax if vmax is not None else max(nums)
    out = {}
    for k, v in values.items():
        if v is None or np.isnan(v):
            continue
        x = 0.5 if hi == lo else (v - lo) / (hi - lo)
        x = 1 - x if reverse else x
        c = px.colors.sample_colorscale(scale, [min(max(x, 0), 1)])[0]
        out[k] = c.replace("rgb", "rgba").replace(")", ",0.65)")
    return out


# ----------------------------------------------------------------------------- barra lateral

with st.sidebar:
    st.markdown("## RotaClima")
    st.caption("Rotação de culturas orientada por dados da NASA")
    st.radio("Navegação", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.markdown(f"**{ss.prop_nome}**")
    st.caption(f"{len(ss.talhoes)} talhões · {ss.talhoes['area_ha'].sum():.0f} ha · "
               f"lat {ss.lat:.2f}, lon {ss.lon:.2f}")
    _df, _meta = climate()
    if _meta:
        st.caption(f"Clima: NASA POWER ({_meta['inicio']:%Y}–{_meta['fim']:%d/%m/%Y}) · {_meta['origem']}")
    st.divider()
    st.caption(DISCLAIMER)


# ----------------------------------------------------------------------------- 1. propriedade

def page_propriedade() -> None:
    st.title("1. Cadastro da propriedade")
    st.write("Informe a localização e os talhões. Os dados climáticos são obtidos automaticamente da "
             "**NASA POWER** para o ponto da propriedade; os dados de solo vêm do produtor (análise de solo).")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        names = list(presets.PRESETS)
        choice = st.selectbox("Propriedade demonstrativa", names, index=names.index(ss.preset),
                              help="Talhões fictícios; clima e NDVI reais (NASA POWER e MODIS).")
        if choice != ss.preset:
            load_preset(choice)
            st.rerun()
        st.caption(presets.PRESETS[ss.preset]["descricao"])
        ss.prop_nome = st.text_input("Nome da propriedade", ss.prop_nome)
        a, b = st.columns(2)
        lat = a.number_input("Latitude", -60.0, 60.0, float(ss.lat), 0.01, format="%.4f")
        lon = b.number_input("Longitude", -180.0, 180.0, float(ss.lon), 0.01, format="%.4f")
        if (lat, lon) != (ss.lat, ss.lon):
            ss.lat, ss.lon = lat, lon
            st.rerun()
    with c2:
        df, meta = climate()
        with st.container(border=True):
            st.markdown("**Dados climáticos (NASA POWER)**")
            if meta:
                st.write(f"Período: {meta['inicio']:%d/%m/%Y} a {meta['fim']:%d/%m/%Y} · origem: {meta['origem']}")
                st.write(f"Célula MERRA-2 ~0,5° × 0,625° (≈ 55 × 65 km) · elevação média {meta['elevacao_m']:.0f} m")
                power_mean = diagnostics.annual_rain_mean(df)
                ss.chuva_local = st.number_input(
                    "Chuva média anual medida na propriedade (mm) — opcional", 0.0, 5000.0, float(ss.chuva_local), 50.0,
                    help=f"A NASA POWER estima {power_mean:.0f} mm/ano para a célula. Se você tem registros de "
                         "pluviômetro, informe a média para calibrar a chuva (0 = sem calibração).")
                if ss.chuva_local:
                    st.caption(f"Fator de calibração aplicado à chuva: × {rain_factor():.2f}")
        shift = st.slider("Data de semeadura das culturas de verão", -20, 30, int(ss.summer_shift), 5,
                          format="%+d dias", help="Deslocamento em relação a 15/out (referência).")
        ss.summer_shift = shift
        st.caption(f"Semeadura de verão: {(dt.date(2000, 10, 15) + dt.timedelta(days=shift)):%d/%m}")

    st.subheader("Talhões")
    st.caption("Edite diretamente na tabela. As colunas de cultura descrevem a **rotação atual** (2 anos), "
               "usada como referência de comparação.")
    edited = st.data_editor(
        ss.talhoes, key=f"ed_talhoes_{ss.talhoes_ver}", num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "nome": st.column_config.TextColumn("Talhão", required=True),
            "lat": st.column_config.NumberColumn("Lat.", format="%.4f", required=True),
            "lon": st.column_config.NumberColumn("Lon.", format="%.4f", required=True),
            "area_ha": st.column_config.NumberColumn("Área (ha)", min_value=0.1, required=True),
            "textura": st.column_config.SelectboxColumn("Textura", options=TEXTURES, required=True),
            "ph": st.column_config.NumberColumn("pH (H₂O)", min_value=3.5, max_value=9.0, step=0.1, required=True),
            "mo_pct": st.column_config.NumberColumn("MO (%)", min_value=0.0, max_value=15.0, step=0.1, required=True),
            "irrigado": st.column_config.CheckboxColumn("Irrigado"),
            "verao_1": st.column_config.SelectboxColumn("Verão ano 1", options=SUMMER_NAMES, required=True),
            "segunda_1": st.column_config.SelectboxColumn("2ª safra ano 1", options=SECOND_NAMES, required=True),
            "verao_2": st.column_config.SelectboxColumn("Verão ano 2", options=SUMMER_NAMES, required=True),
            "segunda_2": st.column_config.SelectboxColumn("2ª safra ano 2", options=SECOND_NAMES, required=True),
        })
    edited = edited.dropna(subset=["nome", "lat", "lon", "area_ha"]).reset_index(drop=True)
    defaults = {"textura": "média", "ph": 5.5, "mo_pct": 2.0, "irrigado": False, "verao_1": CROPS["soja"].nome,
                "segunda_1": CROPS["milho_2s"].nome, "verao_2": CROPS["soja"].nome, "segunda_2": CROPS["milho_2s"].nome}
    edited = edited.fillna(defaults)
    if not edited.equals(ss.talhoes.reset_index(drop=True)):
        ss.talhoes = edited
        ss.talhoes_ver += 1
        st.rerun()

    with st.expander("Importar limites dos talhões (GeoJSON)"):
        st.caption("Polígonos com a propriedade `nome`. Talhões novos recebem valores de solo padrão para edição.")
        up = st.file_uploader("Arquivo GeoJSON", type=["geojson", "json"])
        if up is not None and st.button("Importar"):
            try:
                import_geojson(json.load(up))
                st.success("Talhões importados.")
                st.rerun()
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"GeoJSON inválido: {exc}")

    with st.expander("Premissas econômicas (valores demonstrativos — substitua pelos seus)"):
        eco = st.data_editor(ss.eco, key=f"ed_eco_{ss.eco_ver}", hide_index=True, width="stretch",
                             disabled=["key", "Cultura"], column_config={"key": None})
        if not eco.equals(ss.eco):
            ss.eco = eco
            ss.eco_ver += 1
            st.rerun()
        st.caption("Margem = produtividade estimada × preço − custo − irrigação (R$ 1,50/mm/ha). "
                   "Coberturas não geram receita direta.")
    with st.expander("Base agronômica utilizada pelo motor"):
        st.dataframe(crops_table().drop(columns=["key"]), hide_index=True, width="stretch")
        st.caption("Kc: FAO-56 (Allen et al., 1998). Ky: FAO-33 (Doorenbos & Kassam, 1979). Água disponível por "
                   f"textura (mm/m): {AWC_MM_PER_M}. Limiares de calor/geada e fatores de perda são heurísticos.")


def import_geojson(gj: dict) -> None:
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    rows = []
    for i, f in enumerate(feats):
        geom = f["geometry"]
        coords = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
        poly = [(float(c[1]), float(c[0])) for c in coords]
        name = str(f.get("properties", {}).get("nome") or f.get("properties", {}).get("name") or f"Talhão {i + 1}")
        lat = float(np.mean([p[0] for p in poly[:-1]]))
        lon = float(np.mean([p[1] for p in poly[:-1]]))
        # área aproximada (fórmula do polígono em coordenadas projetadas localmente)
        x = np.radians([p[1] for p in poly]) * 6_371_000 * np.cos(np.radians(lat))
        y = np.radians([p[0] for p in poly]) * 6_371_000
        area = abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2 / 10_000
        ss.polygons[name] = poly
        rows.append({"nome": name, "lat": round(lat, 5), "lon": round(lon, 5), "area_ha": round(area, 1),
                     "textura": "média", "ph": 5.5, "mo_pct": 2.0, "irrigado": False,
                     "verao_1": CROPS["soja"].nome, "segunda_1": CROPS["milho_2s"].nome,
                     "verao_2": CROPS["soja"].nome, "segunda_2": CROPS["milho_2s"].nome})
    if not rows:
        raise ValueError("nenhum polígono encontrado")
    new = pd.DataFrame(rows)
    keep = ss.talhoes[~ss.talhoes["nome"].isin(new["nome"])]
    ss.talhoes = pd.concat([keep, new], ignore_index=True)
    ss.lat, ss.lon = float(new["lat"].mean()), float(new["lon"].mean())
    ss.talhoes_ver += 1


# ----------------------------------------------------------------------------- 2. diagnóstico

def page_diagnostico() -> None:
    st.title("2. Diagnóstico ambiental")
    df = climate_adjusted()
    if df is None:
        return
    _, meta = climate()
    clim = diagnostics.monthly_climatology(df)
    annual = diagnostics.annual_series(df)
    spells = diagnostics.wet_season_dry_spells(df, ss.lat)

    k = st.columns(6)
    k[0].metric("Chuva anual média", f"{annual['chuva'].mean():.0f} mm")
    k[1].metric("ET0 anual (FAO-56)", f"{annual['et0'].mean():.0f} mm", help="Evapotranspiração de referência "
                "calculada por Penman-Monteith com temperatura, umidade, radiação e vento da NASA POWER.")
    k[2].metric("Balanço chuva − ET0", f"{annual['balanco'].mean():+.0f} mm")
    k[3].metric("Temperatura média", f"{annual['temp_media'].mean():.1f} °C")
    k[4].metric("Dias/ano > 35 °C", f"{annual['dias_calor'].mean():.0f}")
    k[5].metric("Maior veranico (dez–fev)", f"{spells.mean():.0f} dias", help="Média do maior período "
                "consecutivo com chuva < 1 mm entre dezembro e fevereiro. Dados em grade (≈ 55 km) "
                "suavizam a chuva e tendem a subestimar veranicos locais; confira com pluviômetro.")

    with st.container(border=True):
        st.markdown("**O que os dados indicam**")
        for line in diagnostics.summary_text(df, ss.lat):
            st.markdown(f"- {line}")
        if ss.chuva_local:
            st.caption(f"Chuva calibrada com o pluviômetro da propriedade (fator × {rain_factor():.2f}).")

    c1, c2 = st.columns(2)
    with c1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=clim.index, y=clim["PRECTOTCORR"], name="Chuva (mm/mês)", marker_color="#4C8BF5",
                    error_y=dict(type="data", symmetric=False, array=clim["P90_chuva"] - clim["PRECTOTCORR"],
                                 arrayminus=clim["PRECTOTCORR"] - clim["P10_chuva"], color="#9ab"))
        fig.add_scatter(x=clim.index, y=clim["ET0"], name="ET0 (mm/mês)", line=dict(color="#E8710A", width=3))
        fig.add_scatter(x=clim.index, y=clim["T2M_MAX"], name="T máx (°C)", line=dict(color="#C62828", dash="dot"),
                        secondary_y=True)
        fig.add_scatter(x=clim.index, y=clim["T2M_MIN"], name="T mín (°C)", line=dict(color="#1565C0", dash="dot"),
                        secondary_y=True)
        fig.update_layout(title="Climatologia mensal (barras: média e faixa P10–P90)", height=380,
                          legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10))
        fig.update_yaxes(title_text="mm/mês", secondary_y=False)
        fig.update_yaxes(title_text="°C", secondary_y=True)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_bar(x=annual.index, y=annual["chuva"], name="Chuva anual", marker_color="#4C8BF5")
        fig.add_hline(y=annual["chuva"].mean(), line_dash="dash", annotation_text="média")
        fig.add_scatter(x=annual.index, y=annual["et0"], name="ET0 anual", line=dict(color="#E8710A"))
        fig.update_layout(title="Chuva e ET0 por ano", height=380, legend=dict(orientation="h", y=-0.2),
                          margin=dict(t=50, b=10), yaxis_title="mm")
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        an = diagnostics.recent_anomaly(df)
        fig = go.Figure(go.Bar(x=an.index, y=an["anomalia"],
                               marker_color=np.where(an["anomalia"] >= 0, "#2E7D32", "#C62828"),
                               customdata=np.c_[an["chuva"], an["normal"]],
                               hovertemplate="%{x|%b/%Y}<br>chuva %{customdata[0]:.0f} mm<br>normal "
                                             "%{customdata[1]:.0f} mm<extra></extra>"))
        fig.update_layout(title="Anomalia de chuva — últimos 24 meses (mm)", height=330, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with c4:
        g = df["GWETROOT"].dropna()
        doy_mean = g.groupby(g.index.dayofyear).mean()
        last_year = g[g.index.year == g.index.max().year]
        fig = go.Figure()
        fig.add_scatter(x=doy_mean.index, y=doy_mean.values, name="média histórica", line=dict(color="#777"))
        fig.add_scatter(x=last_year.index.dayofyear, y=last_year.values, name=str(g.index.max().year),
                        line=dict(color="#2E7D32", width=3))
        fig.update_layout(title="Umidade do solo na zona radicular (GWETROOT, 0–1)", height=330,
                          xaxis_title="dia do ano", margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Fonte e limitações dos dados"):
        st.markdown(f"""
- **NASA POWER** (Prediction Of Worldwide Energy Resources), API diária, comunidade *AG*, ponto
  ({meta['lat']}, {meta['lon']}). Parâmetros: {', '.join(nasa_power.PARAMETERS)}.
  Documentação: <https://power.larc.nasa.gov/docs/>.
- Resolução espacial grosseira (MERRA-2 ~0,5° × 0,625°; radiação CERES 1°): todos os talhões compartilham o
  mesmo clima. Diferenças entre talhões vêm do solo informado e do NDVI MODIS.
- Precipitação de reanálise pode subestimar ou superestimar a chuva local — use a calibração com pluviômetro.
- GWETROOT é um índice de saturação do modelo MERRA-2, não uma medição de campo. Para umidade do solo por
  satélite, a evolução natural é integrar o SMAP (resolução de ~9–36 km).
""")


# ----------------------------------------------------------------------------- 3. mapa

def soil_vulnerability(row: pd.Series, ndvi_mean: float | None) -> tuple[int, str]:
    score = {"arenosa": 2, "média": 1}.get(row["textura"], 0)
    score += 2 if row["mo_pct"] < 1.5 else 1 if row["mo_pct"] < 2.5 else 0
    score += 1 if row["ph"] < 5.0 else 0
    if ndvi_mean is not None and not np.isnan(ndvi_mean) and ndvi_mean < 0.45:
        score += 1
    return score, "alta" if score >= 4 else "média" if score >= 2 else "baixa"


def page_mapa() -> None:
    st.title("3. Mapa dos talhões")
    st.write("Visualização dos talhões com indicadores de solo (informados) e de vegetação por satélite "
             "(**NDVI MODIS MOD13Q1**, 250 m, composição de 16 dias).")
    ndvi = {str(r["nome"]): modis.cached_ndvi(float(r["lat"]), float(r["lon"])) for _, r in ss.talhoes.iterrows()}
    ndvi_last = {k: (float(v["ndvi"].dropna().iloc[-1]) if v is not None and v["ndvi"].notna().any() else np.nan)
                 for k, v in ndvi.items()}
    ndvi_mean = {k: (float(v["ndvi"].mean()) if v is not None else np.nan) for k, v in ndvi.items()}

    c1, c2 = st.columns([3, 1])
    with c2:
        indicador = st.radio("Indicador", ["NDVI mais recente", "NDVI médio (12 meses)", "Vulnerabilidade do solo",
                                           "Matéria orgânica (%)", "pH", "Água disponível no solo (mm)"])
        satellite = st.toggle("Imagem de satélite", value=True)
        missing = [k for k, v in ndvi.items() if v is None]
        if missing:
            st.info(f"NDVI ainda não baixado para: {', '.join(missing)}")
        if st.button("Baixar/atualizar NDVI MODIS", help="ORNL DAAC MODIS Web Service; ~10–60 s por talhão."):
            prog = st.progress(0.0)
            for i, (_, r) in enumerate(ss.talhoes.iterrows()):
                try:
                    modis.fetch_ndvi_point(float(r["lat"]), float(r["lon"]), use_cache=False)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"{r['nome']}: falha ao consultar MODIS ({exc})")
                prog.progress((i + 1) / len(ss.talhoes))
            st.rerun()

    vul = {}
    for _, r in ss.talhoes.iterrows():
        vul[str(r["nome"])] = soil_vulnerability(r, ndvi_mean.get(str(r["nome"])))
    if indicador.startswith("NDVI"):
        vals = ndvi_last if "recente" in indicador else ndvi_mean
        colors = color_scale(vals, "RdYlGn", vmin=0.1, vmax=0.9)
        fmt = lambda v: "sem dado" if v is None or np.isnan(v) else f"{v:.2f}"  # noqa: E731
    elif indicador.startswith("Vulnerabilidade"):
        vals = {k: float(v[0]) for k, v in vul.items()}
        colors = color_scale(vals, "RdYlGn", reverse=True, vmin=0, vmax=6)
        fmt = lambda v: f"{v:.0f}/6"  # noqa: E731
    else:
        col = {"Matéria orgânica (%)": "mo_pct", "pH": "ph"}.get(indicador)
        if col:
            vals = {str(r["nome"]): float(r[col]) for _, r in ss.talhoes.iterrows()}
        else:
            vals = {str(r["nome"]): AWC_MM_PER_M[r["textura"]] * 1.0 for _, r in ss.talhoes.iterrows()}
        colors = color_scale(vals, "RdYlGn")
        fmt = lambda v: f"{v:.1f}"  # noqa: E731
    with c1:
        fig = base_map((ss.talhoes["lat"].mean(), ss.talhoes["lon"].mean()), 12.6, satellite)
        add_talhao_layer(fig, vals, colors, fmt)
        st.plotly_chart(fig, width="stretch")
        st.caption("Polígonos aproximados (quadrados com a área informada) quando não há GeoJSON importado.")

    rows = []
    for _, r in ss.talhoes.iterrows():
        n = str(r["nome"])
        rows.append({"Talhão": n, "Área (ha)": r["area_ha"], "Textura": r["textura"], "MO (%)": r["mo_pct"],
                     "pH": r["ph"], "Água disponível (mm/m)": AWC_MM_PER_M[r["textura"]],
                     "NDVI recente": ndvi_last[n], "NDVI médio 12 m": ndvi_mean[n],
                     "Vulnerabilidade": f"{vul[n][1]} ({vul[n][0]}/6)"})
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")

    series = [v.assign(Talhão=k) for k, v in ndvi.items() if v is not None]
    if series:
        s = pd.concat(series)
        fig = px.line(s, x="data", y="ndvi", color="Talhão", markers=True, color_discrete_sequence=PALETTE,
                      title="NDVI MODIS por talhão (últimos ~12 meses)")
        fig.add_hrect(y0=0, y1=0.3, fillcolor="#C62828", opacity=0.07, line_width=0,
                      annotation_text="solo exposto / palhada", annotation_position="bottom left")
        fig.add_hrect(y0=0.6, y1=1, fillcolor="#2E7D32", opacity=0.07, line_width=0,
                      annotation_text="vegetação vigorosa", annotation_position="top left")
        fig.update_layout(height=360, yaxis=dict(range=[0, 1]), margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("NDVI baixo na entressafra indica solo descoberto — oportunidade para culturas de cobertura. "
                   "Pixels de 250 m podem misturar talhões vizinhos, estradas e matas.")

    with st.expander("Resolução espacial: talhão × célula NASA POWER"):
        b = nasa_power.power_cell_bounds(ss.lat, ss.lon)
        fig = base_map((ss.lat, ss.lon), 7.5, satellite)
        fig.add_trace(go.Scattermap(lat=[b[0], b[0], b[1], b[1], b[0]], lon=[b[2], b[3], b[3], b[2], b[2]],
                                    mode="lines", line=dict(color="#FFD600", width=3), name="Célula NASA POWER"))
        fig.add_trace(go.Scattermap(lat=ss.talhoes["lat"], lon=ss.talhoes["lon"], mode="markers",
                                    marker=dict(size=9, color="#E53935"), name="Talhões"))
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("Os dados climáticos representam a média de uma área de dezenas de quilômetros.")


# ----------------------------------------------------------------------------- 4. planejamento

def weights_controls() -> dict[str, float]:
    modes = list(WEIGHT_PRESETS) + ["Personalizado"]
    ss.weight_mode = st.radio("Prioridade", modes, index=modes.index(ss.weight_mode), horizontal=True)
    if ss.weight_mode == "Personalizado":
        cols = st.columns(5)
        for col, (k, label) in zip(cols, CRITERIA.items()):
            ss.custom_weights[k] = col.slider(label, 0, 100, int(round(ss.custom_weights.get(k, 0.2) * 100)), 5) / 100
    w = current_weights()
    total = sum(w.values()) or 1
    st.caption("Pesos: " + " · ".join(f"{CRITERIA[k]} {100 * v / total:.0f}%" for k, v in w.items()))
    return w


def render_option(i: int, row: pd.Series, t: Talhao, ref: pd.Series | None) -> None:
    exp = recommender.explain(row, t, ref)
    title = f"#{i + 1} · {row['Rotação']} — pontuação {row['pontuacao']:.0f}/100"
    with st.expander(title, expanded=i == 0):
        st.plotly_chart(rotation_timeline(row["rot"], row["delays"]), width="stretch", key=f"tl_{i}_{row.name}")
        c1, c2, c3 = st.columns([1.1, 1.3, 1.3])
        with c1:
            st.plotly_chart(criteria_bar(row), width="stretch", key=f"cb_{i}_{row.name}")
            st.metric("Margem média (2 anos)", brl(row["margem_media"]) + "/ha",
                      help="Média nos anos históricos; P10–P90 abaixo.")
            st.caption(esc(f"P10–P90: {brl(row['margem_p10'])} a {brl(row['margem_p90'])} · "
                           f"água: {row['demanda_mm']:.0f} mm (+{row['irrig_mm']:.0f} irrigação)"))
        with c2:
            st.markdown("**Por que esta rotação**")
            for p in exp["pros"] or ["—"]:
                st.markdown(esc(f"- {p}"))
        with c3:
            st.markdown("**Riscos e cuidados**")
            for c in exp["cons"] or ["Nenhum risco relevante identificado pelas regras do protótipo."]:
                st.markdown(esc(f"- {c}"))
            if exp["comparacao"]:
                st.markdown("**Comparada à rotação atual**")
                st.markdown(esc(" · ".join(exp["comparacao"])))


def page_planejamento() -> None:
    st.title("4. Planejamento de rotação")
    t = select_talhao()
    if t is None:
        return
    st.caption(f"Solo {t.textura}, pH {t.ph}, MO {t.mo_pct}% · {'irrigado' if t.irrigado else 'sequeiro'} · "
               f"rotação atual: {recommender.rotation_short(t.rotacao_atual)}")
    with st.container(border=True):
        options = [k for k in CROPS if k != "pousio"]
        sel = st.multiselect("Culturas que você aceita cultivar", options, default=[k for k in ss.allowed if k in options],
                             format_func=lambda k: CROPS[k].nome)
        if not any(k in SUMMER for k in sel):
            st.warning("Selecione ao menos uma cultura de verão.")
            return
        ss.allowed = sel
        w = weights_controls()
        c1, c2 = st.columns(2)
        use_min = c1.checkbox("Exigir margem mínima")
        min_margin = c2.number_input("Margem média mínima em 2 anos (R$/ha)", value=3000, step=500, disabled=not use_min)

    res = evaluate_talhao(t)
    if res.empty:
        st.error("Nenhuma rotação viável com as culturas selecionadas.")
        return
    ranked = recommender.rank(res, w)
    ref_rows = ranked[ranked["rot"] == t.rotacao_atual]
    ref = ref_rows.iloc[0] if not ref_rows.empty else None
    pool = ranked[ranked["margem_media"] >= min_margin] if use_min else ranked
    top = recommender.top_unique(pool, 5)
    st.subheader("Alternativas recomendadas")
    st.caption(f"{len(res)} rotações viáveis avaliadas com {int(res['n_anos'].iloc[0])} anos históricos NASA POWER. "
               "Cada alternativa foi simulada em todos os anos da série (balanço hídrico diário FAO-56).")
    if top.empty:
        st.warning("Nenhuma rotação atende à margem mínima.")
    for i, row in top.iterrows():
        render_option(i, row, t, ref)
    if ref is not None:
        pos = int(ranked.index[ranked["rot"] == t.rotacao_atual][0]) + 1
        with st.container(border=True):
            st.markdown(f"**Sua rotação atual** — {ref['Rotação']}: pontuação {ref['pontuacao']:.0f}/100 "
                        f"(posição {pos} de {len(ranked)})")
            cols = st.columns(5)
            for col, (k, label) in zip(cols, CRITERIA.items()):
                col.metric(label, f"{ref[k]:.0f}")
    with st.expander("Tabela completa"):
        show = ranked[["Rotação", "pontuacao", *CRITERIA, "margem_media", "margem_p10", "demanda_mm", "irrig_mm",
                       "deficit_pct", "p_quebra", "cobertura_dias", "familias"]].round(1)
        st.dataframe(show, hide_index=True, width="stretch")
        st.download_button("Baixar CSV", show.to_csv(index=False).encode("utf-8"), f"rotacoes_{t.nome}.csv")
    with st.expander("Como a pontuação é calculada"):
        st.markdown("""
- **Água**: déficit hídrico das culturas comerciais (ETa/ETc no balanço diário) e consumo total de água (ETc + irrigação).
- **Solo**: dias de cobertura (cultura + palhada), palhada, diversidade de famílias botânicas, leguminosas,
  raízes profundas e regras de sucessão. Coberturas com pouca água têm a contribuição reduzida.
- **Fitossanidade**: regras verificáveis de sucessão (mesma espécie/família, hospedeiros comuns de mofo-branco,
  nematoide-de-cisto, giberela no trigo após milho, crotalária para nematoides).
- **Risco climático**: frequência de quebra (>30% de perda) nas culturas comerciais em todos os anos históricos,
  considerando déficit hídrico (Ky FAO-33), calor no florescimento e geada.
- **Economia**: margem média e margem no pior decil (P10), normalizadas entre as alternativas.

Pesos definem o cenário (A, B, C ou personalizado). Valores econômicos são demonstrativos.
""")


# ----------------------------------------------------------------------------- 5. comparação

def strategy_set(t: Talhao, res: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    for name, w in WEIGHT_PRESETS.items():
        out[name.split(" — ")[0] + " · " + name.split(" — ")[1]] = recommender.top_unique(recommender.rank(res, w), 1).iloc[0]
    cur = res[res["rot"] == t.rotacao_atual]
    if not cur.empty:
        out["Atual"] = cur.iloc[0]
    return out


def page_comparacao() -> None:
    st.title("5. Comparação de cenários de prioridade")
    t = select_talhao()
    if t is None:
        return
    res = evaluate_talhao(t)
    if res.empty:
        return
    strategies = strategy_set(t, res)
    st.write("Melhor rotação para cada prioridade (A: água, B: solo, C: equilíbrio) comparada à rotação atual.")

    cols = st.columns(len(strategies))
    for col, (name, row) in zip(cols, strategies.items()):
        with col.container(border=True):
            st.markdown(f"**{name}**")
            st.write(row["Rotação"])
            st.metric("Margem média 2 anos", brl(row["margem_media"]))
            st.caption(f"Água {row['demanda_mm'] + row['irrig_mm']:.0f} mm · cobertura {row['cobertura_dias']:.0f} d · "
                       f"quebra {row['p_quebra']:.0f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        labels = list(CRITERIA.values())
        for i, (name, row) in enumerate(strategies.items()):
            vals = [row[k] for k in CRITERIA]
            fig.add_scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], name=name,
                                 line=dict(color=PALETTE[i], width=3, dash="dot" if name == "Atual" else "solid"))
        fig.update_layout(title="Indicadores (0–100)", polar=dict(radialaxis=dict(range=[0, 100])), height=420,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
    with c2:
        box = pd.concat([row["samples"][["margem"]].assign(Estratégia=name) for name, row in strategies.items()])
        fig = px.box(box, x="Estratégia", y="margem", color="Estratégia", points="all",
                     color_discrete_sequence=PALETTE, title="Margem em 2 anos em cada ano histórico (R$/ha)")
        fig.update_layout(height=420, showlegend=False, yaxis_title="R$/ha")
        st.plotly_chart(fig, width="stretch")

    table = pd.DataFrame([{
        "Estratégia": n, "Rotação": r["Rotação"], "Margem média (R$/ha)": r["margem_media"],
        "Margem P10 (R$/ha)": r["margem_p10"], "Consumo de água (mm)": r["demanda_mm"] + r["irrig_mm"],
        "Déficit comerciais (%)": r["deficit_pct"], "Prob. quebra (%)": r["p_quebra"],
        "Cobertura (dias/730)": r["cobertura_dias"], "Famílias": r["familias"],
    } for n, r in strategies.items()]).round(0)
    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Robustez da recomendação aos pesos")
    st.caption("Sorteamos 500 combinações aleatórias de pesos e contamos quantas vezes cada rotação fica em 1º lugar. "
               "Rotações que vencem com frequência são escolhas robustas mesmo com prioridades incertas.")
    rng = np.random.default_rng(42)
    weights = rng.dirichlet(np.ones(len(CRITERIA)), 500)
    scores = res[list(CRITERIA)].to_numpy() @ weights.T
    winners = res["Rotação"].iloc[scores.argmax(axis=0)].value_counts().head(6)
    fig = px.bar(x=winners.values / 5, y=winners.index, orientation="h", labels={"x": "% das combinações", "y": ""},
                 color_discrete_sequence=[PALETTE[1]])
    fig.update_layout(height=280, yaxis=dict(autorange="reversed"), margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------------------- 6. simulador

SIM_DEFAULTS = {"sim_p": 0, "sim_t": 0.0, "sim_atraso": 0, "sim_veranico": 0, "sim_mes": 1, "sim_calor": 0.0,
                "sim_irrig": 400}
SIM_PRESETS = {
    "Seca moderada": {"sim_p": -20, "sim_t": 1.0},
    "Seca severa": {"sim_p": -40, "sim_t": 2.0, "sim_atraso": 20},
    "Veranico em janeiro": {"sim_veranico": 20, "sim_mes": 1},
    "Aquecimento +2 °C": {"sim_t": 2.0, "sim_calor": 1.5},
}


def set_sim(values: dict) -> None:
    for k, v in {**SIM_DEFAULTS, **values}.items():
        ss[k] = v


def page_simulador() -> None:
    st.title("6. Simulador de cenários climáticos")
    st.info("**Cenário hipotético (análise “e se”)**: as alterações abaixo são aplicadas sobre os anos históricos "
            "da NASA POWER para testar a sensibilidade das rotações. **Não é previsão do tempo** (dias a semanas) "
            "**nem projeção climática** (décadas, modelos CMIP6/NEX-GDDP).")
    for k, v in SIM_DEFAULTS.items():
        ss.setdefault(k, v)
    t = select_talhao()
    if t is None:
        return
    cols = st.columns(len(SIM_PRESETS) + 1)
    for col, (name, vals) in zip(cols, SIM_PRESETS.items()):
        col.button(name, on_click=set_sim, args=(vals,), width="stretch")
    cols[-1].button("Restaurar", on_click=set_sim, args=({},), width="stretch")

    with st.container(border=True):
        a, b, c = st.columns(3)
        a.slider("Variação da chuva (%)", -60, 30, step=5, key="sim_p")
        a.slider("Aumento de temperatura (°C)", 0.0, 4.0, step=0.5, key="sim_t")
        b.slider("Atraso no início das chuvas (dias)", 0, 45, step=5, key="sim_atraso")
        b.slider("Veranico: dias sem chuva", 0, 40, step=5, key="sim_veranico")
        b.selectbox("Mês do veranico", list(range(1, 13)), format_func=lambda m: diagnostics.MONTHS[m - 1], key="sim_mes")
        c.slider("Extremos: +°C nos 10% dias mais quentes", 0.0, 4.0, step=0.5, key="sim_calor")
        c.slider("Água disponível para irrigação (mm/safra)", 0, 800, step=50, key="sim_irrig",
                 disabled=not t.irrigado, help="Aplica-se apenas a talhões irrigados (limite de outorga/reservatório).")
    sc = Scenario(ss.sim_p, ss.sim_t, ss.sim_atraso, ss.sim_veranico, ss.sim_mes, ss.sim_calor, float(ss.sim_irrig))
    base_sc = Scenario(limite_irrigacao_mm=float(ss.sim_irrig))

    res = evaluate_talhao(t, base_sc)
    if res.empty:
        return
    strategies = strategy_set(t, res)
    pool = recommender.top_unique(recommender.rank(res, current_weights()), 12)
    options = {r["Rotação"]: r["rot"] for _, r in pool.iterrows()}
    options.update({r["Rotação"]: r["rot"] for r in strategies.values()})
    default = list(dict.fromkeys(r["Rotação"] for r in strategies.values()))
    chosen = st.multiselect("Rotações comparadas", list(options), default=default)
    if not chosen:
        return
    rots = tuple(options[c] for c in chosen)
    base = evaluate_talhao(t, base_sc, rots).set_index("Rotação").loc[chosen]
    scen = evaluate_talhao(t, sc, rots).set_index("Rotação").loc[chosen]

    st.markdown(f"**Cenário:** {sc.describe()}")
    rows = []
    for name in chosen:
        b_, s_ = base.loc[name], scen.loc[name]
        rows.append({"Rotação": name, "Base": b_["margem_media"], "Cenário": s_["margem_media"],
                     "Base P10": b_["margem_p10"], "Cenário P10": s_["margem_p10"],
                     "Base P90": b_["margem_p90"], "Cenário P90": s_["margem_p90"],
                     "Quebra base": b_["p_quebra"], "Quebra cenário": s_["p_quebra"],
                     "Déficit base": b_["deficit_pct"], "Déficit cenário": s_["deficit_pct"],
                     "Anos com quebra": f"{int(s_['anos_quebra'])}/{int(s_['n_anos'])}"})
    comp = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for label, col, color in [("Clima histórico", "Base", "#90A4AE"), ("Cenário", "Cenário", "#E8710A")]:
            fig.add_bar(name=label, x=comp["Rotação"], y=comp[col], marker_color=color,
                        error_y=dict(type="data", symmetric=False, array=comp[f"{col} P90"] - comp[col],
                                     arrayminus=comp[col] - comp[f"{col} P10"]))
        fig.update_layout(barmode="group", title="Margem em 2 anos (média e faixa P10–P90, R$/ha)", height=420,
                          legend=dict(orientation="h", y=-0.35), margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_bar(name="Clima histórico", x=comp["Rotação"], y=comp["Quebra base"], marker_color="#90A4AE")
        fig.add_bar(name="Cenário", x=comp["Rotação"], y=comp["Quebra cenário"], marker_color="#C62828")
        fig.update_layout(barmode="group", title="Probabilidade de quebra (>30%) por cultura comercial (%)",
                          height=420, legend=dict(orientation="h", y=-0.35), margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.markdown("**Interpretação**")
        if sc.is_baseline():
            st.write("Sem alterações aplicadas: resultados iguais ao clima histórico. Ajuste os controles acima.")
        for _, r in comp.iterrows():
            delta = r["Cenário"] - r["Base"]
            rel = 100 * delta / abs(r["Base"]) if r["Base"] else 0
            st.markdown(esc(f"- **{r['Rotação']}**: margem média {brl(r['Cenário'])}/ha "
                            f"({'+' if delta >= 0 else '−'}{abs(rel):.0f}% vs. histórico); em {r['Anos com quebra']} anos "
                            f"simulados houve ao menos uma quebra de safra; pior decil {brl(r['Cenário P10'])}."))
        best = comp.sort_values("Cenário P10", ascending=False).iloc[0]
        st.success(f"Mais resiliente neste cenário (maior margem no pior decil): **{best['Rotação']}**.")

    detail = []
    for name in chosen:
        s_b, s_s = base.loc[name, "samples"], scen.loc[name, "samples"]
        rot = options[name]
        for i, key in enumerate(rot):
            if CROPS[key].comercial:
                detail.append({"Rotação": name, "Época": SLOT_LABELS[i], "Cultura": CROPS[key].nome,
                               "Água atendida — histórico": s_b[f"sat_{i}"].mean(),
                               "Água atendida — cenário": s_s[f"sat_{i}"].mean()})
    if detail:
        d = pd.DataFrame(detail)
        fig = px.bar(d.melt(id_vars=["Rotação", "Época", "Cultura"], var_name="Clima", value_name="ETa/ETc"),
                     x="Cultura", y="ETa/ETc", color="Clima", barmode="group", facet_col="Rotação", facet_col_wrap=2,
                     color_discrete_sequence=["#90A4AE", "#4C8BF5"], title="Fração da demanda hídrica atendida")
        fig.update_layout(height=320 * ((len(chosen) + 1) // 2))
        fig.update_xaxes(matches=None, showticklabels=True)
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Mapa de risco da propriedade neste cenário")
    st.caption("Probabilidade de quebra das culturas comerciais da rotação atual de cada talhão.")
    risk = {}
    for _, r in ss.talhoes.iterrows():
        tt = talhao_obj(r)
        e = evaluate_talhao(tt, sc, (tt.rotacao_atual,))
        e = e[e["rot"] == tt.rotacao_atual]
        risk[tt.nome] = float(e["p_quebra"].iloc[0]) if not e.empty else np.nan
    fig = base_map((ss.talhoes["lat"].mean(), ss.talhoes["lon"].mean()), 12.6, True)
    add_talhao_layer(fig, risk, color_scale(risk, "RdYlGn", reverse=True, vmin=0, vmax=60),
                     lambda v: "—" if v is None or np.isnan(v) else f"{v:.0f}% de quebra")
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------------------- 7. relatório

def build_report() -> tuple[str, pd.DataFrame]:
    df, meta = climate()
    lines = [f"# Relatório RotaClima — {ss.prop_nome}", "",
             f"Gerado em {dt.date.today():%d/%m/%Y}. Localização: lat {ss.lat:.4f}, lon {ss.lon:.4f}.", "",
             "## Dados utilizados", "",
             f"- NASA POWER (diário, comunidade AG): {meta['inicio']:%d/%m/%Y} a {meta['fim']:%d/%m/%Y} "
             f"({meta['origem']}). Parâmetros: {', '.join(nasa_power.PARAMETERS)}.",
             "- NDVI MODIS MOD13Q1 (ORNL DAAC), quando disponível em cache.",
             "- Solo, rotação atual e premissas econômicas informados pelo usuário (valores demonstrativos nas fazendas-exemplo).",
             f"- Calibração local de chuva: {'× %.2f' % rain_factor() if ss.chuva_local else 'não aplicada'}.", "",
             "## Diagnóstico climático", ""]
    lines += [f"- {x}" for x in diagnostics.summary_text(climate_adjusted(), ss.lat)]
    table = []
    for _, r in ss.talhoes.iterrows():
        t = talhao_obj(r)
        res = evaluate_talhao(t)
        if res.empty:
            continue
        strat = strategy_set(t, res)
        ref = strat.get("Atual")
        lines += ["", f"## {t.nome}", "",
                  f"{t.area_ha:.0f} ha · solo {t.textura} · pH {t.ph} · MO {t.mo_pct}% · "
                  f"{'irrigado' if t.irrigado else 'sequeiro'}", "",
                  "| Estratégia | Rotação (2 anos) | Margem média (R$/ha) | Água (mm) | Cobertura (dias) | Prob. quebra |",
                  "|---|---|---:|---:|---:|---:|"]
        for name, row in strat.items():
            lines.append(f"| {name} | {row['Rotação']} | {row['margem_media']:,.0f} | "
                         f"{row['demanda_mm'] + row['irrig_mm']:.0f} | {row['cobertura_dias']:.0f} | {row['p_quebra']:.0f}% |")
            table.append({"Talhão": t.nome, "Estratégia": name, "Rotação": row["Rotação"],
                          **{CRITERIA[k]: round(row[k], 1) for k in CRITERIA},
                          "Margem média": round(row["margem_media"]), "Margem P10": round(row["margem_p10"]),
                          "Água (mm)": round(row["demanda_mm"] + row["irrig_mm"]),
                          "Cobertura (dias)": round(row["cobertura_dias"]), "Prob. quebra (%)": round(row["p_quebra"], 1)})
        best = strat[list(strat)[2]]
        exp = recommender.explain(best, t, ref)
        lines += ["", f"**Justificativa (estratégia C — {best['Rotação']}):**", ""]
        lines += [f"- Vantagem: {p}" for p in exp["pros"]] + [f"- Atenção: {c}" for c in exp["cons"]]
        if exp["comparacao"]:
            lines += ["", "Comparada à rotação atual: " + "; ".join(exp["comparacao"]) + "."]
    lines += ["", "## Limitações", "",
              "- Clima de reanálise com resolução de dezenas de km; todos os talhões compartilham o mesmo clima.",
              "- Balanço hídrico simplificado (coeficiente único FAO-56, sem escoamento superficial e sem ascensão capilar).",
              "- Perdas por calor e geada usam fatores heurísticos não calibrados localmente.",
              "- Valores econômicos demonstrativos; preços e custos variam entre regiões e safras.",
              "- Regras fitossanitárias simplificadas; não substituem diagnóstico de pragas, doenças e nematoides.", "",
              "## Dados necessários para validação em campo", "",
              "- Análise de solo completa (química, granulometria) e, se possível, análise nematológica.",
              "- Histórico de produtividade por talhão e registros de pluviômetro da propriedade.",
              "- Custos reais de produção e preços regionais.",
              "- Consulta ao ZARC (MAPA) para janelas de semeadura e cultivares indicadas.",
              "- Acompanhamento por engenheiro agrônomo responsável.", "", f"_{DISCLAIMER}_"]
    return "\n".join(lines), pd.DataFrame(table)


def page_relatorio() -> None:
    st.title("7. Relatório")
    st.write("Resumo das estratégias por talhão, indicadores utilizados e dados necessários para validação em campo.")
    if climate()[0] is None:
        return
    report, table = build_report()
    c1, c2 = st.columns(2)
    c1.download_button("Baixar relatório (Markdown)", report.encode("utf-8"), "relatorio_rotaclima.md", width="stretch")
    c2.download_button("Baixar indicadores (CSV)", table.to_csv(index=False).encode("utf-8"),
                       "indicadores_rotaclima.csv", width="stretch")
    with st.container(border=True):
        st.markdown(report)


{
    PAGES[0]: page_propriedade, PAGES[1]: page_diagnostico, PAGES[2]: page_mapa, PAGES[3]: page_planejamento,
    PAGES[4]: page_comparacao, PAGES[5]: page_simulador, PAGES[6]: page_relatorio,
}[ss.page]()
