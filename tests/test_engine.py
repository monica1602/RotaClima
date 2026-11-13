import numpy as np
import pandas as pd

from rotaclima import diagnostics, engine, recommender
from rotaclima.crops import CROPS, transition_rules
from rotaclima.engine import Scenario
from rotaclima.recommender import PRESETS, Economics, Talhao


def make_talhao(lat, lon, irrigado=False):
    return Talhao("Teste", 50, "argilosa", 5.8, 3.0, irrigado, ("soja", "milho_2s", "soja", "milho_2s"),
                  lat, lon)


def test_et0_is_physically_plausible(demo_climate):
    _, _, df, _ = demo_climate
    et0 = df["ET0"].dropna()
    assert len(et0) > 3000
    assert 2.0 < et0.mean() < 7.0
    assert et0.min() >= 0


def test_baseline_scenario_keeps_climate(demo_climate):
    lat, _, df, meta = demo_climate
    out = engine.apply_scenario(df, Scenario(), lat, meta["elevacao_m"])
    pd.testing.assert_series_equal(out["PRECTOTCORR"], df["PRECTOTCORR"])


def test_drought_scenario_reduces_rain_and_raises_temperature(demo_climate):
    lat, _, df, meta = demo_climate
    sc = Scenario(delta_chuva_pct=-30, delta_temp=2.0)
    out = engine.apply_scenario(df, sc, lat, meta["elevacao_m"])
    assert out["PRECTOTCORR"].sum() < 0.75 * df["PRECTOTCORR"].sum()
    assert np.isclose((out["T2M"] - df["T2M"]).mean(), 2.0, atol=0.01)
    assert out["ET0"].mean() > df["ET0"].mean()


def test_simulation_water_satisfaction_bounded(demo_climate):
    lat, _, df, _ = demo_climate
    runner = engine.CropRunner(df, lat, awc=165, irrigated=False, irrig_cap=0)
    runs = runner.runs("soja")
    assert len(runs) >= 15
    assert runs["sat"].between(0, 1).all()
    assert runs["rel_yield"].between(0, 1).all()


def test_drought_lowers_margin(demo_climate):
    lat, lon, df, meta = demo_climate
    t = make_talhao(lat, lon)
    rot = t.rotacao_atual

    def margin(sc):
        clim = engine.apply_scenario(df, sc, lat, meta["elevacao_m"])
        runner = engine.CropRunner(clim, lat, t.awc, t.irrigado, sc.limite_irrigacao_mm)
        ev = recommender.evaluate(t, runner, Economics(), set(CROPS), include=[rot], only_include=True)
        return ev.iloc[0]["margem_media"]

    assert margin(Scenario(delta_chuva_pct=-40, delta_temp=2)) < margin(Scenario())


def test_ranking_and_unique_systems(demo_climate):
    lat, lon, df, _ = demo_climate
    t = make_talhao(lat, lon)
    runner = engine.CropRunner(df, lat, t.awc, t.irrigado, 0)
    ev = recommender.evaluate(t, runner, Economics(), set(CROPS), include=[t.rotacao_atual])
    assert not ev["rot"].duplicated().any()
    for w in PRESETS.values():
        ranked = recommender.rank(ev, w)
        assert ranked["pontuacao"].is_monotonic_decreasing
        top = recommender.top_unique(ranked, 5)
        assert len(top) == 5
        assert not top["rot"].map(recommender.system_key).duplicated().any()
        for c in ["agua", "solo", "fito", "risco", "econ"]:
            assert ranked[c].between(0, 100).all()
        exp = recommender.explain(top.iloc[0], t, ranked[ranked["rot"] == t.rotacao_atual].iloc[0])
        assert exp["pros"] or exp["cons"]


def test_transition_rules_penalize_monoculture():
    hits = transition_rules(CROPS["soja"], CROPS["soja"])
    assert sum(h.efeito for h in hits) < 0
    good = transition_rules(CROPS["crotalaria"], CROPS["milho"])
    assert sum(h.efeito for h in good) > 0


def test_diagnostics_summary(demo_climate):
    lat, _, df, _ = demo_climate
    clim = diagnostics.monthly_climatology(df)
    assert len(clim) == 12
    lines = diagnostics.summary_text(df, lat)
    assert any("Chuva anual" in s for s in lines)
